# AGENTS.md — dq_nmpc

Dual-Quaternion Model Predictive Control for quadrotor UAVs.
Uses acados for fast online NMPC solving, CasADi for symbolic math,
Pydantic for I/O schemas, minco-python for trajectory generation, and
POSIX shared memory for communication with the MuJoCo simulator.
ROS 2 is an optional adapter layer (Docker-based).

---

```
src/dq_nmpc/
├── schema.py                 # SINGLE SOURCE OF TRUTH: all frozen Pydantic models
│                              #   Models: NMPCConfig, OCPParams, PhysicsParams, ControlCommand,
│                              #     DualQuaternionState, ClassicalState,
│                              #     TrajectoryPoint, ReferenceTrajectory,
│                              #     SHMConfig, TrajectoryConfig, Se3Config,
│                              #     OutputPaths (lazy artifact paths)
│                              #   Layout constants:
│                              #     TRAJECTORY_CSV_COLUMNS, csv_column_index()
│
├── type.py                   # Scalar, Vector type aliases (numpy | casadi)
│
├── infra/                    # Infrastructure primitives — no classes, no mutable state
│   ├── workspace.py          # project_root(), paths for sim binary and model
│   ├── docker_util.py        # build_sim(), launch_sim_core(), sim_env()
│   └── shm_util.py           # wait_for_shm(), cleanup_shm()
│
├── cli/                      # Zero-logic dispatch — parse args, call one function
│   ├── build.py              # dq-build-sim (→ infra.build_sim)
│   ├── nmpc.py               # dq-run (→ workflows.nmpc_loop), dq-codegen (→ workflows.codegen)
│   └── trajectory.py         # dq-trajectory (→ workflows.generate_trajectory)
│
├── workflows/                # Domain-specific multi-step pipelines
│   ├── run_nmpc.py           # ensure sim built → launch core → wait SHM → NMPC loop → cleanup
│   ├── codegen.py            # load YAML → acados solver codegen
│   └── generate_trajectory.py
│
├── math/                     # Pure math — no acados, no ROS, no SHM
│   ├── quaternion.py         # Quaternion class (numpy, cs.MX, cs.SX backends)
│   ├── dual_quaternion.py    # DualQuaternion class (SE(3) algebra)
│   ├── dq_algebra.py         # symbolic DQ algebra on raw CasADi arrays
│   ├── dq_functions.py       # CasADi Function factories (dualquat_from_pose, etc.)
│   ├── quat_helpers.py       # quaternion-level CasADi helpers
│   └── polynomial.py         # order-9 polynomial basis for min-snap
│
├── nmpc/                     # NMPC solver — requires acados
│   ├── dynamics.py           # Quadrotor ODE, DQ kinematics, flatness
│   ├── ocp_setup.py          # OCP definition: cost, constraints, solver options
│   ├── planner.py            # Flatness-based reference computation
│   ├── runner.py             # SE3 bootstrap → NMPC runtime loop (run_nmpc)
│   ├── se3_controller.py     # SE(3) geometric controller (Lee et al. 2010)
│   └── drone_visualizer.py   # DroneVisualizer — Rerun live + offline recorder
│
├── minco_trajectory/         # minco-python integration
│   ├── generator.py          # GCOPTER optimize → sample flatness → write CSV
│   ├── loader.py             # read CSV → ReferenceTrajectory
│   ├── waypoints.py          # SHAPES, waypoints_for_shape(), make_sfc_box()
│   └── visualization.py      # Plotly interactive trajectory plots
│
└── config/mujoco/default/    # YAML parameter files (nmpc.yaml)

docker/                       # ROS 2 adapter (Docker-based, was src/dq_nmpc/ros/)
├── ros2_adapter_node.py      # ROS2 adapter: SHM ↔ /odom + /cmd topics
├── entrypoint.sh              # Docker entrypoint: source ROS, launch adapter
└── dq_nmpc_ros2.Dockerfile   # ros:humble-ros-core + pydantic + numpy
```

### Dependency layers

```
schema  ──(pydantic)──              # single frozen backbone
math    ──(numpy, casadi)──         # no schema, no acados
infra   ──(schema)──                # infrastructure primitives
nmpc    ──(acados, math, schema)──
minco_trajectory ──(minco-python)──
│
workflows ──(infra, nmpc, minco_trajectory)──   # chains layers
cli      ──(workflows)──                       # dispatch only
docker   ──(rclpy, optional)──               # ROS 2 adapter
```

- `math`, `schema`, `infra` are importable without acados or minco.
- `nmpc` works only when acados is built and on `PYTHONPATH`.
- `minco_trajectory/` needs minco-python built (CMake + scikit-build-core).
- `docker/` is optional — contains the ROS 2 adapter (`docker/dq_nmpc_ros2.Dockerfile`) that provides ROS 2 bridging via Docker.

---

## Build

### Prerequisites

```bash
uv sync                           # install python deps + build minco-python C++ extension
uv sync --extra dev               # also install pytest, ruff
```

ROS 2 dependencies (`rclpy`, `nav_msgs`, `geometry_msgs`, etc.) are only needed if using the Docker-based ROS adapter.

### acados

```bash
git submodule update --init deps/acados
cmake -B deps/acados/build -S deps/acados \
  -DACADOS_WITH_OSQP=ON -DACADOS_PYTHON=ON
cmake --build deps/acados/build
```

acados env vars required before codegen or runtime:

```bash
export ACADOS_SOURCE_DIR="$(realpath deps/acados)"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:$PYTHONPATH"
```

Run codegen once (produces `c_generated_code/`):

```bash
uv run dq-codegen config/mujoco/default/nmpc.yaml
```

### Simulator

The simulator lives as a submodule at `deps/mujoco_quadrotor/`. Its C++ binaries are built via xmake. The workflow (`workflows.run_nmpc`) handles this automatically, or you can build manually:

```bash
cd deps/mujoco_quadrotor && uv run sim build
```

---

## Run

### Main runtime (no ROS)

```bash
# 1. Generate trajectory CSV + NPZ (uses defaults from trajectory.yaml)
uv run dq-trajectory

# 2. acados code generation (first run)
uv run dq-codegen config/mujoco/default/nmpc.yaml

# 3. Run sim core + NMPC
uv run dq-run config/mujoco/default/nmpc.yaml out/circle/trajectory.npz
```

### ROS 2 adapter (optional, Docker)

```bash
docker build -f docker/dq_nmpc_ros2.Dockerfile -t dq_nmpc_ros2 .
docker run --rm --net=host \
  -v /dev/shm/quadrotor_sim:/dev/shm/quadrotor_sim:rw \
  dq_nmpc_ros2
```

Publishes `/odom` (from SHM state), subscribes to `/cmd` (writes SHM control).

---

## Develop

```bash
uv run ruff check src/ tests/       # lint
uv run ruff check --fix src/ tests/ # auto-fix
uv run pytest -v                    # run all tests (14 tests, no acados needed)
uv run pytest -v -m acados          # include NMPC solvability tests (requires acados)
uv run pytest -v -m "not acados"    # skip acados-dependent tests
```

---

## Testing

- All test files live in ``tests/test_*.py``, discovered by pytest automatically.
- Integration tests that require acados are tagged with ``@pytest.mark.acados``.
- The ``acados`` marker is registered in ``pyproject.toml`` under ``tool.pytest.ini_options.markers``.
- No test files should live under ``src/*/test_smoke.py``; move them to ``tests/`` or delete.

---

## CasADi Function Style

For each CasADi cost/constraint function:

- Define it through a `make_*` factory function.
- Use explicit and descriptive symbolic variable names.
- Keep Python variable names, MX/SX symbol names, and CasADi input/output names aligned.
- Document all inputs and outputs with shape information in the docstring.
- Use meaningful intermediate variable names instead of abbreviations.
- Attach a short `.description` metadata string when the function is exposed externally.
- Return a named `casadi.Function` with named input and output ports.

---

## SHM Interface

| Segment | File                         | Size  | Written By  | Read By      |
| ------- | ---------------------------- | ----- | ----------- | ------------ |
| state   | `/dev/shm/quadrotor_sim/state` | 192 B | sim_core    | nmpc/runner  |
| ctrl    | `/dev/shm/quadrotor_sim/ctrl`  | 64 B  | nmpc/runner | sim_core     |

Synchronization: seqlock. Schema contract: `deps/mujoco_quadrotor/python/quadrotor_sim/shm.py`.

### Coordinate frames

| Frame  | Axes               | Used for                   |
| ------ | ------------------ | -------------------------- |
| World  | ENU (X=East, Y=North, Z=Up) | position, orientation, world velocity |
| Body   | FLU (X=Front, Y=Left, Z=Up)  | linear velocity, angular velocity, thrust, torques |

---

## Pydantic Schema Conventions

Schemas live in `src/dq_nmpc/schema.py`. Every schema has:

- `to_array() -> np.ndarray` — serialize to numpy
- `from_array(arr) -> cls` — deserialize from numpy
- Field-level validation (e.g., `thrust >= 0`, `mass > 0`)

The `NMPCConfig` schema wraps YAML parameter files with validation. Use `NMPCConfig.from_yaml(path)` to load and pass directly to `solver(config, codegen=True)`.

---

## Key Acronyms

| Term | Meaning |
|------|---------|
| NMPC | Nonlinear Model Predictive Control |
| OCP  | Optimal Control Problem |
| DQ   | Dual Quaternion (SE(3) representation) |
| SQP_RTI | Sequential Quadratic Programming, Real-Time Iteration |
| IRK  | Implicit Runge-Kutta (integrator) |
| HPIPM | High-Performance Interior Point Method (QP solver) |
| SHM  | Shared Memory (POSIX mmap) |
| SFC  | Safe Flight Corridor (polytope-based obstacle avoidance) |
