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
│                              #   NMPCConfig, NMPCParams, ControlCommand,
│                              #   DualQuaternionState, ClassicalState,
│                              #   TrajectoryPoint, ReferenceTrajectory,
│                              #   SHMConfig, DockerConfig
│
├── schemas/                  # Thin re-export shim (backward compat)
│
├── type.py                   # Scalar, Vector type aliases (numpy | casadi)
│
├── core/                     # Stateless primitives — no classes, no mutable state
│   ├── workspace.py          # project_root(), paths for sim binary and model
│   ├── docker_util.py        # build_sim(), launch_sim_core(), sim_env()
│   └── shm_util.py           # wait_for_shm(), cleanup_shm()
│
├── cli/                      # Zero-logic dispatch — parse args, call one function
│   ├── nmpc.py               # dq-run (→ workflows.nmpc_loop), dq-codegen (→ workflows.codegen)
│   └── trajectory.py         # dq-trajectory (→ workflows.generate_trajectory)
│
├── workflows/                # Domain-specific multi-step pipelines
│   ├── run_nmpc.py           # build sim → launch core → wait SHM → NMPC loop → cleanup
│   ├── codegen.py            # load YAML → acados solver codegen
│   └── generate_trajectory.py
│
├── utils/                    # Standalone helpers
│   ├── casadi_helpers.py     # quaternion CasADi math (from old utils.py)
│   └── waypoints.py          # SHAPES, waypoints_for_shape(), make_sfc_box()
│
├── utils.py                  # Re-export shim (backward compat)
│
├── math/                     # Pure math — no acados, no ROS, no SHM
│   ├── quaternion.py         # Quaternion class (numpy, cs.MX, cs.SX backends)
│   ├── dual_quaternion.py    # DualQuaternion class (SE(3) algebra)
│   └── test_smoke.py         # import + construction smoke test
│
├── nmpc/                     # NMPC solver — requires acados
│   ├── dynamics.py           # Quadrotor ODE, DQ kinematics, flatness
│   ├── ocp_setup.py          # OCP definition: cost, constraints, solver options
│   ├── controller.py         # solver(): build & return AcadosOcpSolver
│   ├── runner.py             # SHM-based NMPC runtime loop (run_nmpc)
│   ├── functions.py          # casadi Function factories (dualquat_from_pose, etc.)
│   └── test_smoke.py         # import + model shape smoke test
│
├── trajectory/               # minco-python integration
│   ├── generator.py          # GCOPTER optimize → sample flatness → write CSV
│   ├── loader.py             # read CSV → ReferenceTrajectory
│   └── test_smoke.py         # loader roundtrip smoke test
│
├── ros/                      # ROS 2 adapter layer (optional, Docker-based)
│   ├── nmpc_node.py          # DQnmpcNode
│   ├── planner_node.py       # PlannerNode
│   └── adapters.py           # ROS msg ↔ Pydantic schema conversion
│
└── config/mujoco/default/    # YAML parameter files (nmpc.yaml)
```

### Dependency layers

```
schema  ──(pydantic)──        # single frozen backbone
math    ──(numpy, casadi)──   # no schema, no acados
core    ──(schema)──          # stateless primitives
utils   ──(numpy, casadi)──
│
nmpc    ──(acados, math, schema)──
trajectory ──(minco-python, utils)──
│
workflows ──(core, nmpc, trajectory)──   # chains layers
cli      ──(workflows)──                 # dispatch only
ros      ──(rclpy, optional, Docker)──
```

- `math`, `schema`, `core`, `utils` are importable without acados or minco.
- `nmpc` works only when acados is built and on `PYTHONPATH`.
- `trajectory/` needs minco-python built (CMake + scikit-build-core).
- `ros` is optional — the Docker adapter (`docker/dq_nmpc_ros2.Dockerfile`) provides ROS 2 bridging.

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
# 1. Generate trajectory CSV
uv run dq-trajectory --shape circle --total-time 5.0 --ts 0.03

# 2. acados code generation (first run)
uv run dq-codegen config/mujoco/default/nmpc.yaml

# 3. Run sim core + NMPC
uv run dq-run config/mujoco/default/nmpc.yaml trajectory.csv
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

```
uv run ruff check src/ tests/       # lint
uv run ruff check --fix src/ tests/ # auto-fix
uv run pytest -v                    # run tests (11 tests, no acados needed)
```

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

The `NMPCConfig` schema wraps YAML parameter files with validation. Use `NMPCConfig.from_yaml(path)` to load and `config.to_params_dict()` to produce the dict expected by the solver (`solver(params, flag=True)`).

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
