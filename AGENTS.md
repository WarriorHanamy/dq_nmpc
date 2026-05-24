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
│   ├── dq_functions.py       # _expr functions + _ca_func wrappers + numpy helpers
│   └── polynomial.py         # order-9 polynomial basis for min-snap
│
├── nmpc/                     # NMPC solver — requires acados
│   ├── config/                #   default.yaml, se3.yaml
│   ├── dynamics.py            # Quadrotor ODE, DQ kinematics (acados model)
│   ├── flatness.py            # Differential flatness: flat outputs → body-frame ref
│   ├── ocp_setup.py           # OCP definition: cost, constraints, solver options
│   ├── reference.py           # traj7 → dense ref_params → belt set (via nmpc.flatness)
│   ├── runner.py              # SE3 bootstrap → NMPC runtime loop (run_nmpc)
│   ├── se3_controller.py      # SE(3) geometric controller (Lee et al. 2010)
│   └── drone_visualizer.py    # DroneVisualizer — Rerun live + offline recorder
│
├── minco_trajectory/          # minco-python integration
│   ├── config/                #   default.yaml, default_gcopter.yaml, lbfgs.yaml
│   ├── generator.py           # GCOPTER optimize → sample geometry → write CSV
│   ├── loader.py              # read CSV → ReferenceTrajectory
│   ├── waypoints.py           # SHAPES, waypoints_for_shape(), make_sfc_box()
│   └── visualization.py       # Plotly interactive trajectory plots

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
nmpc    ──(acados, math, schema)──  # quadrotor physics, OCP, flatness
minco_trajectory ──(minco-python)── # trajectory tools
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

## Physical Layers

Both the trajectory generator and NMPC carry their own physical model
of the quadrotor, and these models are allowed to differ by design.

### Trajectory generator — GCOPTER

GCOPTER (inside `deps/minco-python`) uses an embedded CasADi flatness
model as polynomial-optimisation constraints.  This model lives in C++
and is **not exposed** to the Python layer.  Its output is a
`Trajectory7` — a continuous polynomial that describes position and
its first four time derivatives (vel, acc, jerk, snap).

GCOPTER's internal flatness is tuned for **fast, long-horizon
trajectory generation**.  It may be simpler or numerically optimised
differently than the NMPC model; the result is a geometrically
feasible reference, not a dynamically exact one.

### NMPC — `nmpc/flatness.py` + `nmpc/dynamics.py`

The NMPC physical layer consists of two parts:

| File               | Role                                                  |
| ------------------ | ----------------------------------------------------- |
| `nmpc/flatness.py`   | Differential flatness: (pos, vel, acc, jerk, snap, yaw) → (quat, omega, thrust, torque) |
| `nmpc/dynamics.py`   | DQ kinematics ODE: (dq, twist, ctrl) → (dq_dot, twist_dot) — acados model |

The conversion pipeline is:

```
Trajectory7  ──[reference.py: dense_ref_from_minco()]──→  ref_params (N, 18)
              │
              └── uses nmpc.flatness to reinterpret the continuous
                  polynomial geometry in the NMPC's own physical model
```

Key design points:

| Concern                        | Convention                                                                 |
| ------------------------------ | -------------------------------------------------------------------------- |
| Who owns flatness?             | `nmpc/flatness.py` — part of the NMPC physical layer                        |
| Who produces the geometry?     | GCOPTER (`Trajectory7`) — continuous polynomial in time                     |
| Who reinterprets the geometry? | NMPC (`reference.py`), using its own flatness model                        |
| Why two physical models?       | Trajectory generation trades accuracy for speed; NMPC follows a reference  |
|                                | that may be dynamically infeasible and corrects online via OCP optimisation |

The CSV output from `dq-trajectory` contains only **geometric** data
(position and its first four time derivatives) sampled from
`Trajectory7`.  Differential flatness reinterpretation (quaternion,
body angular velocity, thrust, torque) occurs exclusively in
`nmpc/reference.py` when the NMPC loads the `Trajectory7` from the
`.npz` coefficient file — **not** during trajectory generation.

### Simulator — MuJoCo physics

The MuJoCo simulator (`deps/mujoco_quadrotor/`) implements rigid-body
dynamics that may differ from the NMPC ODE model.  NMPC and the
simulator communicate via shared memory (SHM):

```
NMPC (runner)  ──ctrl (64 B)──→  simulator (core)
NMPC (runner)  ←──state (192 B)──  simulator (core)
```

**The NMPC physical model should stay reasonably close to the
simulator's dynamics** so the open-loop prediction horizon remains
useful.  The same physical parameters (mass, inertia, gravity) flow
from `nmpc.yaml` into both: acados codegen and the simulator config.

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
uv run dq-codegen src/dq_nmpc/nmpc/config/default.yaml
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
# 1. Generate trajectory (CSV + NPZ + interactive HTML visualization)
uv run dq-trajectory --shape circle

# 2. acados code generation (first run)
uv run dq-codegen config/mujoco/default/nmpc.yaml

# 3. Run sim core + NMPC
uv run dq-run src/dq_nmpc/nmpc/config/default.yaml out/circle/trajectory.npz
```

Output artifacts for step 1 are written to `out/{shape}/`:
- `trajectory.csv` — sampled geometry (t, x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz, sx, sy, sz)
- `trajectory.npz` — polynomial coefficients (continuous representation, consumed by NMPC)
- `trajectory.html` — interactive Plotly visualization (opens automatically)

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

## CasADi Architecture

### Expression-first design

All CasADi math follows a strict two-layer pattern:

1. **`_expr` functions** — mathematical source of truth
2. **`_ca_func` factories** — thin wrappers that call `_expr`

No duplication: `_ca_func` factories must call `_expr`, never reimplement logic.

### `_expr` expression functions

Expression functions must:
- Accept CasADi expressions (`ca.MX`, `ca.SX`, or `ca.DM`) and return CasADi expressions
- NOT create symbolic variables internally
- NOT return `ca.Function`
- Use the `_expr` suffix (e.g. `dualquat_mul_conj_expr`, `log_map_dualquat_expr`)
- Include type hints with `CasadiVec` / `CasadiMat` aliases from `type.py`
- Include a docstring with mathematical meaning, I/O shapes, and assumptions
  (e.g. unit quaternion, unit dual quaternion)

```python
from dq_nmpc.type import CasadiVec

def dualquat_mul_conj_expr(qd: CasadiVec, q: CasadiVec) -> CasadiVec:
    """DQ multiplicative error: conj(qd) * q via 8x8 Hamiltonian matrix.

    @param[in] qd  (8,) desired dual quaternion [unit DQ]
    @param[in] q   (8,) current dual quaternion [unit DQ]
    @return        (8,) DQ error [unit DQ]
    """
    ...
```

### `_ca_func` wrappers

Function wrappers must:
- Only create symbolic inputs (via `MX.sym` or `SX.sym`)
- Call the corresponding `_expr` function
- Wrap result into `ca.Function` with named inputs/outputs
- Never duplicate mathematical logic from `_expr`
- Accept `symbolic_type: Literal["MX", "SX"] = "MX"`
- Use `_ca_func` suffix (e.g. `dualquat_kinematics_ca_func`, `inertial_to_body_rotation_ca_func`)
- Attach a short `.description` metadata string

```python
from typing import Literal

def dualquat_kinematics_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: (dualquat(8,1), twist(6,1)) -> dq_dot(8,1)."""
    ...
```

### MX vs SX

- Prefer **MX** for NLP modeling, vector/matrix expressions, and acados OCP formulation
- Use **SX** for small standalone functions or scalar-expanded code generation

### Import convention

Always use `ca.Function`, never bare `Function` from `casadi`.

### Helper `_expr` functions

Factor repeated math blocks into helper `_expr` functions:
- `quat_conjugate_expr`
- `dual_quat_conjugate_expr`
- `quat_left_matrix_expr`
- `error_dual_expr`

### Comment style

Comments explain mathematical meaning, not obvious code mechanics.
Inline `#` comments: only for non-obvious physical or mathematical reasoning.

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
