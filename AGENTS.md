# AGENTS.md — dq_nmpc

Dual-Quaternion Model Predictive Control for quadrotor UAVs.  
Uses acados for fast online NMPC solving, CasADi for symbolic math, Pydantic for I/O schemas, and ROS 2 as the runtime middleware.

---

## Codebase Codemap

```
src/dq_nmpc/
├── type.py                  # Scalar, Vector type aliases (numpy | casadi)
├── utils.py                 # YAML loading, quaternion utility functions
│
├── math/                    # Pure math — no acados, no ROS
│   ├── quaternion.py        # Quaternion class (numpy, cs.MX, cs.SX backends)
│   └── dual_quaternion.py   # DualQuaternion class (SE(3) algebra)
│
├── schemas/                 # Pydantic I/O contracts
│   ├── state.py             # ClassicalState(13D), DualQuaternionState(14D)
│   ├── control.py           # ControlCommand (thrust + 3 torques)
│   ├── trajectory.py        # TrajectoryPoint, ReferenceTrajectory
│   └── config.py            # NMPCConfig: validated YAML→params, with to_params_dict()
│
├── nmpc/                    # NMPC solver — requires acados
│   ├── dynamics.py          # Quadrotor ODE (export_model, quadrotorModel),
│   │                         # dual-quaternion Lie-group error/kinematics,
│   │                         # flatness-based trajectory generation
│   ├── ocp_setup.py         # OCP definition: cost, constraints, solver options
│   ├── controller.py        # solver(): build & return AcadosOcpSolver
│   └── functions.py         # casadi Function factories (dualquat_from_pose, etc.)
│
├── ros/                     # ROS 2 compatibility layer (optional, needs rclpy)
│   ├── nmpc_node.py         # DQnmpcNode: subscriber (odom, position_cmd) → solver → publisher (cmd)
│   ├── planner_node.py      # PlannerNode: generates reference trajectories via flatness
│   └── adapters.py          # ROS msg ↔ Pydantic schema conversion
│
├── config/mujoco/default/   # YAML parameter files
├── launch/                  # ROS 2 launch files
└── tests/                   # pytest tests (schemas, math imports)
```

### Dependency layers

```
math  ──(no deps beyond numpy, casadi)──
schemas ──(pydantic)──
nmpc  ──(acados, math, schemas)──
ros   ──(rclpy, nmpc, schemas)──
```

- `math` and `schemas` are importable without acados or ROS.
- `nmpc` works only when acados is built and on `PYTHONPATH`.
- `ros` is optional — install with the `[ros]` extra (see below).

---

## Build

### Prerequisites

```bash
uv sync                           # install python deps (numpy, casadi, pydantic, …)
uv sync --extra dev               # also install pytest, ruff
```

ROS 2 dependencies (`rclpy`, `nav_msgs`, `geometry_msgs`, etc.) are **not** on PyPI and must be available via your ROS 2 environment:

```bash
source /opt/ros/humble/setup.bash
```

### acados

Clone acados as a submodule (or use the existing `deps/acados/`):

```bash
git submodule update --init deps/acados
```

Build it once:

```bash
./build_dq_nmpc.sh mujoco
```

This script:
1. Builds acados from `deps/acados/` (cmake + make, if not already built)
2. Exports `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`, `PYTHONPATH`
3. Runs `src/dq_nmpc/nmpc/controller.py <config_yaml>` to generate C code into `c_generated_code/`
4. (Optional) copies generated code into `$WS` for colcon dq_cpp build

Generated C code lives in `c_generated_code/` (gitignored).

### Mujoco simulator

The simulator lives as a submodule at `deps/mujoco_quadrotor/`. Build it independently via colcon:

```bash
git submodule update --init deps/mujoco_quadrotor
cd deps/mujoco_quadrotor && colcon build --symlink-install
```

---

## Run

### ROS 2 launch (mujoco simulation + NMPC + planner)

```bash
# In a ROS 2 workspace with dq_nmpc and quadrotor_simulator_mujoco built
ros2 launch dq_nmpc controller_dq.launch.py
```

### Direct entrypoints (via `uv run` or installed wheel)

```bash
uv run dq-nmpc      # → dq_nmpc.ros.nmpc_node:main   (not useful without ROS)
uv run dq-planner   # → dq_nmpc.ros.planner_node:main
```

### Code generation only (no ROS)

```bash
python3 src/dq_nmpc/nmpc/controller.py config/mujoco/default/dq_control.yaml
```

---

## Develop

```
uv run ruff check src/ tests/       # lint
uv run ruff check --fix src/ tests/ # auto-fix
uv run pytest -v                    # run tests (11 tests, no acados needed)
```

---

## Pydantic Schema Conventions

Schemas live in `src/dq_nmpc/schemas/`. Every schema has:
- `to_array() -> np.ndarray` — serialize to numpy
- `from_array(arr) -> cls` — deserialize from numpy
- Field-level validation (e.g., `thrust >= 0`, `mass > 0`)

The `NMPCConfig` schema wraps YAML parameter files with validation. Use `NMPCConfig.from_yaml(path)` to load and `config.to_params_dict()` to produce the dict expected by the solver (`solver(params, flag=True)`).

ROS msg conversion lives in `ros/adapters.py`:
- `odometry_to_classical(msg) -> ClassicalState`
- `position_cmd_to_trajectory(msg) -> ReferenceTrajectory`
- `wrench_from_control(cmd, msg_type) -> Wrench`

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

---

## ROS 2 Topics

| Topic                     | Type                               | Dir        |
|---------------------------|------------------------------------|------------|
| `/quadrotor/odom`         | `nav_msgs/Odometry`               | in         |
| `/quadrotor/position_cmd` | `quadrotor_msgs/PositionCommand`  | in         |
| `/quadrotor/cmd`          | `geometry_msgs/Wrench`            | out        |
| `/quadrotor/desired_frame`| `nav_msgs/Odometry`               | out (viz)  |
| `/quadrotor/desired_path` | `visualization_msgs/Marker`       | out (viz)  |
