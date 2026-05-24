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
│                              #   OutputPaths (lazy artifact paths)
│
├── type.py                   # Scalar, Vector type aliases (numpy | casadi)
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
├── nmpc/                     # NMPC solver — requires acados
│   ├── config/                #   default.yaml, se3.yaml
│   ├── dq_functions.py        # DQ math kernels (_expr + _ca_func pattern)
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
│ ├── generator.py          # GCOPTER optimize → NPZ + HTML visualization
│   ├── loader.py             # read NPZ → Trajectory7
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
infra   ──(schema)──                # infrastructure primitives
nmpc    ──(acados, schema)──        # quadrotor physics, OCP, flatness
minco_trajectory ──(minco-python)── # trajectory tools
│
workflows ──(infra, nmpc, minco_trajectory)──   # chains layers
cli      ──(workflows)──                       # dispatch only
docker   ──(rclpy, optional)──               # ROS 2 adapter
```

- `infra`, `schema` are importable without acados or minco.
- `nmpc` works only when acados is built and on `PYTHONPATH`.
- `minco_trajectory/` needs minco-python built (CMake + scikit-build-core).
- `docker/` is optional — contains the ROS 2 adapter (`docker/dq_nmpc_ros2.Dockerfile`) that provides ROS 2 bridging via Docker.

---

## Module Boundaries

Every directory under `src/dq_nmpc/` that contains Python code is a
module.  Each module has a single public surface: its `__init__.py`.

### Public surface

`__init__.py` is the module facade.  It may **only**:

- re-export public symbols from private internals
- define `__all__`
- contain a module docstring

It must **never**:

- import acados, minco, or any optional dependency at module level
- open files, read environment, or configure logging
- instantiate objects, register hooks, or spawn threads
- run `try/except ImportError` that silently swallows missing deps

```python
# nmpc/__init__.py — correct
"""NMPC solver: DQ math, flatness, OCP, runtime."""

from ._dq_functions import (
    dualquat_from_pose_ca_func,
    dualquat_kinematics_expr,
)
from ._flatness import make_flatness_casadi
from ._reference import dense_ref_from_minco

__all__ = [
    "dualquat_from_pose_ca_func",
    "dualquat_kinematics_expr",
    "make_flatness_casadi",
    "dense_ref_from_minco",
    "solver",
    "run_nmpc",
]

def __getattr__(name: str):
    if name == "solver":
        from ._ocp_setup import solver
        return solver
    if name == "run_nmpc":
        from ._runner import run_nmpc
        return run_nmpc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Private internals

Every implementation file inside a module is private.  Its name starts
with `_`.

```
nmpc/
  __init__.py          # facade
  _dq_functions.py     # private: DQ algebra
  _dynamics.py         # private: acados model
  _flatness.py         # private: differential flatness
  _ocp_setup.py        # private: solver factory
  _reference.py        # private: trajectory → reference
  _runner.py           # private: NMPC runtime loop
  config/              # YAML files (not Python — no prefix)
    default.yaml
```

A private file may define a public name (e.g. `run_nmpc` in
`_runner.py`).  The name becomes public only when the facade
re-exports it.

### Imports

Three rules cover every situation.

**1. Cross-module: absolute, through facade.**

```python
# Correct
from dq_nmpc.nmpc import run_nmpc
from dq_nmpc.infra import launch_sim_core, wait_for_shm
from dq_nmpc.schema import NMPCConfig
from dq_nmpc.minco_trajectory import load_trajectory_npz

# Wrong — reaches into private internals
from dq_nmpc.nmpc._runner import run_nmpc
from dq_nmpc.infra._docker import launch_sim_core
```

**2. Within a module: absolute, fully qualified.**

```python
# Inside nmpc/_runner.py — correct
from dq_nmpc.nmpc._dq_functions import position_from_dualquat_ca_func
from dq_nmpc.nmpc._visualizer import DroneVisualizer
from dq_nmpc.nmpc._se3_controller import se3_control

# Inside nmpc/_runner.py — wrong
from ._dq_functions import position_from_dualquat_ca_func
```

**3. `__init__.py` only: relative, to re-export internals.**

```python
# nmpc/__init__.py — the ONE place relative imports are allowed
from ._dq_functions import dualquat_from_pose_ca_func
from ._runner import run_nmpc
```

Summary:

| Context                | Style    | Example                                    |
| ---------------------- | -------- | ------------------------------------------ |
| Cross-module           | absolute | `from dq_nmpc.nmpc import run_nmpc`          |
| Within module (`_*.py`)  | absolute | `from dq_nmpc.nmpc._dq_functions import ...` |
| `__init__.py` re-export  | relative | `from ._runner import run_nmpc`              |

### Entrypoints

A file being importable does not make it executable.
Only functions listed in `pyproject.toml` `[project.scripts]` are
entrypoints.

```toml
[project.scripts]
dq-run = "dq_nmpc.cli._nmpc:main_run"
dq-codegen = "dq_nmpc.cli._nmpc:main_codegen"
dq-nmpc-vis-ref = "dq_nmpc.cli._nmpc:main_vis_ref"
dq-trajectory = "dq_nmpc.cli._trajectory:main"
dq-build-sim = "dq_nmpc.cli._build:main_build_sim"
```

No file outside `cli/` may contain `if __name__ == "__main__":`.

All CLI entrypoints use schematized defaults (`_DEFAULT_*` module-level
constants) so zero-argument invocation is always valid.  Defaults are
derived from known project paths (``src/dq_nmpc/nmpc/config/default.yaml``,
``out/circle/trajectory.npz``, etc.).  Never require the user to specify
paths that have a canonical home.

Entrypoints must print the output artifact path and, if no viewer is
spawned, how the user can inspect the result.  Silent success is a bug.

### What lives where

| Concept                         | Location               | Public name                          |
| ------------------------------- | ---------------------- | ------------------------------------ |
| Pydantic models, type aliases   | `schema.py`, `type.py`     | as-is (top-level, no prefix)         |
| SHM, subprocess, paths          | `infra/`                 | `infra.__init__`                       |
| DQ math (CasADi, numpy)         | `nmpc/_dq_functions.py`  | `nmpc.__init__`                        |
| Flatness, dynamics, OCP, runner | `nmpc/_*.py`             | `nmpc.__init__` (lazy for acados deps) |
| Trajectory gen + I/O (minco)    | `minco_trajectory/_*.py` | `minco_trajectory.__init__`            |
| Multi-step pipelines            | `workflows/_*.py`        | `workflows.__init__` (lazy)            |
| Argument parsing, dispatch      | `cli/_*.py`              | via `pyproject.toml` scripts           |
| YAML config files               | `*/config/*.yaml`        | not imported, read at runtime        |

### Rules

1. One public surface per module: `__init__.py`.
2. Everything else is `_private.py`.
3. All imports are absolute, except `__init__.py` which uses relative to re-export.
4. `__init__.py` has zero side effects.
5. Only `cli/` files are executable.

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

The continuous polynomial coefficients (`.npz`) from `dq-trajectory` are the
primary interchange format.  NMPC reconstructs a ``Trajectory7`` via
``load_trajectory_npz()``, then resamples it at the NMPC control rate through
``dense_ref_from_minco()``, which applies the NMPC's own flatness model.

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
