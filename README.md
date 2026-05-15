# Dual-Quaternion Model Predictive Control for Quadrotor

Model Predictive Control based on Dual Quaternions (DQ) for quadrotor UAVs.
Uses acados for fast online NMPC solving, CasADi for symbolic math,
minco-python for feasible trajectory generation, and MuJoCo (via
quadrotor_simulator_mujoco) for physics simulation.

Communication between NMPC and the simulator is via POSIX shared memory
(`/dev/shm/quadrotor_sim/{state,ctrl}`) — no ROS required for the core runtime.
An optional ROS 2 adapter runs in Docker for visualization (`rviz2`) and
external tooling.

![Simulation View](images/dq_code.gif)
![On-board Camera View](images/dq_code_camera.gif)

## Submodules

```bash
git submodule update --init
```

| Submodule               | Path                  | Purpose                              |
| ----------------------- | --------------------- | ------------------------------------ |
| acados                  | `deps/acados`           | NMPC solver                          |
| minco-python            | `deps/minco-python`     | MINCO trajectory optimization + flatness |
| quadrotor_simulator_mujoco | `deps/mujoco_quadrotor` | MuJoCo physics engine + SHM IPC     |

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd dq_nmpc

# 2. Install everything (Python deps + acados + minco-python + simulator)
uv sync

# 3. NMPC code generation (one-time)
uv run dq-codegen config/mujoco/default/nmpc.yaml

# 4. Generate trajectory CSV
uv run dq-trajectory --shape circle --total-time 5.0 --ts 0.03

# 5. Run sim core + NMPC
uv run dq-run config/mujoco/default/nmpc.yaml trajectory.csv
```

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- [xmake](https://xmake.io/) (for simulator C++ binaries)
- CMake, gcc/g++ (for acados)

`uv sync` builds acados from `deps/acados/`, minco-python from `deps/minco-python/`, and installs the simulator from `deps/mujoco_quadrotor/`.

## Architecture

```
                             HOST (Linux)
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  ┌──────────────────┐         ┌─────────────────────────┐  │
  │  │ quadrotor_sim_core │         │  dq_nmpc (Python)       │  │
  │  │ (C++, MuJoCo)     │         │                          │  │
  │  │                   │  SHM    │  orchestrator.py         │  │
  │  │  mj_step() ──writes──►     │    ├─ sim_core process    │  │
  │  │  apply_ctrl() ◄──reads──    │    └─ nmpc/runner.py     │  │
  │  └──────────────────┘         └─────────────────────────┘  │
  │             │                            │                  │
  │             └──────────┬─────────────────┘                  │
  │              /dev/shm/quadrotor_sim/                        │
  │              state (192 B) / ctrl (64 B)                    │
  └────────────────────────────────────────────────────────────┘

  Optional: ROS 2 adapter (Docker)
  ┌────────────────────────────────────────────────────────────┐
  │  docker build -f docker/dq_nmpc_ros2.Dockerfile ...       │
  │  Bridges SHM ↔ /odom + /cmd topics for rviz2, etc.         │
  └────────────────────────────────────────────────────────────┘
```

## Entry Points

| Command           | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `uv run dq-trajectory` | Generate feasible trajectory CSV via minco-python     |
| `uv run dq-codegen`    | Generate acados C code from an NMPC YAML config       |
| `uv run dq-run`        | Orchestrator: build & launch sim + NMPC runner        |

### `dq-trajectory`

```
uv run dq-trajectory --shape hover|line|circle|fig8 \
  --output trajectory.csv --ts 0.03 --total-time 5.0
```

### `dq-codegen`

```
uv run dq-codegen config/mujoco/default/nmpc.yaml
```

### `dq-run`

```
uv run dq-run config/mujoco/default/nmpc.yaml trajectory.csv [--max-iter 1000]
```

## Shared Memory Interface

| Segment | File                         | Size    | Written By     | Read By         |
| ------- | ---------------------------- | ------- | -------------- | --------------- |
| state   | `/dev/shm/quadrotor_sim/state` | 192 B   | sim_core       | nmpc/runner     |
| ctrl    | `/dev/shm/quadrotor_sim/ctrl`  | 64 B    | nmpc/runner    | sim_core        |

Synchronization: seqlock (monotonic sequence counter + memory barriers).
The schema contract lives in `deps/mujoco_quadrotor/python/quadrotor_sim/shm.py`.

## ROS 2 Adapter (optional)

For rviz2 visualization or integration with other ROS nodes:

```bash
docker build -f docker/dq_nmpc_ros2.Dockerfile -t dq_nmpc_ros2 .
docker run --rm --net=host \
  -v /dev/shm/quadrotor_sim:/dev/shm/quadrotor_sim:rw \
  dq_nmpc_ros2
```

Publishes `/odom` (from SHM state), subscribes to `/cmd` (writes SHM control).

## Development

```bash
uv run ruff check src/ tests/       # lint
uv run ruff check --fix src/ tests/ # auto-fix
uv run pytest -v                    # run tests
```

## Codebase Codemap

```
src/dq_nmpc/
├── math/                    # Pure math — Quaternion, DualQuaternion (numpy/casadi)
├── schemas/                 # Pydantic I/O: state, control, trajectory, config
├── nmpc/                    # NMPC solver (acados)
│   ├── dynamics.py           #   Quadrotor ODE, dual-quaternion kinematics
│   ├── controller.py         #   AcadosOcpSolver builder + codegen entrypoint
│   ├── runner.py             #   SHM-based NMPC runtime loop
│   └── functions.py          #   CasADi helper factories
├── trajectory/              # minco-python integration
│   ├── generator.py          #   MINCO optimize → sample flatness → CSV
│   └── loader.py             #   CSV → ReferenceTrajectory
├── orchestrator.py          # Launch sim_core + NMPC, handle lifecycle
├── ros/                     # ROS 2 adapter layer (optional, Docker-based)
└── config/mujoco/default/   # NMPC YAML parameter files
```

### Dependency Layers

```
math  ──(numpy, casadi only, no acados)──
schemas ──(pydantic)──
nmpc  ──(acados, math, schemas)──
trajectory ──(minco-python, schemas)──
├── runner ──(nmpc, trajectory, quadrotor_sim.shm)──
└── orchestrator ──(runner, subprocess)──
ros  ──(rclpy, optional)──
```
