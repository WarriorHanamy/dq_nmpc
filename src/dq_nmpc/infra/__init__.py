"""Infrastructure primitives — paths, subprocess, SHM helpers.

No classes. No mutable module-level state. Every function returns a value.
"""

from dq_nmpc.infra.docker_util import build_sim, launch_sim_core, sim_env
from dq_nmpc.infra.shm_util import cleanup_shm, wait_for_shm
from dq_nmpc.infra.workspace import (
    build_dir,
    core_bin,
    model_path,
    mujoco_root,
    project_root,
)

__all__ = [
    "build_dir",
    "build_sim",
    "cleanup_shm",
    "core_bin",
    "launch_sim_core",
    "model_path",
    "mujoco_root",
    "project_root",
    "sim_env",
    "wait_for_shm",
]
