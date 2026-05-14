"""Docker/sim subprocess utilities — build, launch, env strings.

Every function is stateless: same inputs always produce equivalent outputs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dq_nmpc.core.workspace import core_bin, mujoco_root


def sim_env() -> dict[str, str]:
    """Return an environment dict with LD_LIBRARY_PATH set for the simulator."""
    lib_path = str(mujoco_root() / "deps" / "lib")
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}" if existing else lib_path
    return env


def build_sim(cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Build the MuJoCo quadrotor simulator via xmake.

    Returns the completed subprocess.  Raises CalledProcessError on failure.
    """
    if cwd is None:
        cwd = mujoco_root()
    return subprocess.run(
        ["xmake", "build", "-w", "-j"],
        check=True,
        env=sim_env(),
        cwd=str(cwd),
    )


def ensure_sim_built() -> None:
    """Build the simulator binary if it does not already exist."""
    if core_bin().is_file():
        return
    build_sim()


def launch_sim_core(model_path: str | Path) -> subprocess.Popen:
    """Launch quadrotor_sim_core as a subprocess.

    Returns the Popen handle.  The caller is responsible for lifecycle management.
    """
    return subprocess.Popen(
        [str(core_bin()), str(model_path)],
        env=sim_env(),
    )
