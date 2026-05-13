"""Orchestrator: build sim, start core, run NMPC, clean up.

Usage: dq-run <nmpc.yaml> <trajectory.csv> [--no-build]
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_ROOT = PROJECT_ROOT / "deps" / "mujoco_quadrotor"
BUILD_DIR = MUJOCO_ROOT / "build_standalone"
CORE_BIN = BUILD_DIR / "quadrotor_sim_core"
MODEL_PATH = MUJOCO_ROOT / "deps" / "model" / "mujoco" / "drone.xml"

SHM_STATE = "/dev/shm/quadrotor_sim/state"
SHM_CTRL = "/dev/shm/quadrotor_sim/ctrl"


def _sim_env():
    env = os.environ.copy()
    lib_path = str(MUJOCO_ROOT / "deps" / "lib")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}" if existing else lib_path
    return env


def _ensure_sim_built():
    if CORE_BIN.is_file():
        logger.info("sim: binaries already built")
        return
    logger.info("sim: building via xmake...")
    subprocess.run(
        ["xmake", "build", "-w", "-j"],
        check=True,
        env=_sim_env(),
        cwd=str(MUJOCO_ROOT),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="DQ NMPC orchestrator")
    parser.add_argument("config", help="Path to nmpc.yaml")
    parser.add_argument("trajectory", help="Path to trajectory.csv")
    parser.add_argument("--no-build", action="store_true", help="Skip acados C code generation")
    parser.add_argument("--max-iter", type=int, default=0, help="Max NMPC iterations (0=unlimited)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to drone.xml")
    args = parser.parse_args()

    _ensure_sim_built()

    core_proc: subprocess.Popen | None = None

    def _on_signal(sig: int, _frame):
        logger.info("Received signal %d, shutting down...", sig)
        if core_proc:
            core_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        logger.info("Starting simulator core (%s)", args.model)
        core_proc = subprocess.Popen(
            [str(CORE_BIN), args.model],
            env=_sim_env(),
        )

        logger.info("Waiting for SHM segments...")
        max_wait = 5.0
        t0 = time.time()
        while time.time() - t0 < max_wait:
            if os.path.exists(SHM_STATE) and os.path.exists(SHM_CTRL):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("SHM segments not created by sim core")

        from dq_nmpc.nmpc.runner import run_nmpc

        run_nmpc(
            config_path=args.config,
            trajectory_path=args.trajectory,
            flag_build=not args.no_build,
            max_iter=args.max_iter,
        )

    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")
    finally:
        if core_proc:
            logger.info("Stopping simulator core...")
            core_proc.terminate()
            try:
                core_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                core_proc.kill()
                core_proc.wait()
            logger.info("Simulator core stopped")

        # Clean up SHM segments
        for f in (SHM_STATE, SHM_CTRL):
            try:
                os.unlink(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()
