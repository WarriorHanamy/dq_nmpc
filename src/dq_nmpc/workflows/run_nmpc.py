"""Multi-step pipeline: build sim, launch core, run NMPC real-time loop, cleanup.

Chains core primitives (build, launch, SHM wait) with the NMPC runner.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
from pathlib import Path

from dq_nmpc.infra.docker_util import ensure_sim_built, launch_sim_core
from dq_nmpc.infra.shm_util import cleanup_shm, wait_for_shm
from dq_nmpc.infra.workspace import model_path as _default_model
from dq_nmpc.nmpc import run_nmpc

logger = logging.getLogger(__name__)


def nmpc_loop(
    config_path: str | Path,
    trajectory_path: str | Path,
    *,
    max_iter: int = 0,
    se3_path: str | Path | None = None,
    model_path: str | Path | None = None,
    rerun: bool = False,
):
    """Run the full NMPC pipeline: build sim, launch core, control loop, cleanup.

    @param[in] config_path     Path to nmpc.yaml
    @param[in] trajectory_path Path to trajectory.npz (minco format)
    @param[in] max_iter        Maximum NMPC iterations (0 = unlimited)
    @param[in] se3_path        Path to se3.yaml (SE3 controller gains)
    @param[in] model_path      Path to drone.xml
    """
    ensure_sim_built()

    core_proc: subprocess.Popen | None = None

    def _on_signal(sig: int, _frame):
        logger.info("Received signal %d, shutting down...", sig)
        if core_proc:
            core_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        logger.info("Starting simulator core")
        core_proc = launch_sim_core(str(model_path or _default_model()))

        wait_for_shm()

        run_nmpc(
            config_path=config_path,
            trajectory_path=trajectory_path,
            se3_config_path=se3_path,
            max_iter=max_iter,
            rerun=rerun,
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

        cleanup_shm()
