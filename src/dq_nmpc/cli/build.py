"""Zero-logic dispatch for dq-build-sim — build MuJoCo simulator C++ binaries."""

from __future__ import annotations

import logging


def main_build_sim():
    """CLI entrypoint for dq-build-sim: xmake build the quadrotor simulator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from dq_nmpc.infra.docker_util import build_sim

    result = build_sim()
    if result.returncode == 0:
        print("MuJoCo simulator built successfully")
    else:
        print(f"Build failed (exit code {result.returncode})")
