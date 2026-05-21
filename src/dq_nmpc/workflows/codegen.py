"""Acados code generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from dq_nmpc.nmpc.ocp_setup import solver
from dq_nmpc.schema import NMPCConfig

logger = logging.getLogger(__name__)


def codegen(config_path: str | Path) -> None:
    """Load NMPC config and run acados code generation.

    @param[in] config_path  Path to nmpc.yaml
    """
    config_path = Path(config_path)
    config = NMPCConfig.from_yaml(config_path)
    ocp_solver, ocp = solver(config, codegen=True)
    logger.info("Code generation complete for %s", config_path.name)
