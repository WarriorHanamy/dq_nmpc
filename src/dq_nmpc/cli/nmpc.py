"""Zero-logic dispatch for NMPC CLI verbs — dq-run and dq-codegen."""

from __future__ import annotations

import argparse
import logging
import sys


def main_run():
    """CLI entrypoint for dq-run: launch sim core + run NMPC."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from dq_nmpc.core.workspace import model_path
    from dq_nmpc.workflows.run_nmpc import nmpc_loop

    parser = argparse.ArgumentParser(description="DQ NMPC orchestrator")
    parser.add_argument("config", help="Path to nmpc.yaml")
    parser.add_argument("trajectory", help="Path to trajectory.csv")
    parser.add_argument("--no-build", action="store_true", help="Skip acados C code generation")
    parser.add_argument("--max-iter", type=int, default=0, help="Max NMPC iterations (0=unlimited)")
    parser.add_argument("--model", type=str, default=str(model_path()), help="Path to drone.xml")
    args = parser.parse_args()

    nmpc_loop(
        config_path=args.config,
        trajectory_path=args.trajectory,
        flag_build=not args.no_build,
        max_iter=args.max_iter,
    )


def main_codegen():
    """CLI entrypoint for dq-codegen: acados code generation."""
    from dq_nmpc.workflows.codegen import codegen

    if len(sys.argv) < 2:
        print("Usage: dq-codegen <path_to_nmpc.yaml>")
        sys.exit(1)

    codegen(sys.argv[1])
    print("Code generation complete.")
