"""Zero-logic dispatch for NMPC CLI verbs — dq-run and dq-codegen."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _setup_acados_env() -> None:
    """Set acados environment variables and build acados if missing."""
    project = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project))
    from _acados_hook import build_acados  # noqa: E402

    build_acados(project)
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(project / "deps" / "acados"))
    lib_dir = str(project / "_acados_build" / "install" / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir


def main_run():
    """CLI entrypoint for dq-run: launch sim core + run NMPC."""
    _setup_acados_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from dq_nmpc.infra.workspace import model_path, project_root
    from dq_nmpc.workflows.run_nmpc import nmpc_loop

    _root = project_root()

    def _resolve(p: str) -> str:
        """Resolve path relative to CWD, project root, or src/dq_nmpc."""
        path = Path(p)
        if path.exists():
            return str(path)
        for base in (_root, _root / "src" / "dq_nmpc"):
            resolved = base / path
            if resolved.exists():
                return str(resolved)
        return str(path)

    parser = argparse.ArgumentParser(description="DQ NMPC orchestrator")
    parser.add_argument("config", help="Path to nmpc.yaml")
    parser.add_argument("trajectory", help="Path to trajectory.npz")
    parser.add_argument("--no-build", action="store_true", help="Skip acados C code generation")
    parser.add_argument("--max-iter", type=int, default=0, help="Max NMPC iterations (0=unlimited)")
    parser.add_argument("--model", type=str, default=str(model_path()), help="Path to drone.xml")
    parser.add_argument(
        "--se3-config", type=str, default=None, help="Path to se3.yaml (SE3 controller gains)"
    )
    args = parser.parse_args()

    nmpc_loop(
        config_path=_resolve(args.config),
        trajectory_path=_resolve(args.trajectory),
        flag_build=not args.no_build,
        max_iter=args.max_iter,
        se3_path=_resolve(args.se3_config) if args.se3_config else None,
        model_path=args.model,
    )


_DEFAULT_CONFIG = "src/dq_nmpc/config/mujoco/default/nmpc.yaml"


def main_codegen():
    """CLI entrypoint for dq-codegen: acados code generation."""
    _setup_acados_env()
    from dq_nmpc.workflows.codegen import codegen

    config_path = _DEFAULT_CONFIG if len(sys.argv) < 2 else sys.argv[1]
    codegen(config_path)
    print("Code generation complete.")
