"""Zero-logic dispatch for dq-trajectory CLI verb."""

from __future__ import annotations

import argparse

from minco.trajectory import SHAPES


def main():
    """CLI entrypoint for dq-trajectory."""
    from dq_nmpc.schema import TrajectoryConfig
    from dq_nmpc.workflows.generate_trajectory import generate_trajectory

    parser = argparse.ArgumentParser(description="Generate feasible quadrotor trajectory")
    parser.add_argument(
        "--config",
        type=str,
        default="src/dq_nmpc/config/trajectory.yaml",
        help="Path to trajectory config",
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--shape", choices=SHAPES, default=None, help="Override shape in config")
    args = parser.parse_args()

    tc = TrajectoryConfig.from_yaml(args.config)
    if args.shape is not None:
        tc = tc.model_copy(update={"shape": args.shape})

    output_path = generate_trajectory(config=tc, output=args.output)
    print(f"Trajectory written: {output_path}")
