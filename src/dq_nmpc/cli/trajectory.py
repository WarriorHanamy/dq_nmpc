"""Zero-logic dispatch for dq-trajectory CLI verb."""

from __future__ import annotations

import argparse


def main():
    """CLI entrypoint for dq-trajectory."""
    from dq_nmpc.utils.waypoints import SHAPES
    from dq_nmpc.workflows.generate_trajectory import generate_trajectory

    parser = argparse.ArgumentParser(description="Generate feasible quadrotor trajectory")
    parser.add_argument("--shape", choices=SHAPES, default="hover", help="Trajectory shape")
    parser.add_argument("--output", type=str, default="trajectory.csv", help="Output CSV path")
    parser.add_argument("--ts", type=float, default=0.03, help="Sample time [s]")
    parser.add_argument(
        "--total-time", type=float, default=5.0, help="Total trajectory duration [s]"
    )
    parser.add_argument("--mass", type=float, default=1.0, help="Mass [kg]")
    parser.add_argument("--gravity", type=float, default=9.80665, help="Gravity [m/s^2]")
    args = parser.parse_args()

    output_path = generate_trajectory(
        shape=args.shape,
        output=args.output,
        ts=args.ts,
        total_time=args.total_time,
        mass=args.mass,
        gravity=args.gravity,
    )

    print(f"Trajectory written: {output_path}")
