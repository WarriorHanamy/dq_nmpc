"""Trajectory generation pipeline: waypoints → GCOPTER → flatness sampling → CSV.

This workflow imports the library function and wraps it with the CLI-facing interface.
"""

from pathlib import Path

from dq_nmpc.minco_trajectory.generator import generate_trajectory as _generate_trajectory
from dq_nmpc.schema import TrajectoryConfig


def generate_trajectory(
    config: TrajectoryConfig,
    output: str | Path | None = None,
) -> Path:
    """Generate a feasible quadrotor trajectory and write to CSV.

    @param[in] config        TrajectoryConfig with shape, ts, mass, gravity, num_waypoints
    @param[in] output        Output CSV path (defaults to out/{shape}/trajectory.csv)
    @return                  Path to the written CSV file
    """
    return _generate_trajectory(config=config, output=output)
