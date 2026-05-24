"""Trajectory generation pipeline: waypoints → GCOPTER → geometry sampling → CSV."""

from pathlib import Path

from dq_nmpc.minco_trajectory.generator import generate_trajectory as _generate_trajectory
from dq_nmpc.schema import TrajectoryConfig


def generate_trajectory(
    config: TrajectoryConfig,
    output: str | Path | None = None,
) -> Path:
    """Generate a feasible quadrotor trajectory and write to CSV.

    @param[in] config   TrajectoryConfig with shape, control_update_interval, num_waypoints
    @param[in] output   Output CSV path (defaults to out/{shape}/trajectory.csv)
    @return             Path to the written CSV file
    """
    return _generate_trajectory(config=config, output=output)
