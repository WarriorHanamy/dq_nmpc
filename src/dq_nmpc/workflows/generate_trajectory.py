"""Trajectory generation pipeline: waypoints → GCOPTER → NPZ + visualization."""

from pathlib import Path

from dq_nmpc.minco_trajectory.generator import generate_trajectory as _generate_trajectory
from dq_nmpc.schema import TrajectoryConfig


def generate_trajectory(
    config: TrajectoryConfig,
    output: str | Path | None = None,
) -> Path:
    """Generate a feasible quadrotor trajectory and write to NPZ.

    @param[in] config   TrajectoryConfig with shape, control_update_interval, num_waypoints
    @param[in] output   Output base path (defaults to out/{shape}/trajectory.npz)
    @return             Path to the written NPZ file
    """
    return _generate_trajectory(config=config, output=output)
