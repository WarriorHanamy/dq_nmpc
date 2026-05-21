"""Trajectory generation pipeline: waypoints → GCOPTER → flatness sampling → CSV.

This workflow imports the library function and wraps it with the CLI-facing interface.
"""

from pathlib import Path

from dq_nmpc.minco_trajectory.generator import generate_trajectory as _generate_trajectory
from dq_nmpc.schema import PhysicsParams, TrajectoryConfig


def generate_trajectory(
    config: TrajectoryConfig,
    physics: PhysicsParams,
    output: str | Path | None = None,
) -> Path:
    """Generate a feasible quadrotor trajectory and write to CSV.

    @param[in] config   TrajectoryConfig with shape, control_update_interval, num_waypoints
    @param[in] physics  PhysicsParams (mass, Ixx, Iyy, Izz, gravity) for flatness computation
    @param[in] output   Output CSV path (defaults to out/{shape}/trajectory.csv)
    @return             Path to the written CSV file
    """
    return _generate_trajectory(config=config, physics=physics, output=output)
