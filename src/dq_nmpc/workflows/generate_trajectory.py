"""Trajectory generation pipeline: waypoints → GCOPTER → flatness sampling → CSV.

This workflow imports the library function and wraps it with the CLI-facing interface.
"""

from pathlib import Path

from dq_nmpc.trajectory.generator import generate_trajectory as _generate_trajectory


def generate_trajectory(
    shape: str = "hover",
    output: str | Path | None = None,
    ts: float = 0.03,
    total_time: float = 5.0,
    mass: float = 1.0,
    gravity: float = 9.80665,
    num_waypoints: int = 10,
) -> Path:
    """Generate a feasible quadrotor trajectory and write to CSV.

    @param[in] shape         One of hover, line, circle, fig8
    @param[in] output        Output CSV path (defaults to out/{shape}/trajectory.csv)
    @param[in] ts            Sample time [s]
    @param[in] total_time    Total trajectory duration [s]
    @param[in] mass          Vehicle mass [kg]
    @param[in] gravity       Gravitational acceleration [m/s^2]
    @param[in] num_waypoints Number of intermediate waypoints
    @return                  Path to the written CSV file
    """
    return _generate_trajectory(
        shape=shape,
        output=output,
        ts=ts,
        total_time=total_time,
        mass=mass,
        gravity=gravity,
        num_waypoints=num_waypoints,
    )
