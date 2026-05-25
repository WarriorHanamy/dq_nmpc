"""Trajectory generation pipeline: waypoints → GCOPTER → NPZ + visualization."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
from minco.trajectory import generate as _generate
from minco.trajectory import save_npz, visualize

from dq_nmpc.schema import OutputPaths, TrajectoryConfig


def generate_trajectory(
    config: TrajectoryConfig,
    output: str | Path | None = None,
    gcopter_config_path: str | Path | None = None,
) -> Path:
    """Generate a feasible quadrotor trajectory and write to NPZ.

    @param[in] config                 TrajectoryConfig with shape, control_update_interval, num_waypoints
    @param[in] output                 Output base path (defaults to out/{shape}/trajectory.npz)
    @param[in] gcopter_config_path    Path to GCOPTER config YAML (uses dq_nmpc defaults if None)
    @return                           Path to the written NPZ file
    """
    if gcopter_config_path is None:
        gcopter_config_path = "src/dq_nmpc/config/gcopter_trajopt.yaml"

    sfc = tuple(config.sfc_half_extents)
    result = _generate(
        config.shape,
        num_waypoints=config.num_waypoints,
        sfc_half_extents=sfc,
        gcopter_config_path=gcopter_config_path,
    )

    if output is None:
        paths = OutputPaths.from_trajectory_config(config)
        npz_path = paths.trajectory_npz
        viz_path = paths.trajectory_html
    else:
        output = Path(output)
        npz_path = output.with_suffix(".npz")
        viz_path = output.with_suffix(".html")

    t_junctions = np.cumsum(np.insert(result.traj7.durations, 0, 0.0))
    optimized = np.column_stack(
        [np.array(result.traj7.get_pos(t), dtype=np.float64).ravel() for t in t_junctions]
    )

    save_npz(result.traj7, npz_path)
    print(f"Trajectory: {npz_path}  |  viz: {viz_path}")

    visualize(
        result.traj7,
        viz_path,
        ts=config.control_update_interval,
        title=f"GCOPTER — {config.shape} (cost={result.cost:.2f})",
        seed_waypoints=result.waypoints,
        sfc_centers=result.sfc_centers,
        half_extents=result.sfc_half_extents,
        optimized_positions=optimized,
    )
    webbrowser.open(str(viz_path))

    return npz_path
