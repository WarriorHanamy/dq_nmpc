"""Generate feasible quadrotor trajectories via minco-python and export to NPZ."""

from __future__ import annotations

import os
import webbrowser
from contextlib import contextmanager
from pathlib import Path

import minco
import numpy as np

from dq_nmpc.minco_trajectory.visualization import visualize_trajectory
from dq_nmpc.minco_trajectory.waypoints import make_sfc_box, waypoints_for_shape
from dq_nmpc.schema import OutputPaths, TrajectoryConfig

_GCONFIG_ROOT = Path(__file__).resolve().parent

_NOMINAL_SPEED = 2.0  # [m/s] initial piece duration seed


@contextmanager
def _in_dir(path: Path):
    """Temporarily change working directory."""
    old = os.getcwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(old)


def generate_trajectory(
    config: TrajectoryConfig,
    output: str | Path | None = None,
) -> Path:
    if output is None:
        paths = OutputPaths.from_trajectory_config(config)
        npz_path = paths.trajectory_npz
        viz_path = paths.trajectory_html
    else:
        output = Path(output)
        npz_path = output.with_suffix(".npz")
        viz_path = output.with_suffix(".html")

    inner_points = waypoints_for_shape(config.shape, config.num_waypoints)
    num_pieces = inner_points.shape[1] + 1

    head_pvaj = np.column_stack([inner_points[:, 0], np.zeros(3), np.zeros(3), np.zeros(3)])
    tail_pvaj = np.column_stack([inner_points[:, -1], np.zeros(3), np.zeros(3), np.zeros(3)])

    pts = np.column_stack([inner_points[:, 0], inner_points])
    path_len = 0.0
    for i in range(pts.shape[1] - 1):
        path_len += float(np.linalg.norm(pts[:, i + 1] - pts[:, i]))
    init_dt = max(path_len / num_pieces / _NOMINAL_SPEED, 0.01)
    initial_time = np.full(num_pieces, init_dt)

    sfc_centers = []
    sfc_polys = []
    half_extents = tuple(config.sfc_half_extents)
    for k in range(num_pieces):
        if k == 0:
            center = inner_points[:, 0]
        elif k == num_pieces - 1:
            center = inner_points[:, -1]
        else:
            center = inner_points[:, k]
        sfc_centers.append(center.copy())
        sfc_polys.append(make_sfc_box(center, half_extents))

    with _in_dir(_GCONFIG_ROOT):
        opt = minco.gcopter.GCOPTERPolytopeSFC()
    ok = opt.setup_basic_trajectory(
        head_pvaj,
        tail_pvaj,
        initial_time,
        inner_points,
        sfc_polys,
        smoothing_factor=1e-1,
        integral_resolution=24,
    )
    if not ok:
        raise RuntimeError("GCOPTER setup failed")

    cost, traj7 = opt.optimize(rel_cost_tol=1e-3)

    t_junctions = np.cumsum(np.insert(traj7.durations, 0, 0.0))
    optimized = np.column_stack(
        [np.array(traj7.get_pos(t), dtype=np.float64).ravel() for t in t_junctions]
    )

    durations = np.array(list(traj7.durations), dtype=np.float64)
    coeffs = np.stack([traj7[i].get_coeff_mat() for i in range(len(traj7))])
    np.savez(npz_path, durations=durations, coeffs=coeffs)

    print(f"Trajectory: {npz_path}  |  viz: {viz_path}")

    visualize_trajectory(
        traj7,
        output_path=viz_path,
        ts=config.control_update_interval,
        shape=config.shape,
        cost=cost,
        inner_points=inner_points,
        sfc_centers=sfc_centers,
        half_extents=half_extents,
        optimized_positions=optimized,
    )
    webbrowser.open(str(viz_path))

    return npz_path
