"""Generate feasible quadrotor trajectories via minco-python and export to CSV."""

from __future__ import annotations

import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import minco
import numpy as np
from minco.flatness_cache import CachedFlatness

from dq_nmpc.trajectory.visualization import visualize_trajectory
from dq_nmpc.utils.waypoints import make_sfc_box, waypoints_for_shape

_MINCO_ROOT = Path(__file__).resolve().parents[3] / "deps" / "minco-python"


@contextmanager
def _in_dir(path: Path):
    """Temporarily change working directory."""
    old = os.getcwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(old)


def _sample_trajectory(
    traj5: Any,
    flatness: CachedFlatness,
    ts: float,
    horizon_steps: int,
    thrust_hover: float,
) -> list[tuple[float, ...]]:
    total = traj5.total_duration
    num_samples = min(int(total / ts) + 1, horizon_steps * 10)
    rows = []
    dt = total / max(num_samples - 1, 1)

    for i in range(num_samples):
        t = i * dt
        pos = traj5.get_pos(t)
        vel = traj5.get_vel(t)
        acc = traj5.get_acc(t)
        jer = traj5.get_jer(t)

        yaw = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 0.01 else 0.0
        yaw_rate = 0.0

        thrust_nd, quat, body_rates = flatness.forward(vel, acc, jer, yaw, yaw_rate)
        rows.append(
            (
                t,
                pos[0],
                pos[1],
                pos[2],
                vel[0],
                vel[1],
                vel[2],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
                body_rates[0],
                body_rates[1],
                body_rates[2],
                float(thrust_nd * thrust_hover),
            )
        )

    return rows


def generate_trajectory(
    shape: str = "hover",
    output: str | Path | None = None,
    ts: float = 0.03,
    total_time: float = 5.0,
    mass: float = 1.0,
    gravity: float = 9.80665,
    num_waypoints: int = 10,
) -> Path:
    if output is None:
        output = Path(f"out/{shape}/trajectory.csv")
    else:
        output = Path(output)

    thrust_hover = mass * gravity
    flatness = CachedFlatness(mass=mass, gravity=gravity)

    inner_points = waypoints_for_shape(shape, num_waypoints)
    num_pieces = inner_points.shape[1] + 1

    head_pva = np.column_stack([inner_points[:, 0], np.zeros(3), np.zeros(3)])
    tail_pva = np.column_stack([inner_points[:, -1], np.zeros(3), np.zeros(3)])

    piece_duration = total_time / num_pieces
    initial_time = np.full(num_pieces, piece_duration)

    sfc_centers = []
    sfc_polys = []
    half_extents = (0.5, 0.5, 0.5)
    for k in range(num_pieces):
        if k == 0:
            center = inner_points[:, 0]
        elif k == num_pieces - 1:
            center = inner_points[:, -1]
        else:
            center = inner_points[:, k]
        sfc_centers.append(center.copy())
        sfc_polys.append(make_sfc_box(center, half_extents))

    with _in_dir(_MINCO_ROOT):
        opt = minco.gcopter.GCOPTERPolytopeSFC()
    ok = opt.setup_basic_trajectory(
        head_pva,
        tail_pva,
        initial_time,
        inner_points,
        sfc_polys,
        smoothing_factor=1e-1,
        integral_resolution=24,
    )
    if not ok:
        raise RuntimeError("GCOPTER setup failed")

    cost, traj5 = opt.optimize(rel_cost_tol=1e-3)

    horizon_steps = max(int(total_time / ts), 1)
    rows = _sample_trajectory(traj5, flatness, ts, horizon_steps, thrust_hover)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "qw",
                "qx",
                "qy",
                "qz",
                "wx",
                "wy",
                "wz",
                "thrust",
            ]
        )
        writer.writerows(rows)

    print(f"Trajectory saved: {output}  ({len(rows)} points, cost={cost:.4f})")

    viz_path = output.with_suffix(".html")
    visualize_trajectory(
        csv_path=output,
        shape=shape,
        inner_points=inner_points,
        sfc_centers=sfc_centers,
        half_extents=half_extents,
        cost=cost,
        output_path=viz_path,
    )

    return output
