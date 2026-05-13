"""Generate feasible quadrotor trajectories via minco-python and export to CSV.

Each CSV row: t, x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, thrust
- position/velocity: world ENU frame
- orientation: world-frame quaternion (w,x,y,z)
- angular velocity: body FLU frame
- thrust: body FLU frame [N]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import minco
import numpy as np
from minco.flatness_cache import CachedFlatness

SHAPES = ("hover", "line", "circle", "fig8")


def _make_sfc_box(center, half_extents):
    """Create a safe-flight corridor as an axis-aligned box.

    Returns (M, 4) half-space matrix where each row [A,B,C,D] encodes
    Ax + By + Cz + D <= 0.
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents
    A = np.array(
        [
            [1, 0, 0, -(cx - hx)],
            [-1, 0, 0, (cx + hx)],
            [0, 1, 0, -(cy - hy)],
            [0, -1, 0, (cy + hy)],
            [0, 0, 1, -(cz - hz)],
            [0, 0, -1, (cz + hz)],
        ],
        dtype=np.float64,
    )
    return A


def _waypoints_for_shape(shape, num_waypoints=10):
    if shape == "hover":
        center = np.array([0.0, 0.0, 2.0])
        return np.tile(center[:, None], (1, num_waypoints - 1))
    elif shape == "line":
        start = np.array([0.0, 0.0, 1.5])
        end = np.array([5.0, 0.0, 1.5])
        return np.linspace(start, end, num_waypoints - 1).T
    elif shape == "circle":
        angles = np.linspace(0, 2 * np.pi, num_waypoints - 1, endpoint=False)
        radius = 2.0
        z = 2.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.vstack([x, y, np.full_like(x, z)])
    elif shape == "fig8":
        t = np.linspace(0, 2 * np.pi, num_waypoints - 1)
        a = 2.0
        x = a * np.sin(t)
        y = a * np.sin(t) * np.cos(t)
        z = np.full_like(x, 2.0)
        return np.vstack([x, y, z])
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _sample_trajectory(
    traj5, flatness: CachedFlatness, ts: float, horizon_steps: int, thrust_hover: float
):
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
    output: str | Path = "trajectory.csv",
    ts: float = 0.03,
    total_time: float = 5.0,
    mass: float = 1.0,
    gravity: float = 9.80665,
    num_waypoints: int = 10,
) -> Path:
    output = Path(output)

    thrust_hover = mass * gravity
    flatness = CachedFlatness(mass=mass, gravity=gravity)

    inner_points = _waypoints_for_shape(shape, num_waypoints)
    num_pieces = inner_points.shape[1] + 1

    head_pva = np.column_stack([inner_points[:, 0], np.zeros(3), np.zeros(3)])
    tail_pva = np.column_stack([inner_points[:, -1], np.zeros(3), np.zeros(3)])

    piece_duration = total_time / num_pieces
    initial_time = np.full(num_pieces, piece_duration)

    sfc_polys = []
    half_extents = (0.5, 0.5, 0.5)
    for k in range(num_pieces):
        if k == 0:
            center = inner_points[:, 0]
        elif k == num_pieces - 1:
            center = inner_points[:, -1]
        else:
            center = inner_points[:, k]
        sfc_polys.append(_make_sfc_box(center, half_extents))

    opt = minco.gcopter.GCOPTERPolytopeSFC()
    opt.configure_from_file("")
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
    return output


def main():
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
    generate_trajectory(
        shape=args.shape,
        output=args.output,
        ts=args.ts,
        total_time=args.total_time,
        mass=args.mass,
        gravity=args.gravity,
    )


if __name__ == "__main__":
    main()
