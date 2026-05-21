"""Generate feasible quadrotor trajectories via minco-python and export to CSV."""

from __future__ import annotations

import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import minco
import numpy as np

from dq_nmpc.minco_trajectory.flatness_casadi import make_flatness_casadi
from dq_nmpc.minco_trajectory.visualization import visualize_trajectory
from dq_nmpc.minco_trajectory.waypoints import make_sfc_box, waypoints_for_shape
from dq_nmpc.schema import TRAJECTORY_CSV_COLUMNS, OutputPaths, PhysicsParams, TrajectoryConfig

_GCONFIG_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "dq_nmpc" / "config" / "mujoco" / "default"
)

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


def _sample_trajectory(
    traj7: Any,
    ts: float,
    mass: float,
    gravity: float,
    ixx: float,
    iyy: float,
    izz: float,
) -> list[tuple[float, ...]]:
    """Sample a Trajectory7 and decompose via CasADi flatness.

    Returns CSV rows with thrust, quaternion, body rates, and torque.
    """
    flatness_fn = make_flatness_casadi()
    duration = traj7.total_duration
    num_samples = int(duration / ts) + 1
    if num_samples < 2:
        num_samples = 2
    rows = []
    dt = duration / max(num_samples - 1, 1)

    for i in range(num_samples):
        t = i * dt
        pos = np.array(traj7.get_pos(t), dtype=np.float64).ravel()
        vel = np.array(traj7.get_vel(t), dtype=np.float64).ravel()
        acc = np.array(traj7.get_acc(t), dtype=np.float64).ravel()
        jer = np.array(traj7.get_jer(t), dtype=np.float64).ravel()
        sna = np.array(traj7.get_sna(t), dtype=np.float64).ravel()

        yaw = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 0.01 else 0.0

        result = flatness_fn(
            float(acc[0]),
            float(acc[1]),
            float(acc[2]),
            float(jer[0]),
            float(jer[1]),
            float(jer[2]),
            float(sna[0]),
            float(sna[1]),
            float(sna[2]),
            yaw,
            0.0,
            0.0,
            mass,
            ixx,
            iyy,
            izz,
            gravity,
        )
        rows.append(
            (
                t,
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(vel[0]),
                float(vel[1]),
                float(vel[2]),
                float(acc[0]),
                float(acc[1]),
                float(acc[2]),
                float(jer[0]),
                float(jer[1]),
                float(jer[2]),
                float(sna[0]),
                float(sna[1]),
                float(sna[2]),
                float(result[0]),
                float(result[1]),
                float(result[2]),
                float(result[3]),
                float(result[4]),
                float(result[5]),
                float(result[6]),
                float(result[10]),
                float(result[11]),
                float(result[12]),
                float(result[13]),
            )
        )

    return rows


def generate_trajectory(
    config: TrajectoryConfig,
    physics: PhysicsParams,
    output: str | Path | None = None,
) -> Path:
    if output is None:
        paths = OutputPaths.from_trajectory_config(config)
        output = paths.trajectory_csv
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

    # Option B: initial piece duration from arc length / nominal speed
    pts = np.column_stack([inner_points[:, 0], inner_points])
    path_len = 0.0
    for i in range(pts.shape[1] - 1):
        path_len += float(np.linalg.norm(pts[:, i + 1] - pts[:, i]))
    init_dt = max(path_len / num_pieces / _NOMINAL_SPEED, 0.01)
    initial_time = np.full(num_pieces, init_dt)

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

    rows = _sample_trajectory(
        traj7,
        config.control_update_interval,
        physics.mass,
        physics.gravity,
        physics.ixx,
        physics.iyy,
        physics.izz,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        f.write(f"# control_update_interval={config.control_update_interval}\n")
        writer = csv.writer(f)
        writer.writerow(list(TRAJECTORY_CSV_COLUMNS))
        writer.writerows(rows)

    print(f"Trajectory saved: {output}  ({len(rows)} points, cost={cost:.4f})")

    durations = np.array(list(traj7.durations), dtype=np.float64)
    coeffs = np.stack([traj7[i].get_coeff_mat() for i in range(len(traj7))])
    np.savez(npz_path, durations=durations, coeffs=coeffs)
    print(f"Trajectory coeffs saved: {npz_path}")

    visualize_trajectory(
        csv_path=output,
        shape=config.shape,
        inner_points=inner_points,
        sfc_centers=sfc_centers,
        half_extents=half_extents,
        cost=cost,
        output_path=viz_path,
    )

    return output
