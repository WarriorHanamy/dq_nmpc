"""Load trajectory CSV into ReferenceTrajectory and reinterpret minco NPZ for NMPC."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from dq_nmpc.minco_trajectory.flatness_casadi import make_flatness_casadi
from dq_nmpc.schema import (
    TRAJECTORY_CSV_COLUMNS,
    FlatnessTrajectory,
    NMPCConfig,
    ReferenceTrajectory,
    TrajectoryPoint,
)


def _parse_csv_meta(path: Path) -> dict[str, str]:
    """Read '# key=value' comment lines from top of CSV."""
    meta: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                stripped = stripped[1:].strip()
                if "=" in stripped:
                    key, _, val = stripped.partition("=")
                    meta[key.strip()] = val.strip()
            else:
                break
    return meta


def load_trajectory_csv(path: str | Path) -> ReferenceTrajectory:
    path = Path(path)
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None:
            missing = set(TRAJECTORY_CSV_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV header missing columns: {sorted(missing)}. "
                    f"Expected: {list(TRAJECTORY_CSV_COLUMNS)}"
                )
        for row in reader:
            tp = TrajectoryPoint(
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                vz=float(row["vz"]),
                qw=float(row["qw"]),
                qx=float(row["qx"]),
                qy=float(row["qy"]),
                qz=float(row["qz"]),
                wx=float(row["wx"]),
                wy=float(row["wy"]),
                wz=float(row["wz"]),
                thrust=float(row["thrust"]),
            )
            points.append(tp)
    return ReferenceTrajectory(points=points, horizon_steps=len(points))


def load_trajectory_meta(path: str | Path) -> dict[str, str]:
    """Parse metadata from trajectory CSV comment header."""
    return _parse_csv_meta(Path(path))


def validate_trajectory_ts(csv_path: str | Path, config: NMPCConfig) -> None:
    """Verify trajectory sample time matches NMPC config."""
    meta = _parse_csv_meta(Path(csv_path))
    ts_csv = float(meta.get("ts", 0.0))
    ts_nmpc = config.nmpc.ts
    if abs(ts_csv - ts_nmpc) > 1e-6:
        raise ValueError(f"ts mismatch: trajectory={ts_csv} vs nmpc config={ts_nmpc}")


def load_trajectory_npz(path: str | Path) -> Any:
    """Reconstruct a minco Trajectory5 from a .npz coefficient file.

    @param[in] path  Path to trajectory.npz (alongside trajectory.csv)
    @return          minco.poly_traj.Trajectory5 instance
    """
    import minco

    data = np.load(path)
    durations = data["durations"].tolist()
    coeff_mats = [data["coeffs"][i] for i in range(len(durations))]
    return minco.poly_traj.Trajectory5(durations, coeff_mats)


_FLATNESS_FN = None


def _get_flatness_fn():
    """Lazily compile the flatness CasADi Function (JIT-on-first-call)."""
    global _FLATNESS_FN
    if _FLATNESS_FN is None:
        _FLATNESS_FN = make_flatness_casadi()
    return _FLATNESS_FN


def reinterpret_minco_trajectory(
    traj5: Any,
    config: NMPCConfig,
    ts: float,
    *,
    zero_yaw: bool = True,
) -> FlatnessTrajectory:
    """Reinterpret a minco Trajectory5 into a full FlatnessTrajectory.

    Samples the piecewise-polynomial trajectory at ``ts`` intervals and
    computes the flatness decomposition for every sample point:

    * orientation quaternion (CasADi flatness map)
    * body-frame angular velocity (CasADi flatness map)
    * thrust (CasADi flatness map)
    * body-frame angular acceleration (numerical central difference of omega)
    * body-frame torque (analytical from omega and omega_dot)

    When ``zero_yaw=True`` (default), yaw and yaw_dot are set to zero
    throughout the trajectory.  The drone keeps a fixed heading instead
    of yaw-aligning to the velocity direction.

    @param[in] traj5     minco.poly_traj.Trajectory5 instance
    @param[in] config    NMPCConfig providing mass, Ixx, Iyy, Izz, gravity
    @param[in] ts        Sample time [s]
    @param[in] zero_yaw  Force yaw/yaw_dot to zero (default True)
    @return              FlatnessTrajectory with all 14 reference arrays
    """
    _eps = 1e-10
    flatness_fn = _get_flatness_fn()

    mass_v = config.mass
    Ixx_v = config.ixx
    Iyy_v = config.iyy
    Izz_v = config.izz
    gravity_v = config.gravity

    J = np.array(
        [[Ixx_v, 0.0, 0.0], [0.0, Iyy_v, 0.0], [0.0, 0.0, Izz_v]],
        dtype=np.float64,
    )

    duration = float(traj5.total_duration)
    num_pts = int(duration / ts) + 1
    if num_pts < 2:
        num_pts = 2
    t_vec = np.linspace(0.0, duration, num_pts, dtype=np.float64)

    N = num_pts
    ref_pos = np.zeros((N, 3), dtype=np.float64)
    ref_vel = np.zeros((N, 3), dtype=np.float64)
    ref_acc = np.zeros((N, 3), dtype=np.float64)
    ref_jerk = np.zeros((N, 3), dtype=np.float64)
    ref_quat = np.zeros((N, 4), dtype=np.float64)
    ref_omega = np.zeros((N, 3), dtype=np.float64)
    ref_omega_dot = np.zeros((N, 3), dtype=np.float64)
    ref_thrust = np.zeros(N, dtype=np.float64)
    ref_torque = np.zeros((N, 3), dtype=np.float64)
    ref_yaw = np.zeros(N, dtype=np.float64)
    ref_yaw_dot = np.zeros(N, dtype=np.float64)
    ref_yaw_ddot = np.zeros(N, dtype=np.float64)

    for i in range(N):
        t = t_vec[i]

        pos_i = np.array(traj5.get_pos(t), dtype=np.float64).ravel()
        vel_i = np.array(traj5.get_vel(t), dtype=np.float64).ravel()
        acc_i = np.array(traj5.get_acc(t), dtype=np.float64).ravel()
        jerk_i = np.array(traj5.get_jer(t), dtype=np.float64).ravel()

        ref_pos[i] = pos_i
        ref_vel[i] = vel_i
        ref_acc[i] = acc_i
        ref_jerk[i] = jerk_i

        if zero_yaw:
            yaw_i = 0.0
            yaw_dot_i = 0.0
        else:
            vx, vy = float(vel_i[0]), float(vel_i[1])
            ax, ay = float(acc_i[0]), float(acc_i[1])
            den_h = vx * vx + vy * vy
            if den_h > _eps:
                yaw_i = float(np.arctan2(vy, vx))
                yaw_dot_i = (vx * ay - vy * ax) / den_h
            else:
                yaw_i = ref_yaw[i - 1] if i > 0 else 0.0
                yaw_dot_i = 0.0

        ref_yaw[i] = yaw_i
        ref_yaw_dot[i] = yaw_dot_i

        result = flatness_fn(
            float(acc_i[0]),
            float(acc_i[1]),
            float(acc_i[2]),
            float(jerk_i[0]),
            float(jerk_i[1]),
            float(jerk_i[2]),
            yaw_i,
            yaw_dot_i,
            mass_v,
            Ixx_v,
            Iyy_v,
            Izz_v,
            gravity_v,
        )
        qw_i = float(result[0])
        qx_i = float(result[1])
        qy_i = float(result[2])
        qz_i = float(result[3])
        wx_i = float(result[4])
        wy_i = float(result[5])
        wz_i = float(result[6])
        thrust_i = float(result[7])

        ref_quat[i] = [qw_i, qx_i, qy_i, qz_i]
        ref_omega[i] = [wx_i, wy_i, wz_i]
        ref_thrust[i] = thrust_i

    for i in range(1, N):
        if float(np.dot(ref_quat[i], ref_quat[i - 1])) < 0.0:
            ref_quat[i] = -ref_quat[i]
        nrm = float(np.linalg.norm(ref_quat[i]))
        if nrm > 0.0:
            ref_quat[i] /= nrm

    # TODO, this is noisy
    for i in range(N):
        if i == 0:
            omega_dot_i = (ref_omega[i + 1] - ref_omega[i]) / ts
        elif i == N - 1:
            omega_dot_i = (ref_omega[i] - ref_omega[i - 1]) / ts
        else:
            omega_dot_i = (ref_omega[i + 1] - ref_omega[i - 1]) / (2.0 * ts)
        ref_omega_dot[i] = omega_dot_i

        omega_i = ref_omega[i]
        torque_i = J @ omega_dot_i + np.cross(omega_i, J @ omega_i)
        ref_torque[i] = torque_i

    return FlatnessTrajectory(
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        ref_acc=ref_acc,
        ref_jerk=ref_jerk,
        ref_quat=ref_quat,
        ref_omega=ref_omega,
        ref_omega_dot=ref_omega_dot,
        ref_thrust=ref_thrust,
        ref_torque=ref_torque,
        ref_yaw=ref_yaw,
        ref_yaw_dot=ref_yaw_dot,
        ref_yaw_ddot=ref_yaw_ddot,
        t=t_vec,
    )
