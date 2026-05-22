"""Reference trajectory construction: minco → dense ref_params → ReferenceTrajectoryAsBullets."""

from __future__ import annotations

import logging
from typing import Any

import casadi as ca
import numpy as np

from dq_nmpc.math.dq_functions import (
    dualquat_from_pose_ca_func,
    inertial_to_body_rotation_ca_func,
)
from dq_nmpc.minco_trajectory.flatness_casadi import make_flatness_casadi
from dq_nmpc.schema import NMPCConfig, ReferenceTrajectoryAsBullets

logger = logging.getLogger(__name__)

_FLATNESS_FN: ca.Function | None = None
_DQ_FROM_POSE_FN: ca.Function | None = None
_INV_ROT_FN: ca.Function | None = None


def _get_flatness_fn() -> ca.Function:
    """Lazily compile CasADi flatness decomposition Function."""
    global _FLATNESS_FN
    if _FLATNESS_FN is None:
        _FLATNESS_FN = make_flatness_casadi()
    return _FLATNESS_FN


def _get_dq_fns() -> tuple[ca.Function, ca.Function]:
    """Lazily compile dq_from_pose and inertial_to_body_rotation."""
    global _DQ_FROM_POSE_FN, _INV_ROT_FN
    if _DQ_FROM_POSE_FN is None:
        _DQ_FROM_POSE_FN = dualquat_from_pose_ca_func()
    if _INV_ROT_FN is None:
        _INV_ROT_FN = inertial_to_body_rotation_ca_func()
    return _DQ_FROM_POSE_FN, _INV_ROT_FN


def _get_snap(traj7: Any, t_global: float) -> np.ndarray:
    """Get snap (4th derivative) at a global trajectory time.

    @param[in] traj7     minco.poly_traj.Trajectory7 instance
    @param[in] t_global  Global time along the trajectory [s]
    @return              (3,) snap vector [m/s^4]
    """
    return np.array(traj7.get_sna(t_global), dtype=np.float64).ravel()


def dense_ref_from_minco(
    traj7: Any,
    config: NMPCConfig,
    *,
    zero_yaw: bool = True,
) -> np.ndarray:
    """Convert a minco Trajectory7 to dense ref_params array ``(N_total, 18)``.

    For each sample point at ``control_dt`` intervals:

    1. Sample pos, vel, acc, jerk, snap from T7
    2. Compute yaw (or force zero)
    3. ``flatness_fn`` → quat, omega, thrust, torque
    4. ``dq_from_pose(pos, quat)`` → dq ``(8,)``
    5. ``inertial_to_body(quat, vel_world)`` → vel_body ``(3,)``
    6. Stack → ``[dq(8), omega(3), vel_body(3), thrust, tau_xyz]`` → ``(18,)``

    @param[in] traj7     minco.poly_traj.Trajectory7 instance
    @param[in] config    NMPCConfig with physics params
    @param[in] zero_yaw  Force yaw and yaw derivatives to zero (default True)
    @return              ``(N_total, 18)`` float64 array
    """
    _eps = 1e-10
    flatness_fn = _get_flatness_fn()
    dq_from_pose, inv_rot = _get_dq_fns()

    p = config.physics
    mass_v = p.mass
    Ixx_v = p.ixx
    Iyy_v = p.iyy
    Izz_v = p.izz
    gravity_v = p.gravity

    control_dt = config.ocp.control_update_interval
    duration = float(traj7.total_duration)
    N_total = int(duration / control_dt) + 1
    if N_total < 2:
        N_total = 2
    t_vec = np.linspace(0.0, duration, N_total, dtype=np.float64)

    ref_params = np.zeros((N_total, 18), dtype=np.float64)
    yaw_prev = 0.0

    for i in range(N_total):
        t = t_vec[i]
        pos_i = np.array(traj7.get_pos(t), dtype=np.float64).ravel()
        vel_i = np.array(traj7.get_vel(t), dtype=np.float64).ravel()
        acc_i = np.array(traj7.get_acc(t), dtype=np.float64).ravel()
        jerk_i = np.array(traj7.get_jer(t), dtype=np.float64).ravel()
        snap_i = _get_snap(traj7, t)

        if zero_yaw:
            yaw_i = 0.0
            yaw_dot_i = 0.0
            yaw_ddot_i = 0.0
        else:
            vx, vy = float(vel_i[0]), float(vel_i[1])
            ax, ay = float(acc_i[0]), float(acc_i[1])
            jx, jy = float(jerk_i[0]), float(jerk_i[1])
            den_h = vx * vx + vy * vy
            if den_h > _eps:
                yaw_i = float(np.arctan2(vy, vx))
                yaw_dot_i = (vx * ay - vy * ax) / den_h
                num = vx * ay - vy * ax
                num_d = vx * jy - vy * jx
                den_d = 2.0 * (vx * ax + vy * ay)
                yaw_ddot_i = (num_d * den_h - num * den_d) / (den_h * den_h)
            else:
                yaw_i = yaw_prev
                yaw_dot_i = 0.0
                yaw_ddot_i = 0.0

        yaw_prev = yaw_i

        result = flatness_fn(
            float(acc_i[0]),
            float(acc_i[1]),
            float(acc_i[2]),
            float(jerk_i[0]),
            float(jerk_i[1]),
            float(jerk_i[2]),
            float(snap_i[0]),
            float(snap_i[1]),
            float(snap_i[2]),
            yaw_i,
            yaw_dot_i,
            yaw_ddot_i,
            mass_v,
            Ixx_v,
            Iyy_v,
            Izz_v,
            gravity_v,
        )
        quat_i = np.array([float(result[0]), float(result[1]), float(result[2]), float(result[3])])
        omega_i = np.array([float(result[4]), float(result[5]), float(result[6])])
        thrust_i = float(result[10])
        torque_i = np.array([float(result[11]), float(result[12]), float(result[13])])

        dq_i = np.array(
            dq_from_pose(
                float(quat_i[0]),
                float(quat_i[1]),
                float(quat_i[2]),
                float(quat_i[3]),
                float(pos_i[0]),
                float(pos_i[1]),
                float(pos_i[2]),
            )
        ).ravel()

        vel_body = np.array(inv_rot(quat_i.reshape((4, 1)), vel_i.reshape((3, 1)))).ravel()

        ref_params[i, 0:8] = dq_i
        ref_params[i, 8:11] = omega_i
        ref_params[i, 11:14] = vel_body
        ref_params[i, 14] = thrust_i
        ref_params[i, 15] = torque_i[0]
        ref_params[i, 16] = torque_i[1]
        ref_params[i, 17] = torque_i[2]

    for i in range(1, N_total):
        if float(np.dot(ref_params[i, 0:4], ref_params[i - 1, 0:4])) < 0.0:
            ref_params[i, 0:8] = -ref_params[i, 0:8]

    logger.info(
        "Dense ref_params built: N_total=%d, dt=%.3f s, zero_yaw=%s",
        N_total,
        control_dt,
        zero_yaw,
    )
    return ref_params


def bullets_from_dense(
    ref_params: np.ndarray,
    horizon_steps: int,
) -> ReferenceTrajectoryAsBullets:
    """Slide window over dense ref_params to build the bullet belt.

    ``N_c = N_total`` — one bullet per control step.  Bullets that extend
    past the trajectory end are clamped to the last valid ``ref_params`` row.

    @param[in] ref_params      ``(N_total, 18)`` dense reference parameters
    @param[in] horizon_steps   Number of shooting-interval nodes per bullet
    @return                    ``ReferenceTrajectoryAsBullets`` with shape ``(N_c, N, 18)``
    """
    N_total = ref_params.shape[0]
    N_c = N_total
    if N_c < 1:
        raise ValueError(f"N_total={N_total} must be >= 1")
    ref_params = np.asarray(ref_params, dtype=np.float64)

    bullets = np.zeros((N_c, horizon_steps, 18), dtype=np.float64)
    last_valid = ref_params[-1].copy()
    for k in range(N_c):
        start = k
        n_valid = min(horizon_steps, N_total - start)
        bullets[k, :n_valid] = ref_params[start : start + n_valid]
        if n_valid < horizon_steps:
            bullets[k, n_valid:] = last_valid

    logger.info(
        "Bullet belt built: N_c=%d, horizon=%d steps, shape=%s",
        N_c,
        horizon_steps,
        bullets.shape,
    )
    return ReferenceTrajectoryAsBullets(
        bullets=bullets,
        N_c=N_c,
        horizon_steps=horizon_steps,
    )
