"""Flatness-based reference trajectory planner.

Pipeline:
  _circular_reference  →  _rotated_reference  →  _smooth_reference  →  get_flatness_trajectory
  (ref_pos on a          (expm 3D rotation)     (min-snap QP          (full flatness: pose,
   circular path)                                smoothing)            twist, thrust, torque)

Public API:
  get_flatness_trajectory  —  the only function intended for external callers.
"""

from __future__ import annotations

import numpy as np
import osqp
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation as R

from dq_nmpc.math.polynomial import (
    acceleration_time,
    jerk_time,
    position_time,
    snap_time,
    velocity_time,
)
from dq_nmpc.schema import NMPCConfig

__all__ = ["get_flatness_trajectory"]


def _skew_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross-product skew-symmetric matrix of a 3-vector."""
    a1, a2, a3 = x[0], x[1], x[2]
    return np.array([[0.0, -a3, a2], [a3, 0.0, -a1], [-a2, a1, 0.0]], dtype=np.float64)


def _circular_reference(
    t: NDArray[np.float64],
    radius: float,
    angular_speed: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Flat outputs (position through snap) for a circular trajectory in the XY plane.

    Returns 3xN arrays for ref_pos, ref_vel, ref_acc, ref_jerk, ref_snap
    and 1D arrays for ref_yaw and its derivatives (all zero).
    """
    cos_wt = np.cos(angular_speed * t)
    sin_wt = np.sin(angular_speed * t)

    ref_pos_x = radius * cos_wt
    ref_pos_y = radius * sin_wt
    ref_pos_z = np.zeros_like(t)

    ref_vel_x = -radius * angular_speed * sin_wt
    ref_vel_y = radius * angular_speed * cos_wt
    ref_vel_z = np.zeros_like(t)

    ref_acc_x = -radius * angular_speed**2 * cos_wt
    ref_acc_y = -radius * angular_speed**2 * sin_wt
    ref_acc_z = np.zeros_like(t)

    ref_jerk_x = radius * angular_speed**3 * sin_wt
    ref_jerk_y = -radius * angular_speed**3 * cos_wt
    ref_jerk_z = np.zeros_like(t)

    ref_snap_x = radius * angular_speed**4 * cos_wt
    ref_snap_y = radius * angular_speed**4 * sin_wt
    ref_snap_z = np.zeros_like(t)

    ref_yaw = np.zeros_like(t)
    ref_yaw_dot = np.zeros_like(t)

    ref_pos = np.vstack((ref_pos_x, ref_pos_y, ref_pos_z))
    ref_vel = np.vstack((ref_vel_x, ref_vel_y, ref_vel_z))
    ref_acc = np.vstack((ref_acc_x, ref_acc_y, ref_acc_z))
    ref_jerk = np.vstack((ref_jerk_x, ref_jerk_y, ref_jerk_z))
    ref_snap = np.vstack((ref_snap_x, ref_snap_y, ref_snap_z))

    return ref_pos, ref_yaw, ref_vel, ref_yaw_dot, ref_acc, ref_jerk, ref_snap, ref_yaw_dot


def _rotated_reference(
    t: NDArray[np.float64],
    radius: float,
    angular_speed: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Apply a time-varying 3D rotation (expm) to the circular reference.

    The rotation axis is the body X-axis with a sinusoidal angle a*sin(b*t).
    An offset [0, 0, 4] is added to the position so the trajectory hovers
    above the ground plane.
    """
    ref_pos, ref_yaw, ref_vel, ref_yaw_dot, ref_acc, ref_jerk, ref_snap, _ref_yaw_ddotot = (
        _circular_reference(t, radius, angular_speed)
    )
    a = np.pi / 2
    b = 0.05

    N = ref_vel.shape[1]
    rot_pos = np.zeros((3, N), dtype=np.float64)
    rot_vel = np.zeros((3, N), dtype=np.float64)
    rot_acc = np.zeros((3, N), dtype=np.float64)
    rot_jerk = np.zeros((3, N), dtype=np.float64)
    rot_snap = np.zeros((3, N), dtype=np.float64)

    for k in range(N):
        w_k = np.array([a * np.sin(b * t[k]), 0.0, 0.0], dtype=np.float64)
        w_d = np.array([b * a * np.cos(b * t[k]), 0.0, 0.0], dtype=np.float64)
        w_dd = np.array([-(b**2) * a * np.sin(b * t[k]), 0.0, 0.0], dtype=np.float64)
        w_ddd = np.array([-(b**3) * a * np.cos(b * t[k]), 0.0, 0.0], dtype=np.float64)
        w_dddd = np.array([b**4 * a * np.sin(b * t[k]), 0.0, 0.0], dtype=np.float64)

        expm_w = expm(_skew_matrix(w_k))
        sw_d = _skew_matrix(w_d)
        sw_d2 = sw_d @ sw_d
        sw_d3 = sw_d2 @ sw_d
        sw_d4 = sw_d3 @ sw_d
        sw_dd = _skew_matrix(w_dd)
        sw_dd2 = sw_dd @ sw_dd
        sw_ddd = _skew_matrix(w_ddd)
        sw_dddd = _skew_matrix(w_dddd)

        pk = ref_pos[:, k]
        rot_pos[:, k] = expm_w @ pk
        rot_vel[:, k] = expm_w @ (ref_vel[:, k] + sw_d @ pk)
        rot_acc[:, k] = expm_w @ (
            sw_d2 @ pk + 2 * sw_d @ ref_vel[:, k] + ref_acc[:, k] + sw_dd @ pk
        )
        rot_jerk[:, k] = expm_w @ (
            ref_jerk[:, k]
            + sw_ddd @ pk
            + 3 * sw_dd @ ref_vel[:, k]
            + 3 * sw_d @ ref_acc[:, k]
            + sw_d3 @ pk
            + 3 * sw_d2 @ ref_vel[:, k]
            + 3 * sw_d @ sw_dd @ pk
        )
        rot_snap[:, k] = expm_w @ (
            ref_snap[:, k]
            + sw_dddd @ pk
            + 4 * sw_ddd @ ref_vel[:, k]
            + 6 * sw_dd @ ref_acc[:, k]
            + 4 * sw_d @ ref_jerk[:, k]
            + sw_d4 @ pk
            + 3 * sw_dd2 @ pk
            + 4 * sw_d3 @ ref_vel[:, k]
            + 6 * sw_d2 @ ref_acc[:, k]
            + 6 * sw_d2 @ sw_dd @ pk
            + 4 * sw_d @ sw_ddd @ pk
            + 12 * sw_d @ sw_dd @ ref_vel[:, k]
        )

    z_offset = np.vstack(
        (
            np.zeros_like(t),
            np.zeros_like(t),
            4.0 * np.ones_like(t),
        )
    )
    rot_pos += z_offset
    return rot_pos, rot_vel, rot_acc, rot_jerk, rot_snap, ref_yaw, ref_yaw_dot, _ref_yaw_ddotot


# ---------------------------------------------------------------------------
# Minimum-snap QP piecewise-polynomial smoothing
# ---------------------------------------------------------------------------


def _constraint_block_initial() -> NDArray[np.float64]:
    """Initial-condition constraint block (5x10): fixes c0..c4 at t=0."""
    return np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 24, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )


def _constraint_block_continuity() -> NDArray[np.float64]:
    """Midpoint continuity constraint block (5x10): negates c1..c4."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -24, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )


def _constraint_block_zeros_5x10() -> NDArray[np.float64]:
    """5x10 zero block for waypoint spacing."""
    return np.zeros((5, 10), dtype=np.float64)


def _constraint_block_zeros_1x10() -> NDArray[np.float64]:
    """1x10 zero block for waypoint spacing."""
    return np.zeros((1, 10), dtype=np.float64)


def _constraint_block_terminal(t_end: float) -> NDArray[np.float64]:
    """Terminal-condition constraint block (5x10) for a segment of duration t_end."""
    return np.vstack(
        [
            position_time(t_end).T,
            velocity_time(t_end).T,
            acceleration_time(t_end).T,
            jerk_time(t_end).T,
            snap_time(t_end).T,
        ]
    )


def _build_constraint_matrix(
    segment_durations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble the full QP equality constraint matrix for 4-segment min-snap."""
    Z5 = _constraint_block_zeros_5x10()
    Z1 = _constraint_block_zeros_1x10()
    A_med = _constraint_block_continuity()

    A_t = [_constraint_block_terminal(segment_durations[i]) for i in range(4)]

    A_vel = [velocity_time(d).T for d in segment_durations]
    A_acc = [acceleration_time(d).T for d in segment_durations]
    A_jerk = [jerk_time(d).T for d in segment_durations]
    A_snap = [snap_time(d).T for d in segment_durations]

    A_pos0 = position_time(0.0).T

    return np.block(
        [
            [_constraint_block_initial(), Z5, Z5, Z5],
            [A_t[0], A_med, Z5, Z5],
            [Z5, A_t[1], A_med, Z5],
            [Z5, Z5, A_t[2], A_med],
            [Z5, Z5, Z5, A_t[3]],
            [Z1, A_pos0, Z1, Z1],
            [Z1, Z1, A_pos0, Z1],
            [Z1, Z1, Z1, A_pos0],
            [A_vel[0], Z1, Z1, Z1],
            [A_acc[0], Z1, Z1, Z1],
            [A_jerk[0], Z1, Z1, Z1],
            [A_snap[0], Z1, Z1, Z1],
            [Z1, A_vel[1], Z1, Z1],
            [Z1, A_acc[1], Z1, Z1],
            [Z1, A_jerk[1], Z1, Z1],
            [Z1, A_snap[1], Z1, Z1],
            [Z1, Z1, A_vel[2], Z1],
            [Z1, Z1, A_acc[2], Z1],
            [Z1, Z1, A_jerk[2], Z1],
            [Z1, Z1, A_snap[2], Z1],
            [Z1, Z1, Z1, A_vel[3]],
            [Z1, Z1, Z1, A_acc[3]],
            [Z1, Z1, Z1, A_jerk[3]],
            [Z1, Z1, Z1, A_snap[3]],
        ]
    )


def _build_constraint_rhs(
    waypoints: NDArray[np.float64],
    pva_init: NDArray[np.float64],
    pva_final: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble the RHS vector for QP equality constraints."""
    b_1 = np.array([waypoints[0], 0, 0, 0, 0], dtype=np.float64)
    b_2 = np.array([waypoints[1], 0, 0, 0, 0], dtype=np.float64)
    b_3 = np.array([waypoints[2], 0, 0, 0, 0], dtype=np.float64)
    b_4 = np.array([waypoints[3], 0, 0, 0, 0], dtype=np.float64)
    b_5 = np.array([waypoints[4], 0, 0, 0, 0], dtype=np.float64)
    b_6 = np.array([waypoints[1], waypoints[2], waypoints[3]], dtype=np.float64)

    b_first = np.array([pva_init[1], pva_init[2], pva_init[3], pva_init[4]], dtype=np.float64)
    b_second = np.array([pva_final[1], pva_final[2], pva_final[3], pva_final[4]], dtype=np.float64)
    b_third = b_second
    b_fourth = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return np.concatenate((b_1, b_2, b_3, b_4, b_5, b_6, b_first, b_second, b_third, b_fourth))


def _hessian_cost(t: float) -> NDArray[np.float64]:
    """Snap-minimisation Hessian for a single polynomial segment of duration t."""
    H = np.zeros((10, 10), dtype=np.float64)
    H[5, 5] = 14400 * t
    H[5, 6] = 43200 * t**2
    H[5, 7] = 100800 * t**3
    H[5, 8] = 201600 * t**4
    H[5, 9] = 362880 * t**5

    H[6, 5] = 43200 * t**2
    H[6, 6] = 172800 * t**3
    H[6, 7] = 453600 * t**4
    H[6, 8] = 967680 * t**5
    H[6, 9] = 1814400 * t**6

    H[7, 5] = 100800 * t**3
    H[7, 6] = 453600 * t**4
    H[7, 7] = 1270080 * t**5
    H[7, 8] = 2822400 * t**6
    H[7, 9] = 5443200 * t**7

    H[8, 5] = 201600 * t**4
    H[8, 6] = 967680 * t**5
    H[8, 7] = 2822400 * t**6
    H[8, 8] = 6451200 * t**7
    H[8, 9] = 12700800 * t**8

    H[9, 5] = 362880 * t**5
    H[9, 6] = 1814400 * t**6
    H[9, 7] = 5443200 * t**7
    H[9, 8] = 12700800 * t**8
    H[9, 9] = 25401600 * t**9
    return H


def _build_hessian_matrix(
    segment_durations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Block-diagonal QP Hessian (snap cost over 4 segments)."""
    H0 = _hessian_cost(0.0)
    blocks = [_hessian_cost(segment_durations[i]) - H0 for i in range(4)]
    return block_diag(*blocks)


def _solve_minimum_snap_qp(
    segment_durations: NDArray[np.float64],
    waypoints: NDArray[np.float64],
    pva_init: NDArray[np.float64],
    pva_final: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the equality-constrained minimum-snap QP via OSQP.

    Returns the stacked polynomial coefficient vector for all 4 segments.
    """
    A_mat = _build_constraint_matrix(segment_durations)
    b_vec = _build_constraint_rhs(waypoints, pva_init, pva_final)
    P = sparse.csc_matrix(_build_hessian_matrix(segment_durations))
    q = np.zeros(A_mat.shape[1])

    A_eq = sparse.csc_matrix(A_mat)
    b_eq = np.asarray(b_vec, dtype=np.float64)

    prob = osqp.OSQP()
    prob.setup(P, q, A_eq, b_eq, b_eq, verbose=False)
    res = prob.solve()
    return res.x


def _smooth_reference(
    t_initial: float,
    t_trajectory: float,
    t_final: float,
    sample_time: float,
    initial_pos: NDArray[np.float64],
    radius: float,
    angular_speed: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Generate a smooth 3D trajectory by stitching a QP-smoothed polynomial to
    the rotated reference, minimising snap across four segments.

    Returns ref_pos, ref_vel, ref_acc, ref_jerk, ref_snap, ref_yaw, ref_yaw_dot, ref_yaw_ddot, t_vec.
    """
    initial_pos = initial_pos.reshape((3, 1))
    traj_flight_time = np.array([[t_initial, t_trajectory, t_final, t_final]], dtype=np.float64)

    pos_init, vel_init, acc_init, jerk_init, snap_init, _, _, _ = _rotated_reference(
        traj_flight_time[:, 0], radius, angular_speed
    )
    pva_init = np.hstack((pos_init, vel_init, acc_init, jerk_init, snap_init))

    pos_final, vel_final, acc_final, jerk_final, snap_final, _, _, _ = _rotated_reference(
        traj_flight_time[:, 0] + traj_flight_time[:, 1], radius, angular_speed
    )
    pva_final = np.hstack((pos_final, vel_final, acc_final, jerk_final, snap_final))

    waypoints = np.hstack((initial_pos, pos_init, pos_final, initial_pos, initial_pos))
    traj_size = waypoints.shape[1] - 1
    number_points = 1.0 / sample_time
    number_polynomial = 9
    number_coeff = number_polynomial + 1

    t_trajectory_values = np.arange(
        traj_flight_time[0, 0],
        traj_flight_time[0, 1] + traj_flight_time[0, 0],
        sample_time,
    )
    rot_pos, rot_vel, rot_acc, rot_jerk, rot_snap, _, _, _ = _rotated_reference(
        t_trajectory_values, radius, angular_speed
    )

    coeff_x = _solve_minimum_snap_qp(
        traj_flight_time[0, :], waypoints[0, :], pva_init[0, :], pva_final[0, :]
    ).reshape(traj_size, number_coeff)
    coeff_y = _solve_minimum_snap_qp(
        traj_flight_time[0, :], waypoints[1, :], pva_init[1, :], pva_final[1, :]
    ).reshape(traj_size, number_coeff)
    coeff_z = _solve_minimum_snap_qp(
        traj_flight_time[0, :], waypoints[2, :], pva_init[2, :], pva_final[2, :]
    ).reshape(traj_size, number_coeff)

    ref_pos_x_list, ref_vel_x_list, ref_acc_x_list, ref_jerk_x_list, ref_snap_x_list = (
        [],
        [],
        [],
        [],
        [],
    )
    ref_pos_y_list, ref_vel_y_list, ref_acc_y_list, ref_jerk_y_list, ref_snap_y_list = (
        [],
        [],
        [],
        [],
        [],
    )
    ref_pos_z_list, ref_vel_z_list, ref_acc_z_list, ref_jerk_z_list, ref_snap_z_list = (
        [],
        [],
        [],
        [],
        [],
    )

    for k in range(traj_size):
        plot_time = traj_flight_time[0, k] * number_points
        time_step = traj_flight_time[0, k] / plot_time
        if k != 1:
            for j in range(int(plot_time)):
                t_j = j * time_step
                ref_pos_x_list.append(np.dot(coeff_x[k, :], position_time(t_j))[0])
                ref_vel_x_list.append(np.dot(coeff_x[k, :], velocity_time(t_j))[0])
                ref_acc_x_list.append(np.dot(coeff_x[k, :], acceleration_time(t_j))[0])
                ref_jerk_x_list.append(np.dot(coeff_x[k, :], jerk_time(t_j))[0])
                ref_snap_x_list.append(np.dot(coeff_x[k, :], snap_time(t_j))[0])

                ref_pos_y_list.append(np.dot(coeff_y[k, :], position_time(t_j))[0])
                ref_vel_y_list.append(np.dot(coeff_y[k, :], velocity_time(t_j))[0])
                ref_acc_y_list.append(np.dot(coeff_y[k, :], acceleration_time(t_j))[0])
                ref_jerk_y_list.append(np.dot(coeff_y[k, :], jerk_time(t_j))[0])
                ref_snap_y_list.append(np.dot(coeff_y[k, :], snap_time(t_j))[0])

                ref_pos_z_list.append(np.dot(coeff_z[k, :], position_time(t_j))[0])
                ref_vel_z_list.append(np.dot(coeff_z[k, :], velocity_time(t_j))[0])
                ref_acc_z_list.append(np.dot(coeff_z[k, :], acceleration_time(t_j))[0])
                ref_jerk_z_list.append(np.dot(coeff_z[k, :], jerk_time(t_j))[0])
                ref_snap_z_list.append(np.dot(coeff_z[k, :], snap_time(t_j))[0])
        else:
            for j in range(t_trajectory_values.shape[0]):
                ref_pos_x_list.append(rot_pos[0, j])
                ref_vel_x_list.append(rot_vel[0, j])
                ref_acc_x_list.append(rot_acc[0, j])
                ref_jerk_x_list.append(rot_jerk[0, j])
                ref_snap_x_list.append(rot_snap[0, j])

                ref_pos_y_list.append(rot_pos[1, j])
                ref_vel_y_list.append(rot_vel[1, j])
                ref_acc_y_list.append(rot_acc[1, j])
                ref_jerk_y_list.append(rot_jerk[1, j])
                ref_snap_y_list.append(rot_snap[1, j])

                ref_pos_z_list.append(rot_pos[2, j])
                ref_vel_z_list.append(rot_vel[2, j])
                ref_acc_z_list.append(rot_acc[2, j])
                ref_jerk_z_list.append(rot_jerk[2, j])
                ref_snap_z_list.append(rot_snap[2, j])

    ref_pos = np.vstack(
        [
            np.array(ref_pos_x_list),
            np.array(ref_pos_y_list),
            np.array(ref_pos_z_list),
        ]
    )
    ref_vel = np.vstack(
        [
            np.array(ref_vel_x_list),
            np.array(ref_vel_y_list),
            np.array(ref_vel_z_list),
        ]
    )
    ref_acc = np.vstack(
        [
            np.array(ref_acc_x_list),
            np.array(ref_acc_y_list),
            np.array(ref_acc_z_list),
        ]
    )
    ref_jerk = np.vstack(
        [
            np.array(ref_jerk_x_list),
            np.array(ref_jerk_y_list),
            np.array(ref_jerk_z_list),
        ]
    )
    ref_snap = np.vstack(
        [
            np.array(ref_snap_x_list),
            np.array(ref_snap_y_list),
            np.array(ref_snap_z_list),
        ]
    )

    N = ref_pos.shape[1]
    t_vec = np.arange(0, N * sample_time, sample_time)
    ref_yaw = np.zeros(N)
    ref_yaw_dot = np.zeros(N)
    ref_yaw_ddot = np.zeros(N)

    return ref_pos, ref_vel, ref_acc, ref_jerk, ref_snap, ref_yaw, ref_yaw_dot, ref_yaw_ddot, t_vec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_flatness_trajectory(
    params: NMPCConfig,
    initial_pos: NDArray[np.float64],
    t_initial: float,
    t_trajectory: float,
    t_final: float,
    sample_time: float,
    radius: float,
    angular_speed: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute the full flatness-based reference trajectory for NMPC tracking.

    Derives ref_pos (N,3), ref_vel, ref_acc, ref_jerk, ref_snap (N,3 each),
    ref_quat (N,4), ref_omega (N,3), ref_omega_dot (N,3), ref_thrust (N,),
    ref_torque (N,3), ref_yaw (N,), ref_yaw_dot (N,), ref_yaw_ddot (N,), and t (N,).

    Parameters
    ----------
    params : NMPCConfig
        Validated physical parameters (mass, inertia, gravity).
    initial_pos : (3,) ndarray
        Initial position [x, y, z] in world ENU frame [m].
    t_initial : float
        Duration of the initial acceleration segment [s].
    t_trajectory : float
        Duration of the main circular segment [s].
    t_final : float
        Duration of the final deceleration segment [s].
    sample_time : float
        Sampling interval [s].
    radius : float
        Radius of the circular trajectory [m].
    angular_speed : float
        Angular speed along the circle [rad/s].

    Returns
    -------
    ref_pos : (N, 3) ndarray
        Desired position [m].
    ref_vel : (N, 3) ndarray
        Desired world-frame velocity [m/s].
    ref_acc : (N, 3) ndarray
        Desired world-frame acceleration [m/s²].
    ref_jerk : (N, 3) ndarray
        Desired world-frame jerk [m/s³].
    ref_snap : (N, 3) ndarray
        Desired world-frame snap [m/s⁴].
    ref_quat : (N, 4) ndarray
        Desired orientation quaternion [qw, qx, qy, qz].
    ref_omega : (N, 3) ndarray
        Desired body-frame angular velocity [rad/s].
    ref_omega_dot : (N, 3) ndarray
        Desired body-frame angular acceleration [rad/s²].
    ref_thrust : (N,) ndarray
        Desired body-frame thrust [N].
    ref_torque : (N, 3) ndarray
        Desired body-frame torque [N·m].
    ref_yaw : (N,) ndarray
        Desired yaw angle [rad].
    ref_yaw_dot : (N,) ndarray
        Desired yaw rate [rad/s].
    ref_yaw_ddot : (N,) ndarray
        Desired yaw acceleration [rad/s²].
    t : (N,) ndarray
        Time vector [s].
    """
    m = params.physics.mass
    Jxx = params.physics.ixx
    Jyy = params.physics.iyy
    Jzz = params.physics.izz
    g = params.physics.gravity
    J = np.array([[Jxx, 0.0, 0.0], [0.0, Jyy, 0.0], [0.0, 0.0, Jzz]])

    Zw = np.array([[0.0], [0.0], [1.0]])

    ref_pos, ref_vel, ref_acc, ref_jerk, ref_snap, ref_yaw, ref_yaw_dot, ref_yaw_ddot, t = (
        _smooth_reference(
            t_initial,
            t_trajectory,
            t_final,
            sample_time,
            initial_pos,
            radius,
            angular_speed,
        )
    )

    N = ref_pos.shape[1]
    alpha = np.zeros((3, N), dtype=np.float64)
    beta = np.zeros((3, N), dtype=np.float64)
    Yc = np.zeros((3, N), dtype=np.float64)
    Xc = np.zeros((3, N), dtype=np.float64)
    Zc = np.zeros((3, N), dtype=np.float64)
    Yb = np.zeros((3, N), dtype=np.float64)
    Xb = np.zeros((3, N), dtype=np.float64)
    Zb = np.zeros((3, N), dtype=np.float64)
    ref_quat = np.zeros((4, N), dtype=np.float64)
    ref_thrust = np.zeros((1, N), dtype=np.float64)
    f_p = np.zeros((1, N), dtype=np.float64)
    ref_omega = np.zeros((3, N), dtype=np.float64)
    ref_omega_dot = np.zeros((3, N), dtype=np.float64)
    ref_torque = np.zeros((3, N), dtype=np.float64)

    for k in range(N):
        alpha[:, k] = m * ref_acc[:, k] + m * g * Zw[:, 0]
        beta[:, k] = m * ref_acc[:, k] + m * g * Zw[:, 0]

        Yc[:, k] = np.array([-np.sin(ref_yaw[k]), np.cos(ref_yaw[k]), 0.0])
        Xc[:, k] = np.array([np.cos(ref_yaw[k]), np.sin(ref_yaw[k]), 0.0])
        Zc[:, k] = np.array([0.0, 0.0, 1.0])

        Xb[:, k] = np.cross(Yc[:, k], alpha[:, k])
        Xb[:, k] /= np.linalg.norm(Xb[:, k])
        Yb[:, k] = np.cross(beta[:, k], Xb[:, k])
        Yb[:, k] /= np.linalg.norm(Yb[:, k])
        Zb[:, k] = np.cross(Xb[:, k], Yb[:, k])

        R_d = np.array(
            [
                [Xb[0, k], Yb[0, k], Zb[0, k]],
                [Xb[1, k], Yb[1, k], Zb[1, k]],
                [Xb[2, k], Yb[2, k], Zb[2, k]],
            ]
        )
        r_d = R.from_matrix(R_d)
        quad_d_aux = r_d.as_quat()
        ref_quat[:, k] = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])
        if k > 0:
            if np.dot(ref_quat[:, k], ref_quat[:, k - 1]) < 0:
                ref_quat[:, k] = -ref_quat[:, k]
        ref_quat[:, k] /= np.linalg.norm(ref_quat[:, k])

        ref_thrust[0, k] = np.dot(Zb[:, k], m * ref_acc[:, k] + m * g * Zw[:, 0])

        b1 = m * np.dot(Xb[:, k], ref_jerk[:, k])
        b2 = -m * np.dot(Yb[:, k], ref_jerk[:, k])
        b3 = ref_yaw_dot[k] * np.dot(Xc[:, k], Xb[:, k])
        b = np.array([[b1], [b2], [b3]], dtype=np.float64)

        a11, a12, a13 = 0.0, float(ref_thrust[0, k]), 0.0
        a21, a22, a23 = float(ref_thrust[0, k]), 0.0, 0.0
        a31 = 0.0
        a32 = float(-np.dot(Yc[:, k], Zb[:, k]))
        a33 = float(np.linalg.norm(np.cross(Yc[:, k], Zb[:, k])))

        A_mat = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], dtype=np.float64)
        A_inv = np.linalg.inv(A_mat)

        ref_omega[:, k] = (A_inv @ b)[:, 0]

        f_p[0, k] = m * np.dot(Zb[:, k], ref_jerk[:, k])

        wx, wy, wz = ref_omega[0, k], ref_omega[1, k], ref_omega[2, k]
        chi_1 = ref_yaw_ddot[k] * np.dot(Xc[:, k], Xb[:, k])
        chi_2 = -2 * ref_yaw_dot[k] * wy * np.dot(Xc[:, k], Zb[:, k])
        chi_3 = -wy * wx * np.dot(Yc[:, k], Yb[:, k])
        chi_4 = 2 * ref_yaw_dot[k] * wz * np.dot(Xc[:, k], Yb[:, k])
        chi_5 = -wz * wx * np.dot(Yc[:, k], Zb[:, k])
        chi = chi_1 + chi_2 + chi_3 + chi_4 + chi_5

        B1 = m * np.dot(Xb[:, k], ref_snap[:, k]) - ref_thrust[0, k] * wx * wz - 2 * f_p[0, k] * wy
        B2 = -m * np.dot(Yb[:, k], ref_snap[:, k]) - 2 * f_p[0, k] * wx + ref_thrust[0, k] * wy * wz
        B3 = chi
        B = np.array([[B1], [B2], [B3]], dtype=np.float64)

        ref_omega_dot[:, k] = (A_inv @ B)[:, 0]
        ref_torque[:, k] = J @ ref_omega_dot[:, k] + np.cross(ref_omega[:, k], J @ ref_omega[:, k])

    return (
        ref_pos.T,
        ref_vel.T,
        ref_acc.T,
        ref_jerk.T,
        ref_snap.T,
        ref_quat.T,
        ref_omega.T,
        ref_omega_dot.T,
        ref_thrust.ravel(),
        ref_torque.T,
        ref_yaw,
        ref_yaw_dot,
        ref_yaw_ddot,
        t,
    )
