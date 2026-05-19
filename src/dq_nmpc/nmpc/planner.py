"""Flatness-based reference trajectory planner.

Pipeline:
  _circular_reference  →  _rotated_reference  →  _smooth_reference  →  compute_flatness_states
  (flat outputs on       (expm 3D rotation)     (min-snap QP          (full flatness: pose,
   a circular path)                               smoothing)            twist, force, torque)

Public API:
  compute_flatness_states  —  the only function intended for external callers.
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

__all__ = ["compute_flatness_states"]


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

    Returns 3xN arrays for position, velocity, acceleration, jerk, snap
    and 1D arrays for yaw angle and its derivatives (all zero).
    """
    cos_wt = np.cos(angular_speed * t)
    sin_wt = np.sin(angular_speed * t)

    xd = radius * cos_wt
    yd = radius * sin_wt
    zd = np.zeros_like(t)

    xd_p = -radius * angular_speed * sin_wt
    yd_p = radius * angular_speed * cos_wt
    zd_p = np.zeros_like(t)

    xd_pp = -radius * angular_speed**2 * cos_wt
    yd_pp = -radius * angular_speed**2 * sin_wt
    zd_pp = np.zeros_like(t)

    xd_ppp = radius * angular_speed**3 * sin_wt
    yd_ppp = -radius * angular_speed**3 * cos_wt
    zd_ppp = np.zeros_like(t)

    xd_pppp = radius * angular_speed**4 * cos_wt
    yd_pppp = radius * angular_speed**4 * sin_wt
    zd_pppp = np.zeros_like(t)

    theta = np.zeros_like(t)
    theta_p = np.zeros_like(t)

    hd = np.vstack((xd, yd, zd))
    hd_p = np.vstack((xd_p, yd_p, zd_p))
    hd_pp = np.vstack((xd_pp, yd_pp, zd_pp))
    hd_ppp = np.vstack((xd_ppp, yd_ppp, zd_ppp))
    hd_pppp = np.vstack((xd_pppp, yd_pppp, zd_pppp))

    return hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_p


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
    p, theta, p_d, theta_d, p_dd, p_ddd, p_dddd, _theta_dd = _circular_reference(
        t, radius, angular_speed
    )
    a = np.pi / 2
    b = 0.05

    N = p_d.shape[1]
    r = np.zeros((3, N), dtype=np.float64)
    r_d = np.zeros((3, N), dtype=np.float64)
    r_dd = np.zeros((3, N), dtype=np.float64)
    r_ddd = np.zeros((3, N), dtype=np.float64)
    r_dddd = np.zeros((3, N), dtype=np.float64)

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

        pk = p[:, k]
        r[:, k] = expm_w @ pk
        r_d[:, k] = expm_w @ (p_d[:, k] + sw_d @ pk)
        r_dd[:, k] = expm_w @ (sw_d2 @ pk + 2 * sw_d @ p_d[:, k] + p_dd[:, k] + sw_dd @ pk)
        r_ddd[:, k] = expm_w @ (
            p_ddd[:, k]
            + sw_ddd @ pk
            + 3 * sw_dd @ p_d[:, k]
            + 3 * sw_d @ p_dd[:, k]
            + sw_d3 @ pk
            + 3 * sw_d2 @ p_d[:, k]
            + 3 * sw_d @ sw_dd @ pk
        )
        r_dddd[:, k] = expm_w @ (
            p_dddd[:, k]
            + sw_dddd @ pk
            + 4 * sw_ddd @ p_d[:, k]
            + 6 * sw_dd @ p_dd[:, k]
            + 4 * sw_d @ p_ddd[:, k]
            + sw_d4 @ pk
            + 3 * sw_dd2 @ pk
            + 4 * sw_d3 @ p_d[:, k]
            + 6 * sw_d2 @ p_dd[:, k]
            + 6 * sw_d2 @ sw_dd @ pk
            + 4 * sw_d @ sw_ddd @ pk
            + 12 * sw_d @ sw_dd @ p_d[:, k]
        )

    h0 = np.vstack(
        (
            np.zeros_like(t),
            np.zeros_like(t),
            4.0 * np.ones_like(t),
        )
    )
    r += h0
    return r, r_d, r_dd, r_ddd, r_dddd, theta, theta_d, _theta_dd


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
    h_init: NDArray[np.float64],
    h_final: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble the RHS vector for QP equality constraints."""
    b_1 = np.array([waypoints[0], 0, 0, 0, 0], dtype=np.float64)
    b_2 = np.array([waypoints[1], 0, 0, 0, 0], dtype=np.float64)
    b_3 = np.array([waypoints[2], 0, 0, 0, 0], dtype=np.float64)
    b_4 = np.array([waypoints[3], 0, 0, 0, 0], dtype=np.float64)
    b_5 = np.array([waypoints[4], 0, 0, 0, 0], dtype=np.float64)
    b_6 = np.array([waypoints[1], waypoints[2], waypoints[3]], dtype=np.float64)

    b_first = np.array([h_init[1], h_init[2], h_init[3], h_init[4]], dtype=np.float64)
    b_second = np.array([h_final[1], h_final[2], h_final[3], h_final[4]], dtype=np.float64)
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
    h_init: NDArray[np.float64],
    h_final: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the equality-constrained minimum-snap QP via OSQP.

    Returns the stacked polynomial coefficient vector for all 4 segments.
    """
    A_mat = _build_constraint_matrix(segment_durations)
    b_vec = _build_constraint_rhs(waypoints, h_init, h_final)
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

    Returns h, h_d, h_dd, h_ddd, h_dddd, theta, theta_d, theta_dd, t_vec.
    """
    initial_pos = initial_pos.reshape((3, 1))
    traj_flight_time = np.array([[t_initial, t_trajectory, t_final, t_final]], dtype=np.float64)

    r_init, r_d_init, r_dd_init, r_ddd_init, r_dddd_init, _, _, _ = _rotated_reference(
        traj_flight_time[:, 0], radius, angular_speed
    )
    h_init = np.hstack((r_init, r_d_init, r_dd_init, r_ddd_init, r_dddd_init))

    r_final, r_d_final, r_dd_final, r_ddd_final, r_dddd_final, _, _, _ = _rotated_reference(
        traj_flight_time[:, 0] + traj_flight_time[:, 1], radius, angular_speed
    )
    h_final = np.hstack((r_final, r_d_final, r_dd_final, r_ddd_final, r_dddd_final))

    waypoints_1 = np.hstack((initial_pos, r_init, r_final, initial_pos, initial_pos))
    traj_size = waypoints_1.shape[1] - 1
    number_points = 1.0 / sample_time
    number_polynomial = 9
    number_coeff = number_polynomial + 1

    t_trajectory_values = np.arange(
        traj_flight_time[0, 0],
        traj_flight_time[0, 1] + traj_flight_time[0, 0],
        sample_time,
    )
    r, r_d, r_dd, r_ddd, r_dddd, _, _, _ = _rotated_reference(
        t_trajectory_values, radius, angular_speed
    )

    coeff_x = _solve_minimum_snap_qp(
        traj_flight_time[0, :], waypoints_1[0, :], h_init[0, :], h_final[0, :]
    ).reshape(traj_size, number_coeff)
    coeff_y = _solve_minimum_snap_qp(
        traj_flight_time[0, :], waypoints_1[1, :], h_init[1, :], h_final[1, :]
    ).reshape(traj_size, number_coeff)
    coeff_z = _solve_minimum_snap_qp(
        traj_flight_time[0, :], waypoints_1[2, :], h_init[2, :], h_final[2, :]
    ).reshape(traj_size, number_coeff)

    pos_x, vel_x, acc_x, jerk_x, snap_x = [], [], [], [], []
    pos_y, vel_y, acc_y, jerk_y, snap_y = [], [], [], [], []
    pos_z, vel_z, acc_z, jerk_z, snap_z = [], [], [], [], []

    for k in range(traj_size):
        plot_time = traj_flight_time[0, k] * number_points
        time_step = traj_flight_time[0, k] / plot_time
        if k != 1:
            for j in range(int(plot_time)):
                t_j = j * time_step
                pos_x.append(np.dot(coeff_x[k, :], position_time(t_j))[0])
                vel_x.append(np.dot(coeff_x[k, :], velocity_time(t_j))[0])
                acc_x.append(np.dot(coeff_x[k, :], acceleration_time(t_j))[0])
                jerk_x.append(np.dot(coeff_x[k, :], jerk_time(t_j))[0])
                snap_x.append(np.dot(coeff_x[k, :], snap_time(t_j))[0])

                pos_y.append(np.dot(coeff_y[k, :], position_time(t_j))[0])
                vel_y.append(np.dot(coeff_y[k, :], velocity_time(t_j))[0])
                acc_y.append(np.dot(coeff_y[k, :], acceleration_time(t_j))[0])
                jerk_y.append(np.dot(coeff_y[k, :], jerk_time(t_j))[0])
                snap_y.append(np.dot(coeff_y[k, :], snap_time(t_j))[0])

                pos_z.append(np.dot(coeff_z[k, :], position_time(t_j))[0])
                vel_z.append(np.dot(coeff_z[k, :], velocity_time(t_j))[0])
                acc_z.append(np.dot(coeff_z[k, :], acceleration_time(t_j))[0])
                jerk_z.append(np.dot(coeff_z[k, :], jerk_time(t_j))[0])
                snap_z.append(np.dot(coeff_z[k, :], snap_time(t_j))[0])
        else:
            for j in range(t_trajectory_values.shape[0]):
                pos_x.append(r[0, j])
                vel_x.append(r_d[0, j])
                acc_x.append(r_dd[0, j])
                jerk_x.append(r_ddd[0, j])
                snap_x.append(r_dddd[0, j])

                pos_y.append(r[1, j])
                vel_y.append(r_d[1, j])
                acc_y.append(r_dd[1, j])
                jerk_y.append(r_ddd[1, j])
                snap_y.append(r_dddd[1, j])

                pos_z.append(r[2, j])
                vel_z.append(r_d[2, j])
                acc_z.append(r_dd[2, j])
                jerk_z.append(r_ddd[2, j])
                snap_z.append(r_dddd[2, j])

    h = np.vstack(
        [
            np.array(pos_x),
            np.array(pos_y),
            np.array(pos_z),
        ]
    )
    h_d = np.vstack(
        [
            np.array(vel_x),
            np.array(vel_y),
            np.array(vel_z),
        ]
    )
    h_dd = np.vstack(
        [
            np.array(acc_x),
            np.array(acc_y),
            np.array(acc_z),
        ]
    )
    h_ddd = np.vstack(
        [
            np.array(jerk_x),
            np.array(jerk_y),
            np.array(jerk_z),
        ]
    )
    h_dddd = np.vstack(
        [
            np.array(snap_x),
            np.array(snap_y),
            np.array(snap_z),
        ]
    )

    N = h.shape[1]
    t_vec = np.arange(0, N * sample_time, sample_time)
    theta = np.zeros(N)
    theta_d = np.zeros(N)
    theta_dd = np.zeros(N)

    return h, h_d, h_dd, h_ddd, h_dddd, theta, theta_d, theta_dd, t_vec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_flatness_states(
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
]:
    """Compute the full flatness-based reference trajectory for NMPC tracking.

    Derives position (3xN), velocity, acceleration, jerk, snap, orientation
    quaternion (4xN), body angular velocity (3xN), body angular acceleration
    (3xN), thrust (1xN), body torque (3xN), and time vector (N,).

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
    hd : (3, N) ndarray
        Desired position [m].
    hd_d : (3, N) ndarray
        Desired world-frame velocity [m/s].
    hd_dd : (3, N) ndarray
        Desired world-frame acceleration [m/s²].
    hd_ddd : (3, N) ndarray
        Desired world-frame jerk [m/s³].
    hd_dddd : (3, N) ndarray
        Desired world-frame snap [m/s⁴].
    qd : (4, N) ndarray
        Desired orientation quaternion [qw, qx, qy, qz].
    w_d : (3, N) ndarray
        Desired body-frame angular velocity [rad/s].
    w_d_d : (3, N) ndarray
        Desired body-frame angular acceleration [rad/s²].
    f_d : (1, N) ndarray
        Desired body-frame thrust force [N].
    M_d : (3, N) ndarray
        Desired body-frame torque [N·m].
    t : (N,) ndarray
        Time vector [s].
    """
    m = params.mass
    Jxx = params.ixx
    Jyy = params.iyy
    Jzz = params.izz
    g = params.gravity
    J = np.array([[Jxx, 0.0, 0.0], [0.0, Jyy, 0.0], [0.0, 0.0, Jzz]])

    Zw = np.array([[0.0], [0.0], [1.0]])

    hd, hd_d, hd_dd, hd_ddd, hd_dddd, theta, theta_d, theta_dd, t = _smooth_reference(
        t_initial,
        t_trajectory,
        t_final,
        sample_time,
        initial_pos,
        radius,
        angular_speed,
    )

    N = hd.shape[1]
    alpha = np.zeros((3, N), dtype=np.float64)
    beta = np.zeros((3, N), dtype=np.float64)
    Yc = np.zeros((3, N), dtype=np.float64)
    Xc = np.zeros((3, N), dtype=np.float64)
    Zc = np.zeros((3, N), dtype=np.float64)
    Yb = np.zeros((3, N), dtype=np.float64)
    Xb = np.zeros((3, N), dtype=np.float64)
    Zb = np.zeros((3, N), dtype=np.float64)
    q = np.zeros((4, N), dtype=np.float64)
    f = np.zeros((1, N), dtype=np.float64)
    f_p = np.zeros((1, N), dtype=np.float64)
    w = np.zeros((3, N), dtype=np.float64)
    w_p = np.zeros((3, N), dtype=np.float64)
    M = np.zeros((3, N), dtype=np.float64)

    for k in range(N):
        alpha[:, k] = m * hd_dd[:, k] + m * g * Zw[:, 0]
        beta[:, k] = m * hd_dd[:, k] + m * g * Zw[:, 0]

        Yc[:, k] = np.array([-np.sin(theta[k]), np.cos(theta[k]), 0.0])
        Xc[:, k] = np.array([np.cos(theta[k]), np.sin(theta[k]), 0.0])
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
        q[:, k] = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])
        if k > 0:
            if np.dot(q[:, k], q[:, k - 1]) < 0:
                q[:, k] = -q[:, k]
        q[:, k] /= np.linalg.norm(q[:, k])

        f[:, k] = np.dot(Zb[:, k], m * hd_dd[:, k] + m * g * Zw[:, 0])

        b1 = m * np.dot(Xb[:, k], hd_ddd[:, k])
        b2 = -m * np.dot(Yb[:, k], hd_ddd[:, k])
        b3 = theta_d[k] * np.dot(Xc[:, k], Xb[:, k])
        b = np.array([[b1], [b2], [b3]], dtype=np.float64)

        a11, a12, a13 = 0.0, float(f[:, k]), 0.0
        a21, a22, a23 = float(f[:, k]), 0.0, 0.0
        a31 = 0.0
        a32 = float(-np.dot(Yc[:, k], Zb[:, k]))
        a33 = float(np.linalg.norm(np.cross(Yc[:, k], Zb[:, k])))

        A_mat = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], dtype=np.float64)
        A_inv = np.linalg.inv(A_mat)

        w[:, k] = (A_inv @ b)[:, 0]

        f_p[:, k] = m * np.dot(Zb[:, k], hd_ddd[:, k])

        wx, wy, wz = w[0, k], w[1, k], w[2, k]
        chi_1 = theta_dd[k] * np.dot(Xc[:, k], Xb[:, k])
        chi_2 = -2 * theta_d[k] * wy * np.dot(Xc[:, k], Zb[:, k])
        chi_3 = -wy * wx * np.dot(Yc[:, k], Yb[:, k])
        chi_4 = 2 * theta_d[k] * wz * np.dot(Xc[:, k], Yb[:, k])
        chi_5 = -wz * wx * np.dot(Yc[:, k], Zb[:, k])
        chi = chi_1 + chi_2 + chi_3 + chi_4 + chi_5

        B1 = m * np.dot(Xb[:, k], hd_dddd[:, k]) - f[:, k] * wx * wz - 2 * f_p[:, k] * wy
        B2 = -m * np.dot(Yb[:, k], hd_dddd[:, k]) - 2 * f_p[:, k] * wx + f[:, k] * wy * wz
        B3 = chi
        B = np.array([[B1], [B2], [B3]], dtype=np.float64)

        w_p[:, k] = (A_inv @ B)[:, 0]
        M[:, k] = J @ w_p[:, k] + np.cross(w[:, k], J @ w[:, k])

    return hd, hd_d, hd_dd, hd_ddd, hd_dddd, q, w, w_p, f, M, t
