"""CasADi Functions implementing the full differential flatness decomposition.

Port of the flatness map from ``nmpc/planner.py`` (lines 656-734) to
CasADi MX, accepting NMPC physical parameters as symbolic inputs.

Computes body-frame orientation, angular velocity, angular acceleration,
thrust, and torque from flat outputs (acceleration, jerk, snap,
yaw derivatives).

Public API:
    make_flatness_casadi  —  factory returning ca.Function (17 inputs, 14 outputs)
"""

from __future__ import annotations

import casadi as ca

__all__ = ["make_flatness_casadi"]

_EPS = 1e-10


def make_flatness_casadi() -> ca.Function:
    """CasADi full flatness decomposition with snap and yaw_ddot.

    Inputs (scalars, each 1x1):

    Name       Description
    =========  ============================================
    acc_x      world-frame acceleration X [m/s²]
    acc_y      world-frame acceleration Y [m/s²]
    acc_z      world-frame acceleration Z [m/s²]
    jerk_x     world-frame jerk X [m/s³]
    jerk_y     world-frame jerk Y [m/s³]
    jerk_z     world-frame jerk Z [m/s³]
    snap_x     world-frame snap X [m/s⁴]
    snap_y     world-frame snap Y [m/s⁴]
    snap_z     world-frame snap Z [m/s⁴]
    yaw        yaw angle [rad]
    yaw_dot    yaw rate [rad/s]
    yaw_ddot   yaw acceleration [rad/s²]
    mass       vehicle mass [kg]
    Ixx        inertia about body X [kg·m²]
    Iyy        inertia about body Y [kg·m²]
    Izz        inertia about body Z [kg·m²]
    gravity    gravitational acceleration [m/s²]
    =========  ============================================

    Outputs (scalars, each 1x1):

    Name       Description
    =========  ============================================
    qw         orientation quaternion w
    qx         orientation quaternion x
    qy         orientation quaternion y
    qz         orientation quaternion z
    omega_x    body-frame angular velocity X [rad/s]
    omega_y    body-frame angular velocity Y [rad/s]
    omega_z    body-frame angular velocity Z [rad/s]
    omegad_x   body-frame angular acceleration X [rad/s²]
    omegad_y   body-frame angular acceleration Y [rad/s²]
    omegad_z   body-frame angular acceleration Z [rad/s²]
    thrust     body-frame thrust [N]
    tau_x      body-frame torque X [N·m]
    tau_y      body-frame torque Y [N·m]
    tau_z      body-frame torque Z [N·m]
    =========  ============================================

    @return Compiled CasADi Function
    """
    acc_x = ca.MX.sym("acc_x", 1, 1)
    acc_y = ca.MX.sym("acc_y", 1, 1)
    acc_z = ca.MX.sym("acc_z", 1, 1)
    jerk_x = ca.MX.sym("jerk_x", 1, 1)
    jerk_y = ca.MX.sym("jerk_y", 1, 1)
    jerk_z = ca.MX.sym("jerk_z", 1, 1)
    snap_x = ca.MX.sym("snap_x", 1, 1)
    snap_y = ca.MX.sym("snap_y", 1, 1)
    snap_z = ca.MX.sym("snap_z", 1, 1)
    yaw = ca.MX.sym("yaw", 1, 1)
    yaw_dot = ca.MX.sym("yaw_dot", 1, 1)
    yaw_ddot = ca.MX.sym("yaw_ddot", 1, 1)
    mass = ca.MX.sym("mass", 1, 1)
    Ixx = ca.MX.sym("Ixx", 1, 1)
    Iyy = ca.MX.sym("Iyy", 1, 1)
    Izz = ca.MX.sym("Izz", 1, 1)
    gravity = ca.MX.sym("gravity", 1, 1)

    acc = ca.vertcat(acc_x, acc_y, acc_z)
    jerk = ca.vertcat(jerk_x, jerk_y, jerk_z)
    snap = ca.vertcat(snap_x, snap_y, snap_z)

    J = ca.MX.zeros(3, 3)
    J[0, 0] = Ixx
    J[1, 1] = Iyy
    J[2, 2] = Izz

    Zw = ca.DM([0.0, 0.0, 1.0])
    alpha = mass * acc + mass * gravity * Zw

    Yc = ca.vertcat(-ca.sin(yaw), ca.cos(yaw), 0.0)
    Xc = ca.vertcat(ca.cos(yaw), ca.sin(yaw), 0.0)

    Xb = ca.cross(Yc, alpha)
    norm_Xb = ca.sqrt(ca.dot(Xb, Xb) + _EPS)
    Xb = Xb / norm_Xb

    Yb = ca.cross(alpha, Xb)
    norm_Yb = ca.sqrt(ca.dot(Yb, Yb) + _EPS)
    Yb = Yb / norm_Yb

    Zb = ca.cross(Xb, Yb)

    qw_sq = ca.fmax(0.25 * (1.0 + Xb[0] + Yb[1] + Zb[2]), _EPS)
    qw = ca.sqrt(qw_sq)
    s = 0.5 / (qw + _EPS)
    qx = (Yb[2] - Zb[1]) * s
    qy = (Zb[0] - Xb[2]) * s
    qz = (Xb[1] - Yb[0]) * s

    norm_quat = ca.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + _EPS)
    qw = qw / norm_quat
    qx = qx / norm_quat
    qy = qy / norm_quat
    qz = qz / norm_quat

    thrust = ca.dot(Zb, alpha)

    b1 = mass * ca.dot(Xb, jerk)
    b2 = -mass * ca.dot(Yb, jerk)
    b3 = yaw_dot * ca.dot(Xc, Xb)

    cross_Yc_Zb = ca.cross(Yc, Zb)
    a32 = -ca.dot(Yc, Zb)
    a33 = ca.sqrt(ca.dot(cross_Yc_Zb, cross_Yc_Zb) + _EPS)

    A = ca.vertcat(
        ca.horzcat(0.0, thrust, 0.0),
        ca.horzcat(thrust, 0.0, 0.0),
        ca.horzcat(0.0, a32, a33),
    )
    A_inv = ca.inv(A)

    b_col = ca.vertcat(b1, b2, b3)
    omega = A_inv @ b_col

    wx = omega[0]
    wy = omega[1]
    wz = omega[2]

    f_p = mass * ca.dot(Zb, jerk)

    chi = (
        yaw_ddot * ca.dot(Xc, Xb)
        - 2.0 * yaw_dot * wy * ca.dot(Xc, Zb)
        - wy * wx * ca.dot(Yc, Yb)
        + 2.0 * yaw_dot * wz * ca.dot(Xc, Yb)
        - wz * wx * ca.dot(Yc, Zb)
    )

    B1 = mass * ca.dot(Xb, snap) - thrust * wx * wz - 2.0 * f_p * wy
    B2 = -mass * ca.dot(Yb, snap) - 2.0 * f_p * wx + thrust * wy * wz
    B3 = chi

    B_col = ca.vertcat(B1, B2, B3)
    omega_dot = A_inv @ B_col

    J_omega = ca.vertcat(Ixx * wx, Iyy * wy, Izz * wz)
    torque = J @ omega_dot + ca.cross(omega, J_omega)

    input_syms = [
        acc_x,
        acc_y,
        acc_z,
        jerk_x,
        jerk_y,
        jerk_z,
        snap_x,
        snap_y,
        snap_z,
        yaw,
        yaw_dot,
        yaw_ddot,
        mass,
        Ixx,
        Iyy,
        Izz,
        gravity,
    ]
    input_names = [
        "acc_x",
        "acc_y",
        "acc_z",
        "jerk_x",
        "jerk_y",
        "jerk_z",
        "snap_x",
        "snap_y",
        "snap_z",
        "yaw",
        "yaw_dot",
        "yaw_ddot",
        "mass",
        "Ixx",
        "Iyy",
        "Izz",
        "gravity",
    ]
    output_names = [
        "qw",
        "qx",
        "qy",
        "qz",
        "omega_x",
        "omega_y",
        "omega_z",
        "omegad_x",
        "omegad_y",
        "omegad_z",
        "thrust",
        "tau_x",
        "tau_y",
        "tau_z",
    ]

    f = ca.Function(
        "full_flatness_decomposition",
        input_syms,
        [
            qw,
            qx,
            qy,
            qz,
            wx,
            wy,
            wz,
            omega_dot[0],
            omega_dot[1],
            omega_dot[2],
            thrust,
            torque[0],
            torque[1],
            torque[2],
        ],
        input_names,
        output_names,
    )

    f.description = (
        "Full flatness decomposition: maps (acc, jerk, snap, yaw, yaw_dot, yaw_ddot) "
        "and physical parameters to body-frame orientation, angular velocity, "
        "angular acceleration, thrust, and torque."
    )

    return f
