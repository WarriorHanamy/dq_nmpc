"""CasADi Functions implementing the differential flatness decomposition.

Port of the flatness map from ``nmpc/planner.py`` (lines 656-734) to
CasADi MX, accepting NMPC physical parameters as symbolic inputs.

Two variants:

* ``make_flatness_casadi`` — takes yaw and yaw_dot as symbolic inputs
* ``make_zero_yaw_flatness_casadi`` — yaw and yaw_dot hardcoded to zero
  (fixed-heading mode, fewer inputs, smaller compiled function)

Public API:
    make_flatness_casadi
    make_zero_yaw_flatness_casadi
"""

from __future__ import annotations

import casadi as ca

__all__ = ["make_flatness_casadi", "make_zero_yaw_flatness_casadi"]

_EPS = 1e-10


def _build_flatness_expr(
    acc: ca.MX,
    jerk: ca.MX,
    yaw: ca.MX | ca.DM,
    yaw_dot: ca.MX | ca.DM,
    mass: ca.MX,
    gravity: ca.MX | ca.DM,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    """Symbolic flatness decomposition core — shared by both factories.

    @param[in] acc       (3,1) MX — world-frame acceleration
    @param[in] jerk      (3,1) MX — world-frame jerk
    @param[in] yaw        (1,1) MX or DM — yaw angle
    @param[in] yaw_dot    (1,1) MX or DM — yaw rate
    @param[in] mass       (1,1) MX — vehicle mass
    @param[in] gravity    (1,1) MX or DM — gravity
    @return (quat(4,1), omega(3,1), thrust(1,1))
    """
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

    quat = ca.vertcat(qw, qx, qy, qz)

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

    return quat, omega, thrust


def make_flatness_casadi() -> ca.Function:
    """CasADi flatness decomposition with yaw and yaw_dot as symbolic inputs.

    Inputs (scalars, each 1x1):
        acc_x, acc_y, acc_z, jerk_x, jerk_y, jerk_z,
        yaw, yaw_dot, mass, Ixx, Iyy, Izz, gravity

    Outputs (scalars, each 1x1):
        qw, qx, qy, qz, omega_x, omega_y, omega_z, thrust

    @return Compiled CasADi Function
    """
    acc_x = ca.MX.sym("acc_x", 1, 1)
    acc_y = ca.MX.sym("acc_y", 1, 1)
    acc_z = ca.MX.sym("acc_z", 1, 1)
    jerk_x = ca.MX.sym("jerk_x", 1, 1)
    jerk_y = ca.MX.sym("jerk_y", 1, 1)
    jerk_z = ca.MX.sym("jerk_z", 1, 1)
    yaw = ca.MX.sym("yaw", 1, 1)
    yaw_dot = ca.MX.sym("yaw_dot", 1, 1)
    mass = ca.MX.sym("mass", 1, 1)
    Ixx = ca.MX.sym("Ixx", 1, 1)
    Iyy = ca.MX.sym("Iyy", 1, 1)
    Izz = ca.MX.sym("Izz", 1, 1)
    gravity = ca.MX.sym("gravity", 1, 1)

    acc = ca.vertcat(acc_x, acc_y, acc_z)
    jerk = ca.vertcat(jerk_x, jerk_y, jerk_z)

    quat, omega, thrust = _build_flatness_expr(acc, jerk, yaw, yaw_dot, mass, gravity)

    input_syms = [
        acc_x,
        acc_y,
        acc_z,
        jerk_x,
        jerk_y,
        jerk_z,
        yaw,
        yaw_dot,
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
        "yaw",
        "yaw_dot",
        "mass",
        "Ixx",
        "Iyy",
        "Izz",
        "gravity",
    ]
    output_names = ["qw", "qx", "qy", "qz", "omega_x", "omega_y", "omega_z", "thrust"]

    f = ca.Function(
        "flatness_decomposition",
        input_syms,
        [quat[0], quat[1], quat[2], quat[3], omega[0], omega[1], omega[2], thrust],
        input_names,
        output_names,
    )
    f.description = (
        "Flatness decomposition with variable yaw: maps (acc, jerk, yaw, yaw_dot) "
        "and physical parameters to body-frame orientation, angular velocity, and thrust."
    )
    return f


def make_zero_yaw_flatness_casadi() -> ca.Function:
    """CasADi flatness decomposition with yaw ≡ 0 and yaw_dot ≡ 0.

    Zero yaw means the drone keeps a fixed heading (body X = world X).
    Compared to ``make_flatness_casadi``, the ``yaw`` and ``yaw_dot``
    inputs are removed — the function has only 11 inputs.

    Inputs (scalars, each 1x1):
        acc_x, acc_y, acc_z, jerk_x, jerk_y, jerk_z,
        mass, Ixx, Iyy, Izz, gravity

    Outputs (scalars, each 1x1):
        qw, qx, qy, qz, omega_x, omega_y, omega_z, thrust

    @return Compiled CasADi Function
    """
    acc_x = ca.MX.sym("acc_x", 1, 1)
    acc_y = ca.MX.sym("acc_y", 1, 1)
    acc_z = ca.MX.sym("acc_z", 1, 1)
    jerk_x = ca.MX.sym("jerk_x", 1, 1)
    jerk_y = ca.MX.sym("jerk_y", 1, 1)
    jerk_z = ca.MX.sym("jerk_z", 1, 1)
    mass = ca.MX.sym("mass", 1, 1)
    Ixx = ca.MX.sym("Ixx", 1, 1)
    Iyy = ca.MX.sym("Iyy", 1, 1)
    Izz = ca.MX.sym("Izz", 1, 1)
    gravity = ca.MX.sym("gravity", 1, 1)

    acc = ca.vertcat(acc_x, acc_y, acc_z)
    jerk = ca.vertcat(jerk_x, jerk_y, jerk_z)

    yaw = ca.MX(0.0)
    yaw_dot = ca.MX(0.0)

    quat, omega, thrust = _build_flatness_expr(acc, jerk, yaw, yaw_dot, mass, gravity)

    input_syms = [
        acc_x,
        acc_y,
        acc_z,
        jerk_x,
        jerk_y,
        jerk_z,
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
        "mass",
        "Ixx",
        "Iyy",
        "Izz",
        "gravity",
    ]
    output_names = ["qw", "qx", "qy", "qz", "omega_x", "omega_y", "omega_z", "thrust"]

    f = ca.Function(
        "zero_yaw_flatness_decomposition",
        input_syms,
        [quat[0], quat[1], quat[2], quat[3], omega[0], omega[1], omega[2], thrust],
        input_names,
        output_names,
    )
    f.description = (
        "Flatness decomposition with fixed heading (yaw≡0): "
        "maps (acc, jerk) and physical parameters to body-frame "
        "orientation, angular velocity, and thrust."
    )
    return f
