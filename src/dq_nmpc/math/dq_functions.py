"""CasADi Function factories for dual-quaternion operations and NMPC cost terms.

Compiled CasADi Functions wrapping symbolic algebra.  Every cost function
is a standalone CasADi Function — testable, composable, reusable.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
from casadi import Function

__all__ = [
    # Cost Terms
    "make_angular_velocity_error_cost",
    "make_control_error_cost",
    "make_pose_error_cost",
    "make_inertial_velocity_error_cost",
    "make_quat_error_cost",
    "make_translation_error_cost",
    # Helpers
    "make_body_to_inertial_rotation",
    "make_dualquat_acceleration",
    "make_dualquat_from_pose",
    "make_dualquat_kinematics",
    "make_dualquat_mul_conj",
    "make_inertial_to_body_rotation",
    # NumPy helper
    "dualquat_from_pose_np",
]

_EPS = 1e-10

# ---------------------------------------------------------------------------
# Pose / state factories
# ---------------------------------------------------------------------------


def make_dualquat_from_pose() -> Function:
    qw = ca.MX.sym("qw", 1, 1)
    qx = ca.MX.sym("qx", 1, 1)
    qy = ca.MX.sym("qy", 1, 1)
    qz = ca.MX.sym("qz", 1, 1)

    tx = ca.MX.sym("tx", 1, 1)
    ty = ca.MX.sym("ty", 1, 1)
    tz = ca.MX.sym("tz", 1, 1)

    q_dual = 0.5 * ca.vertcat(
        -(qx * tx + qy * ty + qz * tz),
        qw * tx + qy * tz - qz * ty,
        qw * ty - qx * tz + qz * tx,
        qw * tz + qx * ty - qy * tx,
    )
    dq_full = ca.vertcat(qw, qx, qy, qz, q_dual)

    f = Function(
        "dualquat_from_pose",
        [qw, qx, qy, qz, tx, ty, tz],
        [dq_full],
        ["qw", "qx", "qy", "qz", "tx", "ty", "tz"],
        ["dq"],
    )
    f.description = "Build a dual quaternion (8,) from quaternion and translation."
    return f


def make_dualquat_mul_conj() -> Function:
    dq_desired = ca.MX.sym("dq_desired", 8, 1)
    dq_current = ca.MX.sym("dq_current", 8, 1)

    dq_desired_conjugate = ca.vertcat(
        dq_desired[0],
        -dq_desired[1],
        -dq_desired[2],
        -dq_desired[3],
        dq_desired[4],
        -dq_desired[5],
        -dq_desired[6],
        -dq_desired[7],
    )
    real_conjugate = dq_desired_conjugate[0:4]
    dual_conjugate = dq_desired_conjugate[4:8]

    H_real = ca.vertcat(
        ca.horzcat(real_conjugate[0], -real_conjugate[1], -real_conjugate[2], -real_conjugate[3]),
        ca.horzcat(real_conjugate[1], real_conjugate[0], -real_conjugate[3], real_conjugate[2]),
        ca.horzcat(real_conjugate[2], real_conjugate[3], real_conjugate[0], -real_conjugate[1]),
        ca.horzcat(real_conjugate[3], -real_conjugate[2], real_conjugate[1], real_conjugate[0]),
    )
    H_dual = ca.vertcat(
        ca.horzcat(dual_conjugate[0], -dual_conjugate[1], -dual_conjugate[2], -dual_conjugate[3]),
        ca.horzcat(dual_conjugate[1], dual_conjugate[0], -dual_conjugate[3], dual_conjugate[2]),
        ca.horzcat(dual_conjugate[2], dual_conjugate[3], dual_conjugate[0], -dual_conjugate[1]),
        ca.horzcat(dual_conjugate[3], -dual_conjugate[2], dual_conjugate[1], dual_conjugate[0]),
    )
    zeros = ca.DM.zeros(4, 4)
    H_dq = ca.vertcat(ca.horzcat(H_real, zeros), ca.horzcat(H_dual, H_real))
    dq_error = H_dq @ dq_current

    f = Function(
        "dq_error_from_mul_conj",
        [dq_desired, dq_current],
        [dq_error],
        ["dq_desired", "dq_current"],
        ["dq_error"],
    )
    f.description = "Dual quaternion multiplicative error: conj(dq_desired) * dq_current."
    return f


# ---------------------------------------------------------------------------
# Rotation factories (moved from dynamics.py)
# ---------------------------------------------------------------------------


def make_body_to_inertial_rotation() -> Function:
    quat = ca.MX.sym("quat", 4, 1)
    vec = ca.MX.sym("vec", 3, 1)
    vector = ca.vertcat(0.0, vec)
    quat_c = ca.vertcat(quat[0], -quat[1], -quat[2], -quat[3])

    H_p = ca.vertcat(
        ca.horzcat(quat[0], -quat[1], -quat[2], -quat[3]),
        ca.horzcat(quat[1], quat[0], -quat[3], quat[2]),
        ca.horzcat(quat[2], quat[3], quat[0], -quat[1]),
        ca.horzcat(quat[3], -quat[2], quat[1], quat[0]),
    )
    aux = H_p @ vector
    H_a = ca.vertcat(
        ca.horzcat(aux[0], -aux[1], -aux[2], -aux[3]),
        ca.horzcat(aux[1], aux[0], -aux[3], aux[2]),
        ca.horzcat(aux[2], aux[3], aux[0], -aux[1]),
        ca.horzcat(aux[3], -aux[2], aux[1], aux[0]),
    )
    vector_i = H_a @ quat_c
    f = Function("body_to_inertial_rotation", [quat, vec], [vector_i[1:4]])
    f.description = "Rotate a 3D vector from body to inertial frame using a quaternion."
    return f


def make_inertial_to_body_rotation() -> Function:
    quat = ca.MX.sym("quat", 4, 1)
    vec = ca.MX.sym("vec", 3, 1)
    vector = ca.vertcat(0.0, vec)
    quat_c = ca.vertcat(quat[0], -quat[1], -quat[2], -quat[3])

    H_p = ca.vertcat(
        ca.horzcat(quat_c[0], -quat_c[1], -quat_c[2], -quat_c[3]),
        ca.horzcat(quat_c[1], quat_c[0], -quat_c[3], quat_c[2]),
        ca.horzcat(quat_c[2], quat_c[3], quat_c[0], -quat_c[1]),
        ca.horzcat(quat_c[3], -quat_c[2], quat_c[1], quat_c[0]),
    )
    aux = H_p @ vector
    H_a = ca.vertcat(
        ca.horzcat(aux[0], -aux[1], -aux[2], -aux[3]),
        ca.horzcat(aux[1], aux[0], -aux[3], aux[2]),
        ca.horzcat(aux[2], aux[3], aux[0], -aux[1]),
        ca.horzcat(aux[3], -aux[2], aux[1], aux[0]),
    )
    vector_b = H_a @ quat
    f = Function("inertial_to_body_rotation", [quat, vec], [vector_b[1:4]])
    f.description = "Rotate a 3D vector from inertial to body frame using a quaternion."
    return f


# ---------------------------------------------------------------------------
# DQ kinematics / acceleration factories
# ---------------------------------------------------------------------------


def make_dualquat_kinematics() -> Function:
    """Build DQ kinematic derivative: q_dot = 0.5 * H(q) * twist + norm stabilization.

    K_quat = 10 is hardcoded as the quaternion-norm stabilization gain.

    Input ports:
        dualquat  (8,1) MX — dual quaternion [q_real, q_dual]
        twist     (6,1) MX — body-frame twist [wx, wy, wz, vx, vy, vz]

    Output:
        dq_dot  (8,1) MX — dual quaternion time derivative

    @return Compiled CasADi Function
    """
    dualquat = ca.MX.sym("dualquat", 8, 1)
    twist = ca.MX.sym("twist", 6, 1)

    quat_data = dualquat[0:4, 0]
    dual_data = dualquat[4:8, 0]
    K_quat = 10

    norm_r = ca.norm_2(quat_data)
    norm_d = 2 * ca.dot(quat_data, dual_data)
    quat_error = 1 - norm_r
    dual_error = norm_d
    aux_1 = quat_data * (K_quat * quat_error)
    aux_2 = dual_data * (0 * dual_error)
    aux_dual = ca.vertcat(aux_1, aux_2)

    H_r_plus = ca.vertcat(
        ca.horzcat(quat_data[0], -quat_data[1], -quat_data[2], -quat_data[3]),
        ca.horzcat(quat_data[1], quat_data[0], -quat_data[3], quat_data[2]),
        ca.horzcat(quat_data[2], quat_data[3], quat_data[0], -quat_data[1]),
        ca.horzcat(quat_data[3], -quat_data[2], quat_data[1], quat_data[0]),
    )
    H_d_plus = ca.vertcat(
        ca.horzcat(dual_data[0], -dual_data[1], -dual_data[2], -dual_data[3]),
        ca.horzcat(dual_data[1], dual_data[0], -dual_data[3], dual_data[2]),
        ca.horzcat(dual_data[2], dual_data[3], dual_data[0], -dual_data[1]),
        ca.horzcat(dual_data[3], -dual_data[2], dual_data[1], dual_data[0]),
    )
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros), ca.horzcat(H_d_plus, H_r_plus))

    omega = ca.vertcat(0.0, twist[0], twist[1], twist[2], 0.0, twist[3], twist[4], twist[5])
    q_dot = (1 / 2) * (Hplus @ omega) + aux_dual

    f = Function(
        "dualquat_kinematics",
        [dualquat, twist],
        [q_dot],
        ["dualquat", "twist"],
        ["dq_dot"],
    )
    f.description = "Dual quaternion kinematic derivative with norm stabilization (K_quat=10)."
    return f


def make_dualquat_acceleration(L: list[float]) -> Function:
    """Build rigid-body acceleration in body frame from force/torques.

    L = [mass, Ixx, Iyy, Izz, gravity] — numeric physics parameters.

    Input ports:
        dualquat  (8,1) MX — dual quaternion [q_real, q_dual]
        twist     (6,1) MX — body-frame twist [wx, wy, wz, vx, vy, vz]
        u         (4,1) MX — control [thrust, taux, tauy, tauz]

    Output:
        twist_dot  (6,1) MX — body-frame acceleration [wx_dot, wy_dot, wz_dot, vx_dot, vy_dot, vz_dot]

    @return Compiled CasADi Function
    """
    dualquat = ca.MX.sym("dualquat", 8, 1)
    twist = ca.MX.sym("twist", 6, 1)
    u = ca.MX.sym("u", 4, 1)

    force = u[0, 0]
    torques = u[1:4, 0]

    J = ca.DM.zeros(3, 3)
    J[0, 0] = L[1]
    J[1, 1] = L[2]
    J[2, 2] = L[3]
    J_1 = ca.inv(J)

    e3 = ca.DM.zeros(3, 1)
    e3[2, 0] = 1.0

    w = twist[0:3, 0]
    v = twist[3:6, 0]
    quat = dualquat[0:4, 0]

    f_rotation_inv = make_inertial_to_body_rotation()

    F_r = -J_1 @ ca.cross(w, J @ w)
    F_d = ca.cross(v, w) - L[4] * f_rotation_inv(quat, e3)
    U_r = J_1 @ torques
    U_d = (force / L[0]) * e3

    twist_dot = ca.vertcat(F_r + U_r, F_d + U_d)

    f = Function(
        "dualquat_acceleration",
        [dualquat, twist, u],
        [twist_dot],
        ["dualquat", "twist", "u"],
        ["twist_dot"],
    )
    f.description = "Body-frame rigid-body acceleration from DQ state and control inputs."
    return f


# ---------------------------------------------------------------------------
# NMPC cost term factories
# ---------------------------------------------------------------------------


def make_quat_error_cost() -> Function:
    q_desired = ca.MX.sym("q_desired", 4, 1)
    q_current = ca.MX.sym("q_current", 4, 1)
    q_desired_conjugate = ca.vertcat(q_desired[0], -q_desired[1], -q_desired[2], -q_desired[3])

    w, x, y, z = (
        q_desired_conjugate[0],
        q_desired_conjugate[1],
        q_desired_conjugate[2],
        q_desired_conjugate[3],
    )
    H = ca.vertcat(
        ca.horzcat(w, -x, -y, -z),
        ca.horzcat(x, w, -z, y),
        ca.horzcat(y, z, w, -x),
        ca.horzcat(z, -y, x, w),
    )
    q_err = H @ q_current
    q_err_w = q_err[0] + float(np.finfo(np.float64).eps)
    angle = 2 * ca.acos(q_err_w)
    log_q = angle * q_err[1:4] / ca.sqrt(1 - q_err_w * q_err_w)
    f = Function("quat_log_error_cost", [q_desired, q_current], [ca.norm_2(log_q)])
    f.description = "Quaternion geodesic error cost: ||log(q_d^* * q)||."
    return f


def make_translation_error_cost() -> Function:
    t_desired = ca.MX.sym("t_desired", 4, 1)
    t_current = ca.MX.sym("t_current", 4, 1)
    f = Function(
        "translation_error_cost",
        [t_desired, t_current],
        [(t_current - t_desired).T @ (t_current - t_desired)],
    )
    f.description = "Squared homogeneous translation tracking cost."
    return f


def make_pose_error_cost() -> Function:
    """Full SE(3) pose error cost via dual quaternion logarithmic map.

    Computes:  ||ln(conj(dq_desired) * dq_current)_{1:4, 5:8}||^2

    Inputs:
        dq_desired  (8,1) MX — desired dual quaternion
        dq_current  (8,1) MX — current dual quaternion

    Output:
        cost  scalar — 6D logarithmic pose error norm

    @return Compiled CasADi Function
    """
    dq_desired = ca.MX.sym("dq_desired", 8, 1)
    dq_current = ca.MX.sym("dq_current", 8, 1)

    dq_desired_conjugate = ca.vertcat(
        dq_desired[0],
        -dq_desired[1],
        -dq_desired[2],
        -dq_desired[3],
        dq_desired[4],
        -dq_desired[5],
        -dq_desired[6],
        -dq_desired[7],
    )
    real_conjugate = dq_desired_conjugate[0:4]
    dual_conjugate = dq_desired_conjugate[4:8]

    H_r = ca.vertcat(
        ca.horzcat(real_conjugate[0], -real_conjugate[1], -real_conjugate[2], -real_conjugate[3]),
        ca.horzcat(real_conjugate[1], real_conjugate[0], -real_conjugate[3], real_conjugate[2]),
        ca.horzcat(real_conjugate[2], real_conjugate[3], real_conjugate[0], -real_conjugate[1]),
        ca.horzcat(real_conjugate[3], -real_conjugate[2], real_conjugate[1], real_conjugate[0]),
    )
    H_d = ca.vertcat(
        ca.horzcat(dual_conjugate[0], -dual_conjugate[1], -dual_conjugate[2], -dual_conjugate[3]),
        ca.horzcat(dual_conjugate[1], dual_conjugate[0], -dual_conjugate[3], dual_conjugate[2]),
        ca.horzcat(dual_conjugate[2], dual_conjugate[3], dual_conjugate[0], -dual_conjugate[1]),
        ca.horzcat(dual_conjugate[3], -dual_conjugate[2], dual_conjugate[1], dual_conjugate[0]),
    )
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r, zeros), ca.horzcat(H_d, H_r))
    dq_error = Hplus @ dq_current

    q_err_real = dq_error[0:4]
    q_err_real_c = ca.vertcat(
        q_err_real[0],
        -q_err_real[1],
        -q_err_real[2],
        -q_err_real[3],
    )
    q_err_dual = dq_error[4:8]

    norm_v = ca.sqrt(
        q_err_real[1] * q_err_real[1]
        + q_err_real[2] * q_err_real[2]
        + q_err_real[3] * q_err_real[3]
        + _EPS
    )
    angle = 2 * ca.atan2(norm_v, q_err_real[0])
    log_r = angle * q_err_real[1:4] / norm_v

    H_dual = ca.vertcat(
        ca.horzcat(q_err_dual[0], -q_err_dual[1], -q_err_dual[2], -q_err_dual[3]),
        ca.horzcat(q_err_dual[1], q_err_dual[0], -q_err_dual[3], q_err_dual[2]),
        ca.horzcat(q_err_dual[2], q_err_dual[3], q_err_dual[0], -q_err_dual[1]),
        ca.horzcat(q_err_dual[3], -q_err_dual[2], q_err_dual[1], q_err_dual[0]),
    )
    trans_err = 2 * H_dual @ q_err_real_c

    ln_error = ca.vertcat(log_r, trans_err[1:4])
    cost = ln_error.T @ ln_error

    f = Function(
        "pose_error_cost",
        [dq_desired, dq_current],
        [cost],
        ["dq_desired", "dq_current"],
        ["cost"],
    )
    f.description = "SE(3) pose error via DQ logarithmic map: ||ln(conj(dq_d)*dq)[1:4,5:8]||^2."
    return f


def make_control_error_cost() -> Function:
    """Squared weighted control deviation: (u_nom - u)^T @ R @ (u_nom - u).

    Inputs:
        u_nom  (4,1) MX — nominal feedforward control [thrust, τx, τy, τz]
        u      (4,1) MX — actual control
        R      (4,4) MX — diagonal weighting matrix

    Output:
        cost  scalar

    @return Compiled CasADi Function
    """
    u_nom = ca.MX.sym("u_nom", 4, 1)
    u = ca.MX.sym("u", 4, 1)
    R = ca.MX.sym("R", 4, 4)
    err = u_nom - u
    cost = err.T @ R @ err
    f = Function("control_error_cost", [u_nom, u, R], [cost], ["u_nom", "u", "R"], ["cost"])
    f.description = "Weighted control deviation cost: (u_nom-u)^T @ R @ (u_nom-u)."
    return f


def make_angular_velocity_error_cost() -> Function:
    """Squared body-frame angular velocity error: ||w_b - w_b_d||^2.

    Inputs:
        w_b_d  (3,1) MX — desired body angular velocity [rad/s]
        w_b    (3,1) MX — current body angular velocity [rad/s]

    Output:
        cost  scalar

    @return Compiled CasADi Function
    """
    w_b_d = ca.MX.sym("w_b_d", 3, 1)
    w_b = ca.MX.sym("w_b", 3, 1)
    err = w_b - w_b_d
    cost = err.T @ err
    f = Function("angular_velocity_error_cost", [w_b_d, w_b], [cost], ["w_b_d", "w_b"], ["cost"])
    f.description = "Squared body angular velocity error: ||w_b - w_b_d||^2."
    return f


def make_inertial_velocity_error_cost() -> Function:
    """Squared inertial velocity error: ||v_i - v_i_d||^2.

    Inputs:
        v_i_d  (3,1) MX — desired world inertial velocity [m/s]
        v_i    (3,1) MX — current world inertial velocity [m/s]

    Output:
        cost  scalar

    @return Compiled CasADi Function
    """
    v_i_d = ca.MX.sym("v_i_d", 3, 1)
    v_i = ca.MX.sym("v_i", 3, 1)
    err = v_i - v_i_d
    cost = err.T @ err
    f = Function("inertial_velocity_error_cost", [v_i_d, v_i], [cost], ["v_i_d", "v_i"], ["cost"])
    f.description = "Squared inertial velocity error: ||v_i - v_i_d||^2."
    return f


def dualquat_from_pose_np(quat: np.ndarray, trans3: np.ndarray) -> np.ndarray:
    """Build DQ (8,) from unit quaternion (4,) and world translation (3,).

    Dual part = 0.5 * t ⊗ q, matching DualQuaternion.from_pose.

    @param[in] quat   (4,) or (4,1) wxyz quaternion
    @param[in] trans3 (3,) world ENU position [m]
    @return           (8,) dual quaternion [q_real, q_dual]
    """
    qw, qx, qy, qz = quat.ravel()
    tx, ty, tz = trans3.ravel()
    dual = 0.5 * np.array(
        [
            -(qx * tx + qy * ty + qz * tz),
            qw * tx + qy * tz - qz * ty,
            qw * ty - qx * tz + qz * tx,
            qw * tz + qx * ty - qy * tx,
        ]
    )
    return np.concatenate([quat.ravel(), dual])
