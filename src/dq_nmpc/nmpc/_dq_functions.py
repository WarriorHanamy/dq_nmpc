"""CasADi dual-quaternion algebra — expression functions, ca.Function wrappers, and numpy helpers.

Expression-first design: ``_expr`` functions are the mathematical source of truth.
``_ca_func`` factories create compiled ``ca.Function`` wrappers that call ``_expr``.
Numpy helpers provide runtime evaluation without CasADi overhead.
"""

from __future__ import annotations

from typing import Literal

import casadi as ca
import numpy as np

from dq_nmpc.type import CasadiVec

__all__ = [
    "dualquat_from_pose_expr",
    "dualquat_from_pose_ca_func",
    "dualquat_kinematics_expr",
    "dualquat_kinematics_ca_func",
    "dualquat_acceleration_expr",
    "dualquat_acceleration_ca_func",
    "inertial_to_body_rotation_expr",
    "inertial_to_body_rotation_ca_func",
    "dualquat_mul_conj_expr",
    "log_map_dualquat_expr",
    "dualquat_quat_part_expr",
    "dualquat_quat_part_ca_func",
]

# ---------------------------------------------------------------------------
# Expression functions — mathematical source of truth
# ---------------------------------------------------------------------------


def dualquat_mul_conj_expr(qd: CasadiVec, q: CasadiVec) -> CasadiVec:
    """DQ multiplicative error: conj(qd) * q via 8x8 Hamiltonian matrix.

    @param[in] qd  (8,) desired dual quaternion [unit DQ]
    @param[in] q   (8,) current dual quaternion [unit DQ]
    @return        (8,) DQ error [unit DQ]
    """
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    quat_d_data = qd_conjugate[0:4]
    dual_d_data = qd_conjugate[4:8]

    H_r_plus = ca.vertcat(
        ca.horzcat(quat_d_data[0], -quat_d_data[1], -quat_d_data[2], -quat_d_data[3]),
        ca.horzcat(quat_d_data[1], quat_d_data[0], -quat_d_data[3], quat_d_data[2]),
        ca.horzcat(quat_d_data[2], quat_d_data[3], quat_d_data[0], -quat_d_data[1]),
        ca.horzcat(quat_d_data[3], -quat_d_data[2], quat_d_data[1], quat_d_data[0]),
    )
    H_d_plus = ca.vertcat(
        ca.horzcat(dual_d_data[0], -dual_d_data[1], -dual_d_data[2], -dual_d_data[3]),
        ca.horzcat(dual_d_data[1], dual_d_data[0], -dual_d_data[3], dual_d_data[2]),
        ca.horzcat(dual_d_data[2], dual_d_data[3], dual_d_data[0], -dual_d_data[1]),
        ca.horzcat(dual_d_data[3], -dual_d_data[2], dual_d_data[1], dual_d_data[0]),
    )
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros), ca.horzcat(H_d_plus, H_r_plus))
    return Hplus @ q


def log_map_dualquat_expr(q_error: CasadiVec) -> CasadiVec:
    """Logarithmic map of a dual-quaternion error onto SE(3) Lie algebra.

    Uses atan2-based formulation (numeric stable near identity).

    @param[in] q_error  (8,) DQ error from ``dualquat_mul_conj_expr``
    @return             (8,) ln(error) = [ln_rot, ln_trans]
    """
    q_error_real = q_error[0:4, 0]
    q_error_real_c = ca.vertcat(
        q_error_real[0, 0],
        -q_error_real[1, 0],
        -q_error_real[2, 0],
        -q_error_real[3, 0],
    )
    q_error_dual = q_error[4:8, 0]

    norm = ca.norm_2(q_error_real[1:4] + ca.np.finfo(np.float64).eps)
    angle = 2 * ca.atan2(norm, q_error_real[0])

    H_plus_dual = ca.vertcat(
        ca.horzcat(
            q_error_dual[0, 0],
            -q_error_dual[1, 0],
            -q_error_dual[2, 0],
            -q_error_dual[3, 0],
        ),
        ca.horzcat(
            q_error_dual[1, 0],
            q_error_dual[0, 0],
            -q_error_dual[3, 0],
            q_error_dual[2, 0],
        ),
        ca.horzcat(
            q_error_dual[2, 0],
            q_error_dual[3, 0],
            q_error_dual[0, 0],
            -q_error_dual[1, 0],
        ),
        ca.horzcat(
            q_error_dual[3, 0],
            -q_error_dual[2, 0],
            q_error_dual[1, 0],
            q_error_dual[0, 0],
        ),
    )
    trans_error = 2 * H_plus_dual @ q_error_real_c

    ln_quat = ca.vertcat(
        0.0,
        (1 / 2) * angle * q_error_real[1, 0] / norm,
        (1 / 2) * angle * q_error_real[2, 0] / norm,
        (1 / 2) * angle * q_error_real[3, 0] / norm,
    )
    ln_trans = ca.vertcat(
        0.0,
        (1 / 2) * trans_error[1, 0],
        (1 / 2) * trans_error[2, 0],
        (1 / 2) * trans_error[3, 0],
    )
    return ca.vertcat(ln_quat, ln_trans)


def dualquat_from_pose_expr(quat: CasadiVec, trans: CasadiVec) -> CasadiVec:
    """Build dual quaternion (8,) from unit quaternion and world translation.

    Dual part = 0.5 * t ⊗ q.

    @param[in] quat  (4,) unit quaternion [wxyz]
    @param[in] trans (3,) world ENU translation [m]
    @return          (8,) dual quaternion [q_real, q_dual]
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    tx, ty, tz = trans[0], trans[1], trans[2]

    q_dual = 0.5 * ca.vertcat(
        -(qx * tx + qy * ty + qz * tz),
        qw * tx + qy * tz - qz * ty,
        qw * ty - qx * tz + qz * tx,
        qw * tz + qx * ty - qy * tx,
    )
    return ca.vertcat(qw, qx, qy, qz, q_dual)


def dualquat_kinematics_expr(dualquat: CasadiVec, twist: CasadiVec) -> CasadiVec:
    """DQ kinematic derivative with quaternion-norm stabilization.

    q_dot = 0.5 * H(q) * omega + K_quat * (1 - ||q||) * q

    @param[in] dualquat  (8,) dual quaternion [unit DQ]
    @param[in] twist     (6,) body-frame twist [wx, wy, wz, vx, vy, vz]
    @return              (8,) dual quaternion time derivative
    """
    quat_data = dualquat[0:4]
    dual_data = dualquat[4:8]
    K_quat = 10

    norm_r = ca.norm_2(quat_data)
    norm_d = 2 * ca.dot(quat_data, dual_data)
    quat_error = 1 - norm_r
    dual_error = norm_d
    quat_stab_comp = quat_data * (K_quat * quat_error)
    dual_stab_comp = dual_data * (0 * dual_error)

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
    q_dot = (1 / 2) * (Hplus @ omega) + ca.vertcat(quat_stab_comp, dual_stab_comp)
    return q_dot


def inertial_to_body_rotation_expr(quat: CasadiVec, vec: CasadiVec) -> CasadiVec:
    """Rotate a 3D vector from inertial to body frame via quaternion.

    Applies v' = conj(q) * (0, vec) * q, keeping only the vector part.

    @param[in] quat  (4,) orientation quaternion [wxyz]
    @param[in] vec   (3,) vector in inertial frame
    @return          (3,) vector in body frame
    """
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
    return vector_b[1:4]


def dualquat_acceleration_expr(
    dualquat: CasadiVec,
    twist: CasadiVec,
    u: CasadiVec,
    L: list[float],
) -> CasadiVec:
    """Body-frame rigid-body acceleration from DQ state and control.

    twist_dot = [J⁻¹·(τ - ω × Jω),  (F/m)·e₃ + v × ω - g·Rᵀe₃]

    @param[in] dualquat  (8,) dual quaternion [unit DQ]
    @param[in] twist     (6,) body-frame twist [wx, wy, wz, vx, vy, vz]
    @param[in] u         (4,) control [thrust, taux, tauy, tauz]
    @param[in] L         [mass, Ixx, Iyy, Izz, gravity]
    @return              (6,) body-frame acceleration twist_dot
    """
    force = u[0]
    torques = u[1:4]

    J = ca.DM.zeros(3, 3)
    J[0, 0] = L[1]
    J[1, 1] = L[2]
    J[2, 2] = L[3]
    J_1 = ca.inv(J)
    e3 = ca.DM.zeros(3, 1)
    e3[2, 0] = 1.0

    w = twist[0:3]
    v = twist[3:6]
    quat = dualquat[0:4]

    F_r = -J_1 @ ca.cross(w, J @ w)
    rotated_e3 = inertial_to_body_rotation_expr(quat, e3)
    F_d = ca.cross(v, w) - L[4] * rotated_e3
    U_r = J_1 @ torques
    U_d = (force / L[0]) * e3

    return ca.vertcat(F_r + U_r, F_d + U_d)


def dualquat_quat_part_expr(dualquat: CasadiVec) -> CasadiVec:
    """Extract the quaternion (real) part from a dual quaternion.

    @param[in] dualquat  (8,) dual quaternion [unit DQ]
    @return              (4,) quaternion part [wxyz]
    """
    return dualquat[0:4]


# ---------------------------------------------------------------------------
# ca.Function wrappers
# ---------------------------------------------------------------------------


def dualquat_from_pose_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: (qw,qx,qy,qz,tx,ty,tz) -> dq(8,1).

    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    qw = sym("qw", 1, 1)
    qx = sym("qx", 1, 1)
    qy = sym("qy", 1, 1)
    qz = sym("qz", 1, 1)
    tx = sym("tx", 1, 1)
    ty = sym("ty", 1, 1)
    tz = sym("tz", 1, 1)

    quat = ca.vertcat(qw, qx, qy, qz)
    trans = ca.vertcat(tx, ty, tz)
    dq_full = dualquat_from_pose_expr(quat, trans)

    f = ca.Function(
        "dualquat_from_pose",
        [qw, qx, qy, qz, tx, ty, tz],
        [dq_full],
        ["qw", "qx", "qy", "qz", "tx", "ty", "tz"],
        ["dq"],
    )
    f.description = "Build a dual quaternion (8,) from quaternion and translation."
    return f


def position_from_dualquat_expr(dualquat: CasadiVec) -> CasadiVec:
    """Extract world ENU position from unit dual quaternion.

    Dual part d = 0.5 * t ⊗ q  ⇒  t = 2 * d * q^*
    t_i = 2 * (qw * d_i - dw * q_i - (d_vec × q_vec)_i)

    @param[in] dualquat  (8,) unit dual quaternion
    @return              (3,) position [m]
    """
    qw, qx, qy, qz = dualquat[0], dualquat[1], dualquat[2], dualquat[3]
    dw, dx, dy, dz = dualquat[4], dualquat[5], dualquat[6], dualquat[7]
    cross_x = dy * qz - dz * qy
    cross_y = dz * qx - dx * qz
    cross_z = dx * qy - dy * qx
    px = 2.0 * (qw * dx - dw * qx - cross_x)
    py = 2.0 * (qw * dy - dw * qy - cross_y)
    pz = 2.0 * (qw * dz - dw * qz - cross_z)
    return ca.vertcat(px, py, pz)


def position_from_dualquat_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: dualquat(8,1) -> position(3,1).

    Inverse of dualquat_from_pose_ca_func.

    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    dualquat = sym("dualquat", 8, 1)
    pos = position_from_dualquat_expr(dualquat)
    f = ca.Function(
        "position_from_dualquat",
        [dualquat],
        [pos],
    )
    f.description = "Extract world-frame position (3,) from a dual quaternion (8,)."
    return f


def yaw_from_dualquat_expr(dualquat: CasadiVec) -> CasadiVec:
    """Extract yaw angle from unit dual quaternion quaternion part.

    q[0:4] → rotation matrix R → atan2(R[1,0], R[0,0])

    @param[in] dualquat  (8,) unit dual quaternion
    @return              (1,) yaw angle [rad]
    """
    qw, qx, qy, qz = dualquat[0], dualquat[1], dualquat[2], dualquat[3]
    R00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    R10 = 2.0 * (qx * qy + qz * qw)
    return ca.arctan2(R10, R00)


def yaw_from_dualquat_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: dualquat(8,1) -> yaw(1,1).

    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    dualquat = sym("dualquat", 8, 1)
    yaw = yaw_from_dualquat_expr(dualquat)
    f = ca.Function(
        "yaw_from_dualquat",
        [dualquat],
        [yaw],
    )
    f.description = "Extract yaw angle [rad] from a dual quaternion (8,)."
    return f


def dualquat_kinematics_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: (dualquat(8,1), twist(6,1)) -> dq_dot(8,1).

    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    dualquat = sym("dualquat", 8, 1)
    twist = sym("twist", 6, 1)

    q_dot = dualquat_kinematics_expr(dualquat, twist)

    f = ca.Function(
        "dualquat_kinematics",
        [dualquat, twist],
        [q_dot],
        ["dualquat", "twist"],
        ["dq_dot"],
    )
    f.description = "DQ kinematic derivative with norm stabilization (K_quat=10)."
    return f


def dualquat_acceleration_ca_func(
    L: list[float],
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: (dualquat(8,1), twist(6,1), u(4,1)) -> twist_dot(6,1).

    @param[in] L  [mass, Ixx, Iyy, Izz, gravity]
    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    dualquat = sym("dualquat", 8, 1)
    twist = sym("twist", 6, 1)
    u = sym("u", 4, 1)

    twist_dot = dualquat_acceleration_expr(dualquat, twist, u, L)

    f = ca.Function(
        "dualquat_acceleration",
        [dualquat, twist, u],
        [twist_dot],
        ["dualquat", "twist", "u"],
        ["twist_dot"],
    )
    f.description = "Body-frame rigid-body acceleration from DQ state and control inputs."
    return f


def inertial_to_body_rotation_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: (quat(4,1), vec(3,1)) -> vec_body(3,1).

    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    quat = sym("quat", 4, 1)
    vec = sym("vec", 3, 1)

    vector_b = inertial_to_body_rotation_expr(quat, vec)

    f = ca.Function(
        "inertial_to_body_rotation",
        [quat, vec],
        [vector_b],
    )
    f.description = "Rotate a 3D vector from inertial to body frame using a quaternion."
    return f


def dualquat_quat_part_ca_func(
    symbolic_type: Literal["MX", "SX"] = "MX",
) -> ca.Function:
    """Build compiled ca.Function: (dualquat(8,1)) -> quat(4,1).

    @return Compiled ca.Function
    """
    sym = ca.MX.sym if symbolic_type == "MX" else ca.SX.sym
    dualquat = sym("dualquat", 8, 1)

    quat_part = dualquat_quat_part_expr(dualquat)

    f = ca.Function(
        "dualquat_quat_part",
        [dualquat],
        [quat_part],
    )
    f.description = "Extract the quaternion (real) part from a dual quaternion."
    return f
