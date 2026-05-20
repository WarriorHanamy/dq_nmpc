"""CasADi Function factories for dual-quaternion operations.

Compiled CasADi Functions wrapping symbolic algebra from dq_algebra
and DualQuaternion.from_pose.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
from casadi import Function

from dq_nmpc.math.dual_quaternion import DualQuaternion

__all__ = [
    "make_dualquat_from_pose",
    "make_dualquat_mul_conj",
    "make_quat_error_cost",
    "make_translation_error_cost",
]


def make_dualquat_from_pose():
    """
    Create a CasADi Function that builds a dual quaternion from a pose.

    The dual quaternion represents SE(3) pose: real part is the
    orientation quaternion, dual part encodes translation via
    q_dual = 0.5 * t * q_real.

    Inputs:
        qw: quaternion scalar part (w), shape (1, 1)
        qx: quaternion x component, shape (1, 1)
        qy: quaternion y component, shape (1, 1)
        qz: quaternion z component, shape (1, 1)
        tx: translation x [m], shape (1, 1)
        ty: translation y [m], shape (1, 1)
        tz: translation z [m], shape (1, 1)

    Output:
        dq: dual quaternion vector [real; dual], shape (8, 1)
    """
    qw = ca.MX.sym("qw", 1, 1)
    qx = ca.MX.sym("qx", 1, 1)
    qy = ca.MX.sym("qy", 1, 1)
    qz = ca.MX.sym("qz", 1, 1)
    quaternion = ca.vertcat(qw, qx, qy, qz)

    tx = ca.MX.sym("tx", 1, 1)
    ty = ca.MX.sym("ty", 1, 1)
    tz = ca.MX.sym("tz", 1, 1)
    translation = ca.vertcat(0.0, tx, ty, tz)

    dq = DualQuaternion.from_pose(quat=quaternion, trans=translation)
    dq_full = ca.vertcat(dq.Qr.get, dq.Qd.get)

    f = Function(
        "dualquat_from_pose",
        [qw, qx, qy, qz, tx, ty, tz],
        [dq_full],
        ["qw", "qx", "qy", "qz", "tx", "ty", "tz"],
        ["dq"],
    )

    f.description = (
        "Build a dual quaternion (8,) from quaternion (w,x,y,z) and translation (x,y,z) components."
    )

    return f


def make_dualquat_mul_conj():
    """
    Create a CasADi Function that computes the dual quaternion error
    via conjugate multiplication.

    Constructs the 8x8 plus-Hamiltonian product matrix from the
    conjugate of dq_desired, then multiplies by dq_current:
    dq_error = H_plus(conjugate(dq_desired)) @ dq_current

    Inputs:
        dq_desired: desired dual quaternion, shape (8, 1)
        dq_current: current dual quaternion, shape (8, 1)

    Output:
        dq_error: dual quaternion multiplicative error, shape (8, 1)
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
    H_dq = ca.vertcat(
        ca.horzcat(H_real, zeros),
        ca.horzcat(H_dual, H_real),
    )

    dq_error = H_dq @ dq_current

    f = Function(
        "dq_error_from_mul_conj",
        [dq_desired, dq_current],
        [dq_error],
        ["dq_desired", "dq_current"],
        ["dq_error"],
    )

    f.description = (
        "Dual quaternion multiplicative error: dq_error = conjugate(dq_desired) * dq_current."
    )

    return f


def make_quat_error_cost():
    """
    Create a CasADi Function for quaternion orientation tracking cost.

    Computes the geodesic distance on S^3 via the logarithmic map:
    q_error = conjugate(q_desired) * q_current
    cost = ||log(q_error)||_2

    Inputs:
        q_desired: desired quaternion [w, x, y, z], shape (4, 1)
        q_current: current quaternion [w, x, y, z], shape (4, 1)

    Output:
        cost: geodesic distance on S^3, scalar
    """
    q_desired = ca.MX.sym("q_desired", 4, 1)
    q_current = ca.MX.sym("q_current", 4, 1)

    q_desired_conjugate = ca.vertcat(q_desired[0], -q_desired[1], -q_desired[2], -q_desired[3])

    w, x, y, z = (
        q_desired_conjugate[0],
        q_desired_conjugate[1],
        q_desired_conjugate[2],
        q_desired_conjugate[3],
    )
    H_real = ca.vertcat(
        ca.horzcat(w, -x, -y, -z),
        ca.horzcat(x, w, -z, y),
        ca.horzcat(y, z, w, -x),
        ca.horzcat(z, -y, x, w),
    )

    q_error_quat = H_real @ q_current

    q_error_w = q_error_quat[0] + ca.np.finfo(np.float64).eps
    angle = 2 * ca.acos(q_error_w)
    denominator = ca.sqrt(1 - q_error_w * q_error_w)
    log_quaternion = angle * q_error_quat[1:4] / denominator

    cost = ca.norm_2(log_quaternion)

    f = Function(
        "quat_log_error_cost",
        [q_desired, q_current],
        [cost],
        ["q_desired", "q_current"],
        ["cost"],
    )

    f.description = (
        "Quaternion geodesic error cost via logarithmic map: "
        "cost = ||log(q_desired^* * q_current)||."
    )

    return f


def make_translation_error_cost():
    """
    Create a CasADi Function for homogeneous 4D translation tracking cost.

    Inputs:
        translation_desired: desired homogeneous translation, shape (4, 1)
        translation_current: current homogeneous translation, shape (4, 1)

    Output:
        cost: squared translation error,
              (t_current - t_desired)^T (t_current - t_desired)
    """
    translation_desired = ca.MX.sym("translation_desired", 4, 1)
    translation_current = ca.MX.sym("translation_current", 4, 1)

    translation_error = translation_current - translation_desired
    cost = translation_error.T @ translation_error

    f = Function(
        "translation_error_cost",
        [translation_desired, translation_current],
        [cost],
        ["translation_desired", "translation_current"],
        ["cost"],
    )

    f.description = (
        "Squared 4D homogeneous translation tracking cost: "
        "(t_current - t_desired)^T (t_current - t_desired)."
    )

    return f
