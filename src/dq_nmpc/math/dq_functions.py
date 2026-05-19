"""CasADi Function factories for dual-quaternion operations.

Compiled CasADi Functions wrapping symbolic algebra from dq_algebra
and DualQuaternion.from_pose.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
from casadi import Function

from dq_nmpc.math.dual_quaternion import DualQuaternion

_qw_aux = ca.MX.sym("qw_1_aux", 1, 1)
_qx_aux = ca.MX.sym("qx_1_aux", 1, 1)
_qy_aux = ca.MX.sym("qy_1_aux", 1, 1)
_qz_aux = ca.MX.sym("qz_1_aux", 1, 1)
_q_aux = ca.vertcat(_qw_aux, _qx_aux, _qy_aux, _qz_aux)
_tx_aux = ca.MX.sym("tx_1_aux", 1, 1)
_ty_aux = ca.MX.sym("ty_1_aux", 1, 1)
_tz_aux = ca.MX.sym("tz_1_aux", 1, 1)
_t_aux = ca.vertcat(0.0, _tx_aux, _ty_aux, _tz_aux)

_Q1_pose = DualQuaternion.from_pose(quat=_q_aux, trans=_t_aux)


def dualquat_from_pose_casadi():
    values = _Q1_pose.get[:, 0]
    return Function(
        "f_pose", [_qw_aux, _qx_aux, _qy_aux, _qz_aux, _tx_aux, _ty_aux, _tz_aux], [values]
    )


def make_dualquat_mul_conj():
    qd = ca.MX.sym("qd", 8, 1)
    q = ca.MX.sym("q", 8, 1)
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

    q_error = Hplus @ q
    return Function("f_error_dual", [qd, q], [q_error])


def make_quat_error_cost():
    qd = ca.MX.sym("qd", 4, 1)
    q = ca.MX.sym("q", 4, 1)
    qd_conjugate = ca.vertcat(qd[0, 0], -qd[1, 0], -qd[2, 0], -qd[3, 0])
    quat_d_data = qd_conjugate[0:4, 0]

    H_r_plus = ca.vertcat(
        ca.horzcat(quat_d_data[0, 0], -quat_d_data[1, 0], -quat_d_data[2, 0], -quat_d_data[3, 0]),
        ca.horzcat(quat_d_data[1, 0], quat_d_data[0, 0], -quat_d_data[3, 0], quat_d_data[2, 0]),
        ca.horzcat(quat_d_data[2, 0], quat_d_data[3, 0], quat_d_data[0, 0], -quat_d_data[1, 0]),
        ca.horzcat(quat_d_data[3, 0], -quat_d_data[2, 0], quat_d_data[1, 0], quat_d_data[0, 0]),
    )

    q_error = H_r_plus @ q[0:4, 0]

    qw = q_error[0, 0] + ca.np.finfo(np.float64).eps
    angle = 2 * ca.acos(qw)
    denominator = ca.sqrt(1 - qw * qw)

    ln_quaternion = ca.vertcat(
        angle * q_error[1, 0] / denominator,
        angle * q_error[2, 0] / denominator,
        angle * q_error[3, 0] / denominator,
    )

    cost = ca.norm_2(ln_quaternion)
    return Function("f_cost", [qd, q], [cost])


def make_translation_error_cost():
    td = ca.MX.sym("td", 4, 1)
    t = ca.MX.sym("t", 4, 1)

    te = td - t
    cost = te.T @ te
    return Function("f_cost", [td, t], [cost])
