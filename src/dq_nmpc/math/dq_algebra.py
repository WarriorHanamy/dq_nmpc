"""Pure casadi dual-quaternion algebra operations used by NMPC OCP formulation.

No acados dependency. Operates on raw casadi MX arrays (8x1 or 4x1).
"""

from __future__ import annotations

import casadi as ca
import numpy as np


def adjoint_map(qd, v):
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    quat_d_data = qd_conjugate[0:4]
    dual_d_data = qd_conjugate[4:8]

    wb = v[0:3, 0]
    vb = v[3:6, 0]

    vector = ca.vertcat(0.0, wb, 0.0, vb)

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

    aux_dual = Hplus @ vector
    quat_aux_data = aux_dual[0:4]
    dual_aux_data = aux_dual[4:8]

    H_r_aux_plus = ca.vertcat(
        ca.horzcat(quat_aux_data[0], -quat_aux_data[1], -quat_aux_data[2], -quat_aux_data[3]),
        ca.horzcat(quat_aux_data[1], quat_aux_data[0], -quat_aux_data[3], quat_aux_data[2]),
        ca.horzcat(quat_aux_data[2], quat_aux_data[3], quat_aux_data[0], -quat_aux_data[1]),
        ca.horzcat(quat_aux_data[3], -quat_aux_data[2], quat_aux_data[1], quat_aux_data[0]),
    )

    H_d_aux_plus = ca.vertcat(
        ca.horzcat(dual_aux_data[0], -dual_aux_data[1], -dual_aux_data[2], -dual_aux_data[3]),
        ca.horzcat(dual_aux_data[1], dual_aux_data[0], -dual_aux_data[3], dual_aux_data[2]),
        ca.horzcat(dual_aux_data[2], dual_aux_data[3], dual_aux_data[0], -dual_aux_data[1]),
        ca.horzcat(dual_aux_data[3], -dual_aux_data[2], dual_aux_data[1], dual_aux_data[0]),
    )

    Haux_plus = ca.vertcat(ca.horzcat(H_r_aux_plus, zeros), ca.horzcat(H_d_aux_plus, H_r_aux_plus))

    vector_b_dual = Haux_plus @ qd
    vector_b = ca.vertcat(vector_b_dual[1:4], vector_b_dual[5:8])
    return vector_b


def log_error_dualquat(qd, q):
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

    q_e_aux = Hplus @ q

    q_error = q_e_aux

    q_error_real = q_error[0:4, 0]
    q_error_real_c = ca.vertcat(
        q_error_real[0, 0], -q_error_real[1, 0], -q_error_real[2, 0], -q_error_real[3, 0]
    )
    q_error_dual = q_error[4:8, 0]

    ## Real Part
    norm = ca.norm_2(q_error_real[1:4] + ca.np.finfo(np.float64).eps)
    angle = ca.atan2(norm, q_error_real[0])

    ## Dual Part
    H_error_dual_plus = ca.vertcat(
        ca.horzcat(
            q_error_dual[0, 0], -q_error_dual[1, 0], -q_error_dual[2, 0], -q_error_dual[3, 0]
        ),
        ca.horzcat(q_error_dual[1, 0], q_error_dual[0, 0], -q_error_dual[3, 0], q_error_dual[2, 0]),
        ca.horzcat(q_error_dual[2, 0], q_error_dual[3, 0], q_error_dual[0, 0], -q_error_dual[1, 0]),
        ca.horzcat(q_error_dual[3, 0], -q_error_dual[2, 0], q_error_dual[1, 0], q_error_dual[0, 0]),
    )

    trans_error = 2 * H_error_dual_plus @ q_error_real_c
    # Computing log map
    ln_quaternion = ca.vertcat(
        0.0,
        (1 / 2) * angle * q_error_real[1, 0] / norm,
        (1 / 2) * angle * q_error_real[2, 0] / norm,
        (1 / 2) * angle * q_error_real[3, 0] / norm,
    )
    ln_trans = ca.vertcat(
        0.0, (1 / 2) * trans_error[1, 0], (1 / 2) * trans_error[2, 0], (1 / 2) * trans_error[3, 0]
    )

    q_e_ln = ca.vertcat(ln_quaternion, ln_trans)
    return q_e_ln


def log_map_dualquat(q_error):
    q_error_real = q_error[0:4, 0]
    q_error_real_c = ca.vertcat(
        q_error_real[0, 0], -q_error_real[1, 0], -q_error_real[2, 0], -q_error_real[3, 0]
    )
    q_error_dual = q_error[4:8, 0]

    ## Real Part
    norm = ca.norm_2(q_error_real[1:4] + ca.np.finfo(np.float64).eps)
    angle = 2 * ca.atan2(norm, q_error_real[0])

    ## Dual Part
    H_error_dual_plus = ca.vertcat(
        ca.horzcat(
            q_error_dual[0, 0], -q_error_dual[1, 0], -q_error_dual[2, 0], -q_error_dual[3, 0]
        ),
        ca.horzcat(q_error_dual[1, 0], q_error_dual[0, 0], -q_error_dual[3, 0], q_error_dual[2, 0]),
        ca.horzcat(q_error_dual[2, 0], q_error_dual[3, 0], q_error_dual[0, 0], -q_error_dual[1, 0]),
        ca.horzcat(q_error_dual[3, 0], -q_error_dual[2, 0], q_error_dual[1, 0], q_error_dual[0, 0]),
    )

    trans_error = 2 * H_error_dual_plus @ q_error_real_c
    # Computing log map
    ln_quaternion = ca.vertcat(
        0.0,
        (1 / 2) * angle * q_error_real[1, 0] / norm,
        (1 / 2) * angle * q_error_real[2, 0] / norm,
        (1 / 2) * angle * q_error_real[3, 0] / norm,
    )
    ln_trans = ca.vertcat(
        0.0, (1 / 2) * trans_error[1, 0], (1 / 2) * trans_error[2, 0], (1 / 2) * trans_error[3, 0]
    )

    q_e_ln = ca.vertcat(ln_quaternion, ln_trans)

    return q_e_ln


def dualquat_conjugate(qd):
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    return qd_conjugate


def dualquat_mul_conj(qd, q):
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

    q_e_aux = Hplus @ q

    q_error = q_e_aux

    return q_error
