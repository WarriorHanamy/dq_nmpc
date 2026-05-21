"""Quadrotor dynamics in dual-quaternion representation for acados NMPC.

Public API
  export_acados_model              build AcadosModel from NMPCConfig
  make_quadrotor_model             build AcadosModel from system parameter list
  make_body_velocity_from_twist    factory: twist -> body-frame [w; v] mapping
  make_inertial_velocity_from_twist  factory: twist -> inertial-frame [w; v] mapping
  make_get_quaternion              factory: extract quaternion from dual quaternion
  make_get_translation             factory: extract translation from dual quaternion
  apply_noise                      apply position/quaternion/twist noise to state
"""

from __future__ import annotations

from types import SimpleNamespace

import casadi as ca
import numpy as np
from acados_template import AcadosModel
from casadi import Function

from dq_nmpc.math.dq_functions import (
    make_body_to_inertial_rotation,
    make_inertial_to_body_rotation,
)
from dq_nmpc.math.dual_quaternion import DualQuaternion
from dq_nmpc.math.quaternion import Quaternion
from dq_nmpc.schema import CONTROL_SYM_NAMES, NMPCConfig

__all__ = [
    "apply_noise",
    "export_acados_model",
    "make_body_velocity_from_twist",
    "make_get_quaternion",
    "make_get_translation",
    "make_inertial_velocity_from_twist",
    "make_quadrotor_model",
]

# ---- Inner Symbols ---------------------------------------------------------

_qw_sym = ca.MX.sym("qw", 1, 1)
_qx_sym = ca.MX.sym("qx", 1, 1)
_qy_sym = ca.MX.sym("qy", 1, 1)
_qz_sym = ca.MX.sym("qz", 1, 1)
_q_sym = ca.vertcat(_qw_sym, _qx_sym, _qy_sym, _qz_sym)
_dw_sym = ca.MX.sym("dw", 1, 1)
_dx_sym = ca.MX.sym("dx", 1, 1)
_dy_sym = ca.MX.sym("dy", 1, 1)
_dz_sym = ca.MX.sym("dz", 1, 1)
_d_sym = ca.vertcat(_dw_sym, _dx_sym, _dy_sym, _dz_sym)
_dual_sym = ca.vertcat(_qw_sym, _qx_sym, _qy_sym, _qz_sym, _dw_sym, _dx_sym, _dy_sym, _dz_sym)

_qw_des_sym = ca.MX.sym("qw_1d", 1, 1)
_qx_des_sym = ca.MX.sym("qx_1d", 1, 1)
_qy_des_sym = ca.MX.sym("qy_1d", 1, 1)
_qz_des_sym = ca.MX.sym("qz_1d", 1, 1)
_q_des_sym = ca.vertcat(_qw_des_sym, _qx_des_sym, _qy_des_sym, _qz_des_sym)

_dw_des_sym = ca.MX.sym("dw_1d", 1, 1)
_dx_des_sym = ca.MX.sym("dx_1d", 1, 1)
_dy_des_sym = ca.MX.sym("dy_1d", 1, 1)
_dz_des_sym = ca.MX.sym("dz_1d", 1, 1)
_d_des_sym = ca.vertcat(_dw_des_sym, _dx_des_sym, _dy_des_sym, _dz_des_sym)
_dual_des_sym = ca.vertcat(
    _qw_des_sym,
    _qx_des_sym,
    _qy_des_sym,
    _qz_des_sym,
    _dw_des_sym,
    _dx_des_sym,
    _dy_des_sym,
    _dz_des_sym,
)

_vx_des_sym = ca.MX.sym("vx_1d", 1, 1)
_vy_des_sym = ca.MX.sym("vy_1d", 1, 1)
_vz_des_sym = ca.MX.sym("vz_1d", 1, 1)
_wx_des_sym = ca.MX.sym("wx_1d", 1, 1)
_wy_des_sym = ca.MX.sym("wy_1d", 1, 1)
_wz_des_sym = ca.MX.sym("wz_1d", 1, 1)
_w_des_sym = ca.vertcat(
    _wx_des_sym, _wy_des_sym, _wz_des_sym, _vx_des_sym, _vy_des_sym, _vz_des_sym
)

_dq_sym = DualQuaternion(q_real=Quaternion(q=_q_sym), q_dual=Quaternion(q=_d_sym))
_dq_des_sym = DualQuaternion(q_real=Quaternion(q=_q_des_sym), q_dual=Quaternion(q=_d_des_sym))

# ---- Rotation factories (imported from dq_functions.py) ----
_f_rotation_sym = make_body_to_inertial_rotation()
_f_rotation_inverse_sym = make_inertial_to_body_rotation()


# ---- DQ accessor factories ----
def _make_dualquat_get_all():
    values = _dq_sym.get[:, 0]
    return Function("dualquaternion_f", [_dual_sym], [values])


def make_get_translation():
    values = _dq_sym.get_trans.get[:, 0]
    return Function("f_trans", [_dual_sym], [values])


def _make_get_real_part():
    values = _dq_sym.Qr.get[:, 0]
    return Function("f_real", [_dual_sym], [values])


def _make_get_dual_part():
    values = _dq_sym.Qd.get[:, 0]
    return Function("f_dual", [_dual_sym], [values])


def make_get_quaternion():
    values = _dq_sym.Qr.get[:, 0]
    return Function("f_quat", [_dual_sym], [values])


_get_real_sym = _make_get_real_part()
_get_dual_sym = _make_get_dual_part()
_get_trans_sym = make_get_translation()
_get_quat_sym = make_get_quaternion()


# ---- DQ kinematics ----
def _dualquat_kinematics(quat, omega):
    quat_data = quat[0:4, 0]
    dual_data = quat[4:8, 0]
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

    omega = ca.vertcat(0.0, omega[0], omega[1], omega[2], 0.0, omega[3], omega[4], omega[5])
    q_dot = (1 / 2) * (Hplus @ omega) + aux_dual
    return q_dot


# ---- Velocity/twist mappings ----
def make_body_velocity_from_twist(_w_des_sym=_w_des_sym, _dual_des_sym=_dual_des_sym):
    twist = ca.vertcat(0.0, _w_des_sym[0:3, 0], 0.0, _w_des_sym[3:6, 0])
    w = _get_real_sym(twist)[1:4, 0]
    v = _get_dual_sym(twist)[1:4, 0]
    quat_aux = _get_quat_sym(_dual_des_sym)
    velocity = ca.vertcat(w, _f_rotation_inverse_sym(quat_aux, v))
    return Function("f_velocity", [_w_des_sym, _dual_des_sym], [velocity])


def make_inertial_velocity_from_twist(_w_des_sym=_w_des_sym, _dual_des_sym=_dual_des_sym):
    twist = ca.vertcat(0.0, _w_des_sym[0:3, 0], 0.0, _w_des_sym[3:6, 0])
    w = _get_real_sym(twist)[1:4, 0]
    dual = _get_dual_sym(twist)[1:4, 0]
    quat_aux = _get_quat_sym(_dual_des_sym)
    velocity = ca.vertcat(w, _f_rotation_sym(quat_aux, dual))
    return Function("f_velocity", [_w_des_sym, _dual_des_sym], [velocity])


# ---- DQ acceleration (rigid-body dynamics) ----
def _dualquat_acceleration(dual, omega, u, L):
    force = u[0, 0]
    torques = u[1:4, 0]
    J = ca.DM.zeros(3, 3)
    J[0, 0] = L[1]
    J[1, 1] = L[2]
    J[2, 2] = L[3]
    J_1 = ca.inv(J)
    e3 = ca.DM.zeros(3, 1)
    e3[2, 0] = 1.0

    w = omega[0:3, 0]
    v = omega[3:6, 0]
    q = _get_quat_sym(dual)

    F_r = -J_1 @ ca.cross(w, J @ w)
    F_d = ca.cross(v, w) - L[4] * _f_rotation_inverse_sym(q, e3)
    U_r = J_1 @ torques
    U_d = (force / L[0]) @ e3
    return ca.vertcat(F_r + U_r, F_d + U_d)


# ---- AcadosModel builders ----
def export_acados_model(config: NMPCConfig) -> SimpleNamespace:
    """Build acados model and quaternion-norm constraint from config.

    @return  SimpleNamespace with: model, constraint
    """
    nmpc = config.ocp
    L = [
        config.physics.mass,
        config.physics.ixx,
        config.physics.iyy,
        config.physics.izz,
        config.physics.gravity,
    ]

    constraint = ca.types.SimpleNamespace()

    model = AcadosModel()
    model.name = "quadrotor"
    model.z = []

    qw = ca.MX.sym("qw", 1, 1)
    qx = ca.MX.sym("qx", 1, 1)
    qy = ca.MX.sym("qy", 1, 1)
    qz = ca.MX.sym("qz", 1, 1)
    dw = ca.MX.sym("dw", 1, 1)
    dx = ca.MX.sym("dx", 1, 1)
    dy = ca.MX.sym("dy", 1, 1)
    dz = ca.MX.sym("dz", 1, 1)
    vx = ca.MX.sym("vx", 1, 1)
    vy = ca.MX.sym("vy", 1, 1)
    vz = ca.MX.sym("vz", 1, 1)
    wx = ca.MX.sym("wx", 1, 1)
    wy = ca.MX.sym("wy", 1, 1)
    wz = ca.MX.sym("wz", 1, 1)

    X = ca.vertcat(qw, qx, qy, qz, dw, dx, dy, dz, wx, wy, wz, vx, vy, vz)
    model.x = X

    twist = X[8:14, 0]
    dualquat = X[0:8, 0]

    qw_dot = ca.MX.sym("qw_dot", 1, 1)
    qx_dot = ca.MX.sym("qx_dot", 1, 1)
    qy_dot = ca.MX.sym("qy_dot", 1, 1)
    qz_dot = ca.MX.sym("qz_dot", 1, 1)
    dw_dot = ca.MX.sym("dw_dot", 1, 1)
    dx_dot = ca.MX.sym("dx_dot", 1, 1)
    dy_dot = ca.MX.sym("dy_dot", 1, 1)
    dz_dot = ca.MX.sym("dz_dot", 1, 1)
    vx_dot = ca.MX.sym("vx_dot", 1, 1)
    vy_dot = ca.MX.sym("vy_dot", 1, 1)
    vz_dot = ca.MX.sym("vz_dot", 1, 1)
    wx_dot = ca.MX.sym("wx_dot", 1, 1)
    wy_dot = ca.MX.sym("wy_dot", 1, 1)
    wz_dot = ca.MX.sym("wz_dot", 1, 1)

    X_dot = ca.vertcat(
        qw_dot,
        qx_dot,
        qy_dot,
        qz_dot,
        dw_dot,
        dx_dot,
        dy_dot,
        dz_dot,
        wx_dot,
        wy_dot,
        wz_dot,
        vx_dot,
        vy_dot,
        vz_dot,
    )

    u_syms = [ca.MX.sym(name) for name in CONTROL_SYM_NAMES]
    u = ca.vertcat(*u_syms)
    model.u = u

    dual_dot = _dualquat_kinematics(dualquat, twist)
    twist_dot = _dualquat_acceleration(dualquat, twist, u, L)
    f_expl = ca.vertcat(dual_dot, twist_dot)
    f_impl = X_dot - f_expl

    ref_params = ca.MX.sym("ref_params", nmpc.nx + nmpc.nu, 1)
    cost_params = ca.MX.sym("cost_params", nmpc.nx + nmpc.nx + nmpc.nu, 1)
    model.p = ca.vertcat(ref_params, cost_params)

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.xdot = X_dot

    norm_q = ca.norm_2(_get_quat_sym(X[0:8]))
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0

    return SimpleNamespace(model=model, constraint=constraint)


def make_quadrotor_model(L: list) -> AcadosModel:
    """Build acados model from raw system parameter list [mass, Ixx, Iyy, Izz, gravity].

    @param[in] L  List of 5 floats
    @return       AcadosModel with 14D DQ state, 4D control, and quaternion-norm constraint
    """
    constraint = ca.types.SimpleNamespace()

    qw = ca.MX.sym("qw", 1, 1)
    qx = ca.MX.sym("qx", 1, 1)
    qy = ca.MX.sym("qy", 1, 1)
    qz = ca.MX.sym("qz", 1, 1)
    dw = ca.MX.sym("dw", 1, 1)
    dx = ca.MX.sym("dx", 1, 1)
    dy = ca.MX.sym("dy", 1, 1)
    dz = ca.MX.sym("dz", 1, 1)
    vx = ca.MX.sym("vx", 1, 1)
    vy = ca.MX.sym("vy", 1, 1)
    vz = ca.MX.sym("vz", 1, 1)
    wx = ca.MX.sym("wx", 1, 1)
    wy = ca.MX.sym("wy", 1, 1)
    wz = ca.MX.sym("wz", 1, 1)

    X = ca.vertcat(qw, qx, qy, qz, dw, dx, dy, dz, wx, wy, wz, vx, vy, vz)
    twist = X[8:14, 0]
    dualquat = X[0:8, 0]

    qw_dot = ca.MX.sym("qw_dot", 1, 1)
    qx_dot = ca.MX.sym("qx_dot", 1, 1)
    qy_dot = ca.MX.sym("qy_dot", 1, 1)
    qz_dot = ca.MX.sym("qz_dot", 1, 1)
    dw_dot = ca.MX.sym("dw_dot", 1, 1)
    dx_dot = ca.MX.sym("dx_dot", 1, 1)
    dy_dot = ca.MX.sym("dy_dot", 1, 1)
    dz_dot = ca.MX.sym("dz_dot", 1, 1)
    vx_dot = ca.MX.sym("vx_dot", 1, 1)
    vy_dot = ca.MX.sym("vy_dot", 1, 1)
    vz_dot = ca.MX.sym("vz_dot", 1, 1)
    wx_dot = ca.MX.sym("wx_dot", 1, 1)
    wy_dot = ca.MX.sym("wy_dot", 1, 1)
    wz_dot = ca.MX.sym("wz_dot", 1, 1)

    X_dot = ca.vertcat(
        qw_dot,
        qx_dot,
        qy_dot,
        qz_dot,
        dw_dot,
        dx_dot,
        dy_dot,
        dz_dot,
        wx_dot,
        wy_dot,
        wz_dot,
        vx_dot,
        vy_dot,
        vz_dot,
    )

    u_syms = [ca.MX.sym(name) for name in CONTROL_SYM_NAMES]
    u = ca.vertcat(*u_syms)

    dual_dot = _dualquat_kinematics(dualquat, twist)
    twist_dot = _dualquat_acceleration(dualquat, twist, u, L)

    norm_q = ca.norm_2(_get_quat_sym(X[0:8]))
    constraint.norm = Function("norm", [X], [norm_q])
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0

    f_expl = ca.vertcat(dual_dot, twist_dot)
    f_impl = X_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = X_dot
    model.u = u
    model.z = []
    model.p = ca.MX.sym("p", 18, 1)
    model.name = "quadrotor"

    return model, constraint


def apply_noise(x, noise):
    dual = x[0:8]
    twist = x[8:14]
    trans = _get_trans_sym(dual)
    trans_np = np.array(trans[1:4]).reshape((3,))
    quat_data = _get_quat_sym(dual)

    noise_position = noise[0:3]
    trans_noise = trans_np + noise_position
    trans_noise_aux = np.array([0.0, trans_noise[0], trans_noise[1], trans_noise[2]])

    noise_quat = noise[3:6]
    squared_norm_delta = noise_quat[0] ** 2 + noise_quat[1] ** 2 + noise_quat[2] ** 2
    q_delta = np.zeros((4, 1))
    if squared_norm_delta > 0:
        norm_delta = np.sqrt(squared_norm_delta)
        sin_by_delta = np.sin(norm_delta) / norm_delta
        q_delta[0, 0] = np.cos(norm_delta)
        q_delta[1, 0] = sin_by_delta * noise_quat[0]
        q_delta[2, 0] = sin_by_delta * noise_quat[1]
        q_delta[3, 0] = sin_by_delta * noise_quat[2]
    else:
        q_delta[0, 0] = 1.0

    H_r = ca.vertcat(
        ca.horzcat(quat_data[0], -quat_data[1], -quat_data[2], -quat_data[3]),
        ca.horzcat(quat_data[1], quat_data[0], -quat_data[3], quat_data[2]),
        ca.horzcat(quat_data[2], quat_data[3], quat_data[0], -quat_data[1]),
        ca.horzcat(quat_data[3], -quat_data[2], quat_data[1], quat_data[0]),
    )
    quat_noise_aux = np.array(H_r) @ q_delta
    Q1_pose = DualQuaternion.from_pose(quat=quat_noise_aux, trans=trans_noise_aux)
    values_pose = np.array(Q1_pose.get[:, 0]).reshape((8,))

    values_twist = np.array(twist).reshape((6,)) + noise[6:12]
    return np.array(
        [
            values_pose[0],
            values_pose[1],
            values_pose[2],
            values_pose[3],
            values_pose[4],
            values_pose[5],
            values_pose[6],
            values_pose[7],
            values_twist[0],
            values_twist[1],
            values_twist[2],
            values_twist[3],
            values_twist[4],
            values_twist[5],
        ]
    )
