"""Quadrotor dynamics in dual-quaternion representation for acados NMPC.

Public API
  export_acados_model     build AcadosModel from NMPCConfig
"""

from __future__ import annotations

from types import SimpleNamespace

import casadi as ca
from acados_template import AcadosModel
from casadi import Function

from dq_nmpc.math.dq_functions import (
    make_dualquat_acceleration,
    make_dualquat_kinematics,
)
from dq_nmpc.schema import CONTROL_SYM_NAMES, NMPC_REF_DIM, NMPCConfig

__all__ = ["export_acados_model"]

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

_dual_sym = ca.vertcat(_qw_sym, _qx_sym, _qy_sym, _qz_sym, _dw_sym, _dx_sym, _dy_sym, _dz_sym)

_get_quat_sym = Function("f_quat", [_dual_sym], [_q_sym])


# ---- AcadosModel builders ----
def export_acados_model(config: NMPCConfig) -> SimpleNamespace:
    """Build acados model and quaternion-norm constraint from config.

    @return  SimpleNamespace with: model, constraint
    """
    L = [
        config.physics.mass,
        config.physics.ixx,
        config.physics.iyy,
        config.physics.izz,
        config.physics.gravity,
    ]

    _dq_kin = make_dualquat_kinematics()
    _dq_accel = make_dualquat_acceleration(L)

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

    dual_dot = _dq_kin(dualquat, twist)
    twist_dot = _dq_accel(dualquat, twist, u)
    f_expl = ca.vertcat(dual_dot, twist_dot)
    f_impl = X_dot - f_expl

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.xdot = X_dot

    model.p = ca.MX.sym("p", NMPC_REF_DIM, 1)

    norm_q = ca.norm_2(_get_quat_sym(X[0:8]))
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0

    return SimpleNamespace(model=model, constraint=constraint)
