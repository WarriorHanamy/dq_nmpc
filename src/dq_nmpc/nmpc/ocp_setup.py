"""Acados OCP solver factory.

Single public function: ``solver(config, *, codegen=True)``.
Reads all configuration from ``NMPCConfig`` schema fields.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat

from dq_nmpc.nmpc.dynamics import export_acados_model
from dq_nmpc.schema import (
    CONTROL_INPUT,
    COST_PARAMS_DIM,
    NMPC_REF_DIM,
    NMPC_REF_DQ_SLICE,
    NMPC_REF_OMEGA_SLICE,
    NMPC_REF_UNOM_SLICE,
    NMPC_REF_VEL_SLICE,
    NMPCConfig,
)

__all__ = ["solver"]


def _build_model(config: NMPCConfig) -> SimpleNamespace:
    """Build acados model and CasADi function bundle from config.

    @return  SimpleNamespace with: model, constraint, dual_error,
             ln, rotation, conjugate, Ad
    """
    params = config.to_params_dict()
    m, _, _, c, err, dual, logmap, Ad, conj, rot = export_acados_model(params)
    return SimpleNamespace(
        model=m,
        constraint=c,
        dual_error=dual,
        ln=logmap,
        rotation=rot,
        conjugate=conj,
        Ad=Ad,
    )


def solver(
    config: NMPCConfig,
    *,
    codegen: bool = True,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create and build acados OCP solver from NMPCConfig.

    @param[in] config   Validated NMPC configuration
    @param[in] codegen  Build and generate C code (True) or load existing (False)
    @return  (acados_solver, ocp)
    """
    nmpc = config.nmpc
    ubu = np.array(nmpc.ubu, dtype=np.float64)
    lbu = np.array(nmpc.lbu, dtype=np.float64)

    m = _build_model(config)

    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp = AcadosOcp()
    ocp.code_export_directory = "c_generated_code"

    ocp.model = m.model
    ocp.p = m.model.p
    ocp.dims.N = nmpc.horizon_steps

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    R = MX.zeros(4, 4)
    for i, name in enumerate(CONTROL_INPUT):
        R[i, i] = 20 / ubu[i] if "thrust" in name else 60 / ubu[i]

    dual_d = ocp.p[NMPC_REF_DQ_SLICE]
    dual = m.model.x[0:8]
    dq_err = m.dual_error(dual_d, dual)
    ln_full = m.ln(dq_err)
    ln_err = vertcat(ln_full[1:4], ln_full[5:8])

    u_nom = ocp.p[NMPC_REF_UNOM_SLICE]
    err_u = u_nom - m.model.u[0:4]

    w_b = m.model.x[8:11]
    v_b = m.model.x[11:14]
    v_i = m.rotation(m.model.x[0:4], v_b)
    w_b_d = ocp.p[NMPC_REF_OMEGA_SLICE]
    v_i_d = ocp.p[NMPC_REF_VEL_SLICE]
    err_w = w_b - w_b_d
    err_v = v_i - v_i_d

    Q_l = MX.zeros(6, 6)
    for i in range(3):
        Q_l[i, i] = 0.5
    for i in range(3, 6):
        Q_l[i, i] = 2.0

    ocp.model.cost_expr_ext_cost = (
        10 * (ln_err.T @ Q_l @ ln_err)
        + 1 * (err_u.T @ R @ err_u)
        + 1 * (err_w.T @ err_w)
        + 1 * (err_v.T @ err_v)
    )
    ocp.model.cost_expr_ext_cost_e = (
        10 * (ln_err.T @ Q_l @ ln_err) + 1 * (err_w.T @ err_w) + 1 * (err_v.T @ err_v)
    )

    ref_params = np.zeros(NMPC_REF_DIM, dtype=np.float64)
    ref_params[0] = 1.0
    ocp.parameter_values = np.concatenate([ref_params, np.ones(COST_PARAMS_DIM)])

    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = lbu.copy()
    ocp.constraints.ubu = ubu.copy()
    ocp.constraints.idxbu = np.arange(len(CONTROL_INPUT))
    ocp.constraints.x0 = x0

    ocp.model.con_h_expr = m.constraint.expr
    nh = m.constraint.expr.shape[0]
    ns = nh
    ocp.cost.zl = 100 * np.ones(ns)
    ocp.cost.Zl = 100 * np.ones(ns)
    ocp.cost.Zu = 100 * np.ones(ns)
    ocp.cost.zu = 100 * np.ones(ns)
    ocp.constraints.lh = np.array([m.constraint.min])
    ocp.constraints.uh = np.array([m.constraint.max])
    ocp.constraints.lsh = np.zeros(ns)
    ocp.constraints.ush = np.zeros(ns)
    ocp.constraints.idxsh = np.arange(ns)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = nmpc.horizon_steps // 4
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = nmpc.control_update_interval
    ocp.solver_options.tf = nmpc.horizon_time

    acados_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_mpc.json", build=codegen, generate=codegen
    )
    return acados_solver, ocp
