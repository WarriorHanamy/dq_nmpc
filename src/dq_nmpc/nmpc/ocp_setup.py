"""Acados OCP solver factory — NONLINEAR_LS cost type.

Single public function: ``solver(config, *, codegen=True)``.
Reads all configuration from ``NMPCConfig`` schema fields.
"""

from __future__ import annotations

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import vertcat

from dq_nmpc.nmpc.dq_functions import dualquat_mul_conj_expr, log_map_dualquat_expr
from dq_nmpc.nmpc.dynamics import export_acados_model
from dq_nmpc.schema import (
    CONTROL_INPUT,
    NMPC_REF_DIM,
    NMPC_REF_DQ_SLICE,
    NMPC_REF_OMEGA_SLICE,
    NMPC_REF_UNOM_SLICE,
    NMPC_REF_VEL_SLICE,
    NMPCConfig,
)

__all__ = ["solver"]


def solver(
    config: NMPCConfig,
    *,
    codegen: bool = True,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    ocp_cfg = config.ocp
    ubu = np.array(ocp_cfg.ubu, dtype=np.float64)
    lbu = np.array(ocp_cfg.lbu, dtype=np.float64)
    Qp = np.array(ocp_cfg.Q, dtype=np.float64)
    Rp = np.array(ocp_cfg.R, dtype=np.float64)

    m = export_acados_model(config)

    ocp = AcadosOcp()
    ocp.code_export_directory = "c_generated_code"

    ocp.model = m.model
    ocp.p = m.model.p
    ocp.dims.N = ocp_cfg.horizon_steps

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    dual_d = ocp.p[NMPC_REF_DQ_SLICE]
    dual = m.model.x[0:8]
    u_nom = ocp.p[NMPC_REF_UNOM_SLICE]
    u = m.model.u
    w_b_d = ocp.p[NMPC_REF_OMEGA_SLICE]
    w_b = m.model.x[8:11]
    v_b_d = ocp.p[NMPC_REF_VEL_SLICE]
    v_b = m.model.x[11:14]

    dq_err = dualquat_mul_conj_expr(dual_d, dual)
    ln_full = log_map_dualquat_expr(dq_err)
    ln_err = vertcat(ln_full[1:4], ln_full[5:8])

    Q_pose_np = np.zeros((6, 6))
    for i in range(3):
        Q_pose_np[i, i] = Qp[i]
        Q_pose_np[i + 3, i + 3] = Qp[3 + i]

    R_ctrl_np = np.diag(Rp)

    Q_angvel_np = np.diag(Qp[6:9])
    Q_vel_np = np.diag(Qp[9:12])

    print(f"Q_pose      = {np.diag(Q_pose_np)}")
    print(f"Q_angvel    = {np.diag(Q_angvel_np)}")
    print(f"Q_vel       = {np.diag(Q_vel_np)}")
    print(f"R_ctrl      = {np.diag(R_ctrl_np)}")

    W = np.block(
        [
            [Q_pose_np, np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((6, 4))],
            [np.zeros((3, 6)), Q_angvel_np, np.zeros((3, 3)), np.zeros((3, 4))],
            [np.zeros((3, 6)), np.zeros((3, 3)), Q_vel_np, np.zeros((3, 4))],
            [np.zeros((4, 6)), np.zeros((4, 3)), np.zeros((4, 3)), R_ctrl_np],
        ]
    )
    W_e = np.block(
        [
            [Q_pose_np, np.zeros((6, 3)), np.zeros((6, 3))],
            [np.zeros((3, 6)), Q_angvel_np, np.zeros((3, 3))],
            [np.zeros((3, 6)), np.zeros((3, 3)), Q_vel_np],
        ]
    )

    ocp.model.cost_y_expr = vertcat(ln_err, w_b - w_b_d, v_b - v_b_d, u - u_nom)
    ocp.model.cost_y_expr_e = vertcat(ln_err, w_b - w_b_d, v_b - v_b_d)

    ny = ln_err.shape[0] + 3 + 3 + 4
    ny_e = ln_err.shape[0] + 3 + 3
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)
    ocp.cost.W = W
    ocp.cost.W_e = W_e

    ref_params = np.zeros(NMPC_REF_DIM, dtype=np.float64)
    ref_params[0] = 1.0
    ocp.parameter_values = ref_params

    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = lbu.copy()
    ocp.constraints.ubu = ubu.copy()
    ocp.constraints.idxbu = np.arange(len(CONTROL_INPUT))
    x0 = np.zeros(m.model.x.shape[0], dtype=np.float64)
    x0[0] = 1.0
    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = ocp_cfg.horizon_time

    ocp.solver_options.ext_fun_compile_flags = "-Ofast -march=native"
    ocp.solver_options.hpipm_mode = "SPEED"

    ## NLP Solver Settings
    ocp.solver_options.nlp_solver_max_iter = 200  # Maximum iterations
    ocp.solver_options.nlp_solver_tol_stat = 1e-2  # Tolerance for stationarity
    ocp.solver_options.print_level = 0  # Suppress print output

    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.sim_method_newton_iter = 2

    acados_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_mpc.json", build=codegen, generate=codegen
    )
    return acados_solver, ocp
