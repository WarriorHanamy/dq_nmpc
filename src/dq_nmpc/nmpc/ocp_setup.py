"""Acados OCP solver factory.

Single public function: ``solver(config, *, codegen=True)``.
Reads all configuration from ``NMPCConfig`` schema fields;
cost terms are imported from ``dq_functions.py`` as compiled CasADi Functions.
"""

from __future__ import annotations

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat

from dq_nmpc.math.dq_algebra import dualquat_mul_conj, log_map_dualquat
from dq_nmpc.math.dq_functions import make_body_to_inertial_rotation
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

BODY_TO_INERTIAL_FN = make_body_to_inertial_rotation()


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

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    R_ctrl = MX.zeros(4, 4)
    for i in range(4):
        R_ctrl[i, i] = Rp[i]

    dual_d = ocp.p[NMPC_REF_DQ_SLICE]
    dual = m.model.x[0:8]
    u_nom = ocp.p[NMPC_REF_UNOM_SLICE]
    u = m.model.u
    w_b_d = ocp.p[NMPC_REF_OMEGA_SLICE]
    w_b = m.model.x[8:11]
    v_i_d = ocp.p[NMPC_REF_VEL_SLICE]
    v_b = m.model.x[11:14]
    v_i = BODY_TO_INERTIAL_FN(m.model.x[0:4], v_b)

    dq_err = dualquat_mul_conj(dual_d, dual)
    ln_full = log_map_dualquat(dq_err)
    ln_err = vertcat(ln_full[1:4], ln_full[5:8])

    Q_pose = MX.zeros(6, 6)
    for i in range(3):
        Q_pose[i, i] = Qp[1 + i]
        Q_pose[i + 3, i + 3] = Qp[5 + i]

    pose_cost = ln_err.T @ Q_pose @ ln_err
    ctrl_cost = (u_nom - u).T @ R_ctrl @ (u_nom - u)

    Q_angvel = MX.zeros(3, 3)
    for i in range(3):
        Q_angvel[i, i] = Qp[8 + i]

    Q_vel = MX.zeros(3, 3)
    for i in range(3):
        Q_vel[i, i] = Qp[11 + i]
    angvel_cost = (w_b - w_b_d).T @ Q_angvel @ (w_b - w_b_d)
    vel_cost = (v_i - v_i_d).T @ Q_vel @ (v_i - v_i_d)

    ocp.model.cost_expr_ext_cost = pose_cost + ctrl_cost + angvel_cost + vel_cost
    ocp.model.cost_expr_ext_cost_e = pose_cost + angvel_cost + vel_cost

    ref_params = np.zeros(NMPC_REF_DIM, dtype=np.float64)
    ref_params[0] = 1.0
    ocp.parameter_values = np.concatenate([ref_params, np.ones(COST_PARAMS_DIM)])

    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = lbu.copy()
    ocp.constraints.ubu = ubu.copy()
    ocp.constraints.idxbu = np.arange(len(CONTROL_INPUT))
    ocp.constraints.x0 = ref_params[: m.model.x.rows()]

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = ocp_cfg.horizon_steps // 4
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"  # Implicit Runge-Kutta (IRK)

    ## Regularization (stabilizes optimization)
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = ocp_cfg.control_update_interval
    ocp.solver_options.tf = ocp_cfg.horizon_time

    ## Compilation flags for external functions (optional for performance)
    ocp.solver_options.ext_fun_compile_flags = "-Ofast -march=native"
    ocp.solver_options.hpipm_mode = "SPEED"  # Prioritize speed in QP solver

    # Parallelization
    ocp.solver_options.cg_use_openmp = True  # Enable OpenMP parallelization
    ocp.solver_options.cg_hardcode_constraints = False  # Allow runtime constraint changes
    ocp.solver_options.cg_use_variable_weighting_matrix = True  # Support time-varying costs

    ocp.solver_options.sim_method_num_stages = 4  # IRK-GL4: 4 stages for accuracy
    ocp.solver_options.sim_method_num_steps = 1  # Number of integration steps
    ocp.solver_options.sim_method_newton_iter = 2  # Newton iterations for convergence

    acados_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_mpc.json", build=codegen, generate=codegen
    )
    return acados_solver, ocp
