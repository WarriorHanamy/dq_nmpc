import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX

from dq_nmpc.nmpc.dynamics import export_acados_model, make_quadrotor_model


def _setup_ocp(
    ocp,
    model,
    constraint,
    error_lie_2,
    dual_error,
    ln,
    Ad,
    conjugate,
    rotation,
    N_horizon,
    t_horizon,
    ts,
    F_max,
    F_min,
    tau_1_max,
    tau_1_min,
    tau_2_max,
    tau_2_min,
    tau_3_max,
    tau_3_min,
    x0,
):
    """Common OCP configuration shared by create_ocp_solver and solver()."""
    ocp.model = model
    ocp.p = model.p
    ocp.dims.N = N_horizon

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Control effort using gain matrices
    R = MX.zeros(4, 4)
    R[0, 0] = 20 / F_max
    R[1, 1] = 60 / tau_1_max
    R[2, 2] = 60 / tau_2_max
    R[3, 3] = 60 / tau_3_max

    # Desired Dual Quaternion
    dual_d = ocp.p[0:8]

    # Current Dual Quaternion
    dual = model.x[0:8]

    error_total_lie = error_lie_2(dual_d, dual)
    error = dual_error(dual_d, dual)
    error_c = conjugate(error)
    ln_error = ln(error)

    # Inputs
    nominal_input = ocp.p[14:18]
    error_nominal_input = nominal_input - model.u[0:4]

    # Angular velocities
    w_b = model.x[8:11]
    v_b = model.x[11:14]
    v_i = rotation(model.x[0:4], v_b)

    w_b_d = ocp.p[8:11]
    v_i_d = ocp.p[11:14]
    error_w = w_b - w_b_d
    error_v = v_i - v_i_d

    # Gain Matrix complete error
    Q_l = MX.zeros(6, 6)
    Q_l[0, 0] = 0.5
    Q_l[1, 1] = 0.5
    Q_l[2, 2] = 0.5
    Q_l[3, 3] = 2
    Q_l[4, 4] = 2
    Q_l[5, 5] = 2

    ocp.model.cost_expr_ext_cost = (
        10 * (ln_error.T @ Q_l @ ln_error)
        + 1 * (error_nominal_input.T @ R @ error_nominal_input)
        + 1 * (error_w.T @ error_w)
        + 1 * (error_v.T @ error_v)
    )
    ocp.model.cost_expr_ext_cost_e = (
        10 * (ln_error.T @ Q_l @ ln_error) + 1 * (error_w.T @ error_w) + 1 * (error_v.T @ error_v)
    )

    # Parameter initial values
    ocp.parameter_values = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # Constraints
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = np.array([F_min, tau_1_min, tau_2_min, tau_3_min])
    ocp.constraints.ubu = np.array([F_max, tau_1_max, tau_2_max, tau_3_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = x0

    # Nonlinear constraints (quaternion unit norm)
    ocp.model.con_h_expr = constraint.expr
    nsbx = 0
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx

    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.Zl = 100 * np.ones((ns,))
    ocp.cost.Zu = 100 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))

    ocp.constraints.lh = np.array([constraint.min])
    ocp.constraints.uh = np.array([constraint.max])
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))

    # Solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = ts
    ocp.solver_options.tf = t_horizon


def create_ocp_solver(
    x0,
    N_horizon,
    t_horizon,
    F_max,
    F_min,
    tau_1_max,
    tau_1_min,
    tau_2_max,
    tau_2_min,
    tau_3_max,
    tau_3_min,
    L,
    ts,
    path,
) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.code_export_directory = path

    model, get_trans, get_quat, constraint, error_lie_2, dual_error, ln, Ad, conjugate, rotation = (
        make_quadrotor_model(L)
    )

    _setup_ocp(
        ocp,
        model,
        constraint,
        error_lie_2,
        dual_error,
        ln,
        Ad,
        conjugate,
        rotation,
        N_horizon,
        t_horizon,
        ts,
        F_max,
        F_min,
        tau_1_max,
        tau_1_min,
        tau_2_max,
        tau_2_min,
        tau_3_max,
        tau_3_min,
        x0,
    )
    return ocp


def solver(params, flag=True):
    """Create and build acados OCP solver from NMPC params dict.

    @param[in] params  NMPC configuration dict (from NMPCConfig.to_params_dict())
    @param[in] flag    build and generate code (True) or load existing (False)
    @return (acados_solver, ocp)
    """
    nmpc = params["nmpc"]

    model, get_trans, get_quat, constraint, error_lie_2, dual_error, ln, Ad, conjugate, rotation = (
        export_acados_model(params)
    )

    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp = AcadosOcp()
    ocp.code_export_directory = "c_generated_code"

    _setup_ocp(
        ocp,
        model,
        constraint,
        error_lie_2,
        dual_error,
        ln,
        Ad,
        conjugate,
        rotation,
        nmpc["horizon_steps"],
        nmpc["horizon_time"],
        nmpc["ts"],
        nmpc["ubu"][0],
        nmpc["lbu"][0],
        nmpc["ubu"][1],
        nmpc["lbu"][1],
        nmpc["ubu"][2],
        nmpc["lbu"][2],
        nmpc["ubu"][3],
        nmpc["lbu"][3],
        x0,
    )

    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_mpc.json", build=flag, generate=flag)
    return acados_solver, ocp
