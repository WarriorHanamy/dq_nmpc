import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat

from dq_nmpc.nmpc.dynamics import export_acados_model, make_quadrotor_model
from dq_nmpc.schema import CONTROL_INPUT, control_index


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
    R[control_index("thrust"), control_index("thrust")] = 20 / F_max
    R[control_index("tau_x"), control_index("tau_x")] = 60 / tau_1_max
    R[control_index("tau_y"), control_index("tau_y")] = 60 / tau_2_max
    R[control_index("tau_z"), control_index("tau_z")] = 60 / tau_3_max

    # Desired Dual Quaternion
    dual_d = ocp.p[0:8]

    # Current Dual Quaternion
    dual = model.x[0:8]

    error = dual_error(dual_d, dual)
    ln_error_full = ln(error)
    ln_error = vertcat(ln_error_full[1:4], ln_error_full[5:8])

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

    # Parameter initial values: ref_params (nx + nu) + cost_params (nx + nx + nu)
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ref_params = np.zeros(nx + nu)
    ref_params[0] = 1.0
    cost_params = np.ones(nx + nx + nu)
    ocp.parameter_values = np.concatenate([ref_params, cost_params])

    # Constraints
    ocp.constraints.constr_type = "BGH"
    n_ctrl = len(CONTROL_INPUT)
    ocp.constraints.lbu = np.zeros(n_ctrl)
    ocp.constraints.ubu = np.zeros(n_ctrl)
    ocp.constraints.lbu[control_index("thrust")] = F_min
    ocp.constraints.ubu[control_index("thrust")] = F_max
    ocp.constraints.lbu[control_index("tau_x")] = tau_1_min
    ocp.constraints.ubu[control_index("tau_x")] = tau_1_max
    ocp.constraints.lbu[control_index("tau_y")] = tau_2_min
    ocp.constraints.ubu[control_index("tau_y")] = tau_2_max
    ocp.constraints.lbu[control_index("tau_z")] = tau_3_min
    ocp.constraints.ubu[control_index("tau_z")] = tau_3_max
    ocp.constraints.idxbu = np.arange(n_ctrl)
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
