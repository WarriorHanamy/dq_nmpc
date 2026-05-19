import casadi as ca
import numpy as np
from acados_template import AcadosModel
from casadi import Function

from dq_nmpc.math.dq_algebra import (
    adjoint_map,
    dualquat_conjugate,
    dualquat_mul_conj,
    log_error_dualquat,
    log_map_dualquat,
)
from dq_nmpc.math.dual_quaternion import DualQuaternion
from dq_nmpc.math.quaternion import Quaternion

# Sample time symbolic
_ts_sym = ca.MX.sym("ts", 1, 1)

# Defining Dual Quaternion informtatio
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

# Creating auxiliar variables
_dual_sym = ca.vertcat(_qw_sym, _qx_sym, _qy_sym, _qz_sym, _dw_sym, _dx_sym, _dy_sym, _dz_sym)

# Defining Desired Frame
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

# Symbolic Variables
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

# Defining the desired Velocity using symbolics
_vx_des_sym = ca.MX.sym("vx_1d", 1, 1)
_vy_des_sym = ca.MX.sym("vy_1d", 1, 1)
_vz_des_sym = ca.MX.sym("vz_1d", 1, 1)

_wx_des_sym = ca.MX.sym("wx_1d", 1, 1)
_wy_des_sym = ca.MX.sym("wy_1d", 1, 1)
_wz_des_sym = ca.MX.sym("wz_1d", 1, 1)

_Vd_sym = ca.vertcat(0.0, _vx_des_sym, _vy_des_sym, _vz_des_sym)
_Wd_sym = ca.vertcat(0.0, _wx_des_sym, _wy_des_sym, _wz_des_sym)

# Symbolic variables desired velocities
_w_des_sym = ca.vertcat(
    _wx_des_sym, _wy_des_sym, _wz_des_sym, _vx_des_sym, _vy_des_sym, _vz_des_sym
)

# Defining the control gains using symbolic variables
_kr1_sym = ca.MX.sym("kr1", 1, 1)
_kr2_sym = ca.MX.sym("kr2", 1, 1)
_kr3_sym = ca.MX.sym("kr3", 1, 1)

_kd1_sym = ca.MX.sym("kd1", 1, 1)
_kd2_sym = ca.MX.sym("kd2", 1, 1)
_kd3_sym = ca.MX.sym("kd3", 1, 1)

_Kr_sym = ca.vertcat(0.0, _kr1_sym, _kr2_sym, _kr3_sym)
_Kd_sym = ca.vertcat(0.0, _kd1_sym, _kd2_sym, _kd3_sym)


# Creating states of the current dualquaternion
_dq_sym = DualQuaternion(q_real=Quaternion(q=_q_sym), q_dual=Quaternion(q=_d_sym))

# Creating the desired quaternion
_dq_des_sym = DualQuaternion(q_real=Quaternion(q=_q_des_sym), q_dual=Quaternion(q=_d_des_sym))

# Creating the Desired dualquaternion twist
_twist_des_sym = DualQuaternion(q_real=Quaternion(q=_Wd_sym), q_dual=Quaternion(q=_Vd_sym))


# Quaternion rotation
def make_body_to_inertial_rotation():
    # Function that enables the rotation of a vector using quaternions

    # Creation of the symbolic variables for the quaternion and the vector
    quat_aux_1 = ca.MX.sym("quat_aux_1", 4, 1)
    vector_aux_1 = ca.MX.sym("vector_aux_1", 3, 1)

    # Defining the pure quaternion based on the vector information
    vector = ca.vertcat(0.0, vector_aux_1)

    # Compute conjugate of the quaternion
    quat = quat_aux_1
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])

    # v' = q x v x q*
    # Rotation to the inertial frame

    H_plus_q = ca.vertcat(
        ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
        ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
        ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
        ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]),
    )

    # Computing the first multiplication
    aux_value = H_plus_q @ vector

    # Multiplication by the conjugate part
    H_plus_aux = ca.vertcat(
        ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
        ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
        ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
        ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]),
    )

    # Computing the vector rotate respect the quaternion
    vector_i = H_plus_aux @ quat_c

    # Create function
    f_rot = ca.Function("f_rot", [quat_aux_1, vector_aux_1], [vector_i[1:4, 0]])
    return f_rot


def make_inertial_to_body_rotation():
    # Creation of the symbolic variables for the quaternion and the vector
    quat_aux_1 = ca.MX.sym("quat_aux_1", 4, 1)
    vector_aux_1 = ca.MX.sym("vector_aux_1", 3, 1)

    # Auxiliary pure quaternion based on the information of the vector
    vector = ca.vertcat(0.0, vector_aux_1)

    # Quaternion
    quat = quat_aux_1

    # Quaternion conjugate
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])
    # v' = q* x v x q
    # Rotation to the body Frame

    # QUaternion Multiplication vector form
    H_plus_q_c = ca.vertcat(
        ca.horzcat(quat_c[0, 0], -quat_c[1, 0], -quat_c[2, 0], -quat_c[3, 0]),
        ca.horzcat(quat_c[1, 0], quat_c[0, 0], -quat_c[3, 0], quat_c[2, 0]),
        ca.horzcat(quat_c[2, 0], quat_c[3, 0], quat_c[0, 0], -quat_c[1, 0]),
        ca.horzcat(quat_c[3, 0], -quat_c[2, 0], quat_c[1, 0], quat_c[0, 0]),
    )

    # First Multiplication
    aux_value = H_plus_q_c @ vector

    # Quaternion multiplication second element
    H_plus_aux = ca.vertcat(
        ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
        ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
        ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
        ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]),
    )

    # Rotated vector repected to the body frame
    vector_b = H_plus_aux @ quat

    # Defining function using casadi
    f_rot_inv = ca.Function("f_rot_inv", [quat_aux_1, vector_aux_1], [vector_b[1:4, 0]])
    return f_rot_inv


# Creating functions which are going to be used later
# _f_rotation_sym move a vector from the body frame to the inertial frame
_f_rotation_sym = make_body_to_inertial_rotation()
# _f_rotation_sym move a vector from the inertial frame to the body frame
_f_rotation_inverse_sym = make_inertial_to_body_rotation()


def rotate_vector_body_to_inertial(quat_aux_1, vector_aux_1):
    # Function that enables the rotation of a vector using quaternions

    # Defining the pure quaternion based on the vector information
    vector = ca.vertcat(0.0, vector_aux_1)

    # Compute conjugate of the quaternion
    quat = quat_aux_1
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])

    # v' = q x v x q*
    # Rotation to the inertial frame

    H_plus_q = ca.vertcat(
        ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
        ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
        ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
        ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]),
    )

    # Computing the first multiplication
    aux_value = H_plus_q @ vector

    # Multiplication by the conjugate part
    H_plus_aux = ca.vertcat(
        ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
        ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
        ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
        ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]),
    )

    # Computing the vector rotate respect the quaternion
    vector_i = H_plus_aux @ quat_c

    return vector_i[1:4, 0]


def make_dualquat_get_all():
    # Function that obtains the elements of the dual quaternion  real an dual part
    values = _dq_sym.get[:, 0]
    dualquaternion_f = Function("dualquaternion_f", [_dual_sym], [values])
    return dualquaternion_f


def make_get_translation():
    values = _dq_sym.get_trans.get[:, 0]
    f_trans = Function("f_trans", [_dual_sym], [values])
    return f_trans


def make_get_real_part():
    values = _dq_sym.Qr.get[:, 0]
    f_real = Function("f_real", [_dual_sym], [values])
    return f_real


def make_get_dual_part():
    values = _dq_sym.Qd.get[:, 0]
    f_dual = Function("f_dual", [_dual_sym], [values])
    return f_dual


def make_get_quaternion():
    values = _dq_sym.Qr.get[:, 0]
    f_quat = Function("f_quat", [_dual_sym], [values])
    return f_quat


# Creating Functions
_get_real_sym = make_get_real_part()
_get_dual_sym = make_get_dual_part()
_get_trans_sym = make_get_translation()
_get_quat_sym = make_get_quaternion()


# Creation of dualquaternion kinemtics
def dualquat_kinematics(quat, omega):
    # Functions that computes the differential kinematics based on dualquaternions

    # Split values real and dual
    quat_data = quat[0:4, 0]
    dual_data = quat[4:8, 0]

    # Auxiliary variable in order to avoid numerical issues
    # norm_r = ca.dot(quat_data, quat_data)
    K_quat = 10
    # norm_d = 2*(quat_data.T@dual_data)

    norm_r = ca.norm_2(quat_data)
    norm_d = 2 * ca.dot(quat_data, dual_data)

    quat_error = 1 - norm_r
    dual_error = norm_d

    aux_1 = quat_data * (K_quat * quat_error)
    aux_2 = dual_data * (0 * dual_error)

    aux_dual = ca.vertcat(aux_1, aux_2)

    # Creatin aux Variables
    H_r_plus = ca.vertcat(
        ca.horzcat(quat_data[0, 0], -quat_data[1, 0], -quat_data[2, 0], -quat_data[3, 0]),
        ca.horzcat(quat_data[1, 0], quat_data[0, 0], -quat_data[3, 0], quat_data[2, 0]),
        ca.horzcat(quat_data[2, 0], quat_data[3, 0], quat_data[0, 0], -quat_data[1, 0]),
        ca.horzcat(quat_data[3, 0], -quat_data[2, 0], quat_data[1, 0], quat_data[0, 0]),
    )

    H_d_plus = ca.vertcat(
        ca.horzcat(dual_data[0, 0], -dual_data[1, 0], -dual_data[2, 0], -dual_data[3, 0]),
        ca.horzcat(dual_data[1, 0], dual_data[0, 0], -dual_data[3, 0], dual_data[2, 0]),
        ca.horzcat(dual_data[2, 0], dual_data[3, 0], dual_data[0, 0], -dual_data[1, 0]),
        ca.horzcat(dual_data[3, 0], -dual_data[2, 0], dual_data[1, 0], dual_data[0, 0]),
    )
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros), ca.horzcat(H_d_plus, H_r_plus))

    # Auxiliar variable veloicities
    omega = ca.vertcat(
        0.0, omega[0, 0], omega[1, 0], omega[2, 0], 0.0, omega[3, 0], omega[4, 0], omega[5, 0]
    )
    q_dot = (1 / 2) * (Hplus @ omega) + aux_dual
    return q_dot


def make_body_velocity_from_twist(_w_des_sym=_w_des_sym, _dual_des_sym=_dual_des_sym):
    # Funtions that computes the twist based on dualquaternions
    twist = ca.vertcat(0.0, _w_des_sym[0:3, 0], 0.0, _w_des_sym[3:6, 0])
    w_aux = _get_real_sym(twist)
    w = w_aux[1:4, 0]

    v_aux = _get_dual_sym(twist)
    v = v_aux[1:4, 0]

    quat_aux = _get_quat_sym(_dual_des_sym)

    real = w
    dual = _f_rotation_inverse_sym(quat_aux, v)

    velocity = ca.vertcat(real, dual)
    f_velocity = Function("f_velocity", [_w_des_sym, _dual_des_sym], [velocity])
    return f_velocity


def make_inertial_velocity_from_twist(_w_des_sym=_w_des_sym, _dual_des_sym=_dual_des_sym):
    # Get Real and dual values
    twist = ca.vertcat(0.0, _w_des_sym[0:3, 0], 0.0, _w_des_sym[3:6, 0])

    w_aux = _get_real_sym(twist)
    w = w_aux[1:4, 0]

    dual_aux = _get_dual_sym(twist)
    dual = dual_aux[1:4, 0]

    quat_aux = _get_quat_sym(_dual_des_sym)

    w_body = w
    v_inertial = _f_rotation_sym(quat_aux, dual)

    velocity = ca.vertcat(w_body, v_inertial)
    f_velocity = Function("f_velocity", [_w_des_sym, _dual_des_sym], [velocity])
    return f_velocity


def dualquat_acceleration(dual, omega, u, L):
    # Split Control Actions
    force = u[0, 0]
    torques = u[1:4, 0]

    # System Matrices
    J = ca.DM.zeros(3, 3)
    J[0, 0] = L[1]
    J[1, 1] = L[2]
    J[2, 2] = L[3]
    J_1 = ca.inv(J)
    e3 = ca.DM.zeros(3, 1)
    e3[2, 0] = 1.0
    g = L[4]
    m = L[0]

    # Compute linear and angular velocity from twist velocity
    w = omega[0:3, 0]
    v = omega[3:6, 0]
    p = _get_trans_sym(dual)
    q = _get_quat_sym(dual)
    p = p[1:4, 0]

    # Compute unforced part
    # a = ca.cross(-J_1@w, J@w)
    F_r = -J_1 @ ca.cross(w, J @ w)
    F_d = ca.cross(v, w) - g * (_f_rotation_inverse_sym(q, e3))

    # Compute forced part
    U_r = J_1 @ torques
    U_d = (force / m) @ e3

    T_r = F_r + U_r
    T_d = F_d + U_d
    T = ca.vertcat(T_r, T_d)

    return T


def export_acados_model(params):
    # Constraints variable
    constraint = ca.types.SimpleNamespace()

    # Parameters Model
    L = [params["mass"], params["ixx"], params["iyy"], params["izz"], params["gravity"]]
    print(L)

    # Model section parameters
    model = AcadosModel()
    model.name = params["mav_name"]
    model.z = []

    # States of the system
    qw = ca.MX.sym("qw", 1, 1)
    qx = ca.MX.sym("qx", 1, 1)
    qy = ca.MX.sym("qy", 1, 1)
    qz = ca.MX.sym("qz", 1, 1)

    dw = ca.MX.sym("dw", 1, 1)
    dx = ca.MX.sym("dx", 1, 1)
    dy = ca.MX.sym("dy", 1, 1)
    dz = ca.MX.sym("dz", 1, 1)

    # Defining the desired Velocity using symbolics
    vx = ca.MX.sym("vx", 1, 1)
    vy = ca.MX.sym("vy", 1, 1)
    vz = ca.MX.sym("vz", 1, 1)

    wx = ca.MX.sym("wx", 1, 1)
    wy = ca.MX.sym("wy", 1, 1)
    wz = ca.MX.sym("wz", 1, 1)

    X = ca.vertcat(
        qw,
        qx,
        qy,
        qz,
        dw,
        dx,
        dy,
        dz,
        wx,
        wy,
        wz,
        vx,
        vy,
        vz,
    )
    model.x = X

    # Split States of the system
    twist = X[8:14, 0]
    dualquat = X[0:8, 0]

    # Auxiliary variables implicit function
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

    # Control Actions
    F_ref = ca.MX.sym("F_ref")
    tau_1_ref = ca.MX.sym("tau_1_ref")
    tau_2_ref = ca.MX.sym("tau_2_ref")
    tau_3_ref = ca.MX.sym("tau_3_ref")

    u = ca.vertcat(F_ref, tau_1_ref, tau_2_ref, tau_3_ref)
    model.u = u

    # System Dynamics
    dual_dot = dualquat_kinematics(dualquat, twist)
    twist_dot = dualquat_acceleration(dualquat, twist, u, L)
    f_expl = ca.vertcat(dual_dot, twist_dot)
    f_impl = X_dot - f_expl

    # External parameters
    ref_params = ca.MX.sym("ref_params", params["nmpc"]["nx"] + params["nmpc"]["nu"], 1)
    cost_params = ca.MX.sym(
        "cost_params", params["nmpc"]["nx"] + params["nmpc"]["nx"] + params["nmpc"]["nu"], 1
    )
    model.p = ca.vertcat(ref_params, cost_params)

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.xdot = X_dot

    # Constraint system
    norm_q = ca.norm_2(_get_quat_sym(X[0:8]))
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0
    return (
        model,
        _get_trans_sym,
        _get_quat_sym,
        constraint,
        log_error_dualquat,
        dualquat_mul_conj,
        log_map_dualquat,
        adjoint_map,
        dualquat_conjugate,
        rotate_vector_body_to_inertial,
    )


def make_quadrotor_model(L: list) -> AcadosModel:
    # Dynamics of the quadrotor based on unit quaternions
    # INPUT
    # L                                                          - system parameters(mass, Inertias and gravity)
    # OUTPUT
    # model                                                      - Acados model
    model_name = "quadrotor"
    constraint = ca.types.SimpleNamespace()
    # Defining Desired Frame
    qw = ca.MX.sym("qw", 1, 1)
    qx = ca.MX.sym("qx", 1, 1)
    qy = ca.MX.sym("qy", 1, 1)
    qz = ca.MX.sym("qz", 1, 1)

    dw = ca.MX.sym("dw", 1, 1)
    dx = ca.MX.sym("dx", 1, 1)
    dy = ca.MX.sym("dy", 1, 1)
    dz = ca.MX.sym("dz", 1, 1)

    # Defining the desired Velocity using symbolics
    vx = ca.MX.sym("vx", 1, 1)
    vy = ca.MX.sym("vy", 1, 1)
    vz = ca.MX.sym("vz", 1, 1)

    wx = ca.MX.sym("wx", 1, 1)
    wy = ca.MX.sym("wy", 1, 1)
    wz = ca.MX.sym("wz", 1, 1)

    X = ca.vertcat(
        qw,
        qx,
        qy,
        qz,
        dw,
        dx,
        dy,
        dz,
        wx,
        wy,
        wz,
        vx,
        vy,
        vz,
    )

    # Split States of the system
    twist = X[8:14, 0]
    dualquat = X[0:8, 0]

    # Auxiliary variables implicit function
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

    # Control Actions
    F_ref = ca.MX.sym("F_ref")
    tau_1_ref = ca.MX.sym("tau_1_ref")
    tau_2_ref = ca.MX.sym("tau_2_ref")
    tau_3_ref = ca.MX.sym("tau_3_ref")

    u = ca.vertcat(F_ref, tau_1_ref, tau_2_ref, tau_3_ref)

    dual_dot = dualquat_kinematics(dualquat, twist)
    twist_dot = dualquat_acceleration(dualquat, twist, u, L)

    norm_q = ca.norm_2(_get_quat_sym(X[0:8]))
    dot_real_dual = 2 * ca.dot(X[0:4], X[4:8])
    constraint.norm = Function("norm", [X], [norm_q])
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0
    constraint.min2 = 0.0
    constraint.max2 = 0.0
    # Explicit and implicit functions
    f_expl = ca.vertcat(dual_dot, twist_dot)
    f_impl = X_dot - f_expl
    p = ca.MX.sym("p", 18, 1)

    # Algebraic variables
    z = []

    # Dynamics
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = X_dot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    return (
        model,
        _get_trans_sym,
        _get_quat_sym,
        constraint,
        log_error_dualquat,
        dualquat_mul_conj,
        log_map_dualquat,
        adjoint_map,
        dualquat_conjugate,
        rotate_vector_body_to_inertial,
    )


def apply_noise(x, noise):
    # Get position and quaternion
    dual = x[0:8]
    twist = x[8:14]
    trans = _get_trans_sym(dual)
    trans_np = np.array(trans[1:4]).reshape((3,))
    quat_data = _get_quat_sym(dual)

    # Split noise
    noise_position = noise[0:3]

    # Translation part
    trans_noise = trans_np + noise_position
    trans_noise_aux = np.array([0.0, trans_noise[0], trans_noise[1], trans_noise[2]])

    # Rotational part
    noise_quat = noise[3:6]
    squared_norm_delta = (
        noise_quat[0] * noise_quat[0]
        + noise_quat[1] * noise_quat[1]
        + noise_quat[2] * noise_quat[2]
    )
    q_delta = np.zeros((4, 1))

    if squared_norm_delta > 0:
        norm_delta = np.sqrt(squared_norm_delta)
        sin_delta_by_delta = np.sin(norm_delta) / norm_delta
        q_delta[0, 0] = np.cos(norm_delta)
        q_delta[1, 0] = sin_delta_by_delta * noise_quat[0]
        q_delta[2, 0] = sin_delta_by_delta * noise_quat[1]
        q_delta[3, 0] = sin_delta_by_delta * noise_quat[2]
    else:
        q_delta[0, 0] = 1.0
        q_delta[1, 0] = 0.0
        q_delta[2, 0] = 0.0
        q_delta[3, 0] = 0.0

    H_r_plus = ca.vertcat(
        ca.horzcat(quat_data[0, 0], -quat_data[1, 0], -quat_data[2, 0], -quat_data[3, 0]),
        ca.horzcat(quat_data[1, 0], quat_data[0, 0], -quat_data[3, 0], quat_data[2, 0]),
        ca.horzcat(quat_data[2, 0], quat_data[3, 0], quat_data[0, 0], -quat_data[1, 0]),
        ca.horzcat(quat_data[3, 0], -quat_data[2, 0], quat_data[1, 0], quat_data[0, 0]),
    )
    H_r_plus = np.array(H_r_plus)
    quat_noise_aux = H_r_plus @ q_delta
    Q1_pose = DualQuaternion.from_pose(quat=quat_noise_aux, trans=trans_noise_aux)
    values_pose = np.array(Q1_pose.get[:, 0]).reshape((8,))

    # Twist Noise
    values_twist_aux = noise[6:12]
    values_twist = np.array(twist).reshape((6,)) + values_twist_aux
    x_noise = np.array(
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
    return x_noise
