"""Python port of quadrotor_sim::Se3Controller::Compute().

Implements the geometric SE(3) tracking controller from
Lee, Leok & McClamroch (2010) "Geometric tracking control of a
quadrotor UAV on SE(3)", using pure numpy.
"""

from __future__ import annotations

import numpy as np


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def vee_map(R_err: np.ndarray) -> np.ndarray:
    """vee operator: extract (3,) vector from skew-symmetric 3x3 matrix.

    0.5 * vee(R_err - R_err^T) extracts the SO(3) attitude error.
    """
    return np.array(
        [
            0.5 * (R_err[2, 1] - R_err[1, 2]),
            0.5 * (R_err[0, 2] - R_err[2, 0]),
            0.5 * (R_err[1, 0] - R_err[0, 1]),
        ],
        dtype=np.float64,
    )


def se3_control(
    pos_world: np.ndarray,
    quat_wxyz: np.ndarray,
    lin_vel_body: np.ndarray,
    ang_vel_body: np.ndarray,
    target_pos: np.ndarray,
    target_yaw: float,
    K_p: np.ndarray,
    K_v: np.ndarray,
    K_R: np.ndarray,
    K_w: np.ndarray,
    mass: float,
    gravity: float = 9.80665,
    target_vel: np.ndarray | None = None,
    max_acc_xy: float = 10.0,
) -> tuple[float, float, float, float]:
    """Compute body-frame thrust and torques via SE(3) geometric control.

    @param[in] pos_world     World ENU position [m]        shape (3,)
    @param[in] quat_wxyz     World ENU orientation (w,x,y,z) shape (4,)
    @param[in] lin_vel_body  Body FLU linear velocity [m/s]  shape (3,)
    @param[in] ang_vel_body  Body FLU angular velocity [rad/s] shape (3,)
    @param[in] target_pos    Desired world ENU position [m]  shape (3,)
    @param[in] target_yaw    Desired yaw angle [rad]
    @param[in] K_p           Position gains [N/m]            shape (3,)
    @param[in] K_v           Velocity gains [N·s/m]          shape (3,)
    @param[in] K_R           Attitude gains [Nm]             shape (3,)
    @param[in] K_w           Rate damping gains [Nm·s/rad]   shape (3,)
    @param[in] mass          Vehicle mass [kg]
    @param[in] gravity       Gravitational acceleration [m/s²]
    @param[in] target_vel    Desired world ENU velocity [m/s] shape (3,)
    @param[in] max_acc_xy    Maximum horizontal acceleration [m/s²]
    @return (thrust, tau_x, tau_y, tau_z) body FLU [N, Nm]
    """
    qw, qx, qy, qz = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    R = quat_to_rotmat(qw, qx, qy, qz)

    if target_vel is None:
        target_vel = np.zeros(3, dtype=np.float64)
    else:
        target_vel = np.asarray(target_vel, dtype=np.float64).ravel()

    e_p = pos_world - target_pos

    v_world = R @ lin_vel_body
    e_v = v_world - target_vel

    F_des = np.zeros(3, dtype=np.float64)
    for i in range(3):
        F_des[i] = -K_p[i] * e_p[i] - K_v[i] * e_v[i]
    F_des[2] -= mass * (-gravity)  # compensate gravity: -m * g_z where g_z = -gravity

    F_xy_norm = float(np.linalg.norm(F_des[:2]))
    max_force_xy = mass * max_acc_xy
    if F_xy_norm > max_force_xy:
        F_des[:2] *= max_force_xy / F_xy_norm

    thrust_raw = np.dot(F_des, R[:, 2])
    thrust = float(np.clip(thrust_raw, 0.0, 21.0))

    z_b = F_des / np.linalg.norm(F_des)
    x_c = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0], dtype=np.float64)
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)

    R_des = np.column_stack((x_b, y_b, z_b))

    R_err = R_des.T @ R
    D = R_err - R_err.T
    e_R = np.array([0.5 * D[2, 1], 0.5 * D[0, 2], 0.5 * D[1, 0]], dtype=np.float64)

    tau_x = float(np.clip(-K_R[0] * e_R[0] - K_w[0] * ang_vel_body[0], -0.5, 0.5))
    tau_y = float(np.clip(-K_R[1] * e_R[1] - K_w[1] * ang_vel_body[1], -0.5, 0.5))
    tau_z = float(np.clip(-K_R[2] * e_R[2] - K_w[2] * ang_vel_body[2], -0.5, 0.5))

    return thrust, tau_x, tau_y, tau_z
