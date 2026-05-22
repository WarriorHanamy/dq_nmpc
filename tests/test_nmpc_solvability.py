"""NMPC solver solvability tests — no MuJoCo simulator or SHM required.

Acados-dependent; run with: ``uv run pytest -v -m acados``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dq_nmpc.math.dq_functions import dualquat_from_pose_np, make_inertial_to_body_rotation
from dq_nmpc.minco_trajectory.flatness_casadi import make_flatness_casadi
from dq_nmpc.minco_trajectory.loader import load_trajectory_npz, reinterpret_minco_trajectory
from dq_nmpc.nmpc.ocp_setup import solver
from dq_nmpc.schema import (
    NMPC_REF_DIM,
    NMPC_REF_DQ_SLICE,
    NMPC_REF_UNOM_SLICE,
    NMPCConfig,
    control_index,
)

NMPC_YAML = "src/dq_nmpc/config/mujoco/default/nmpc.yaml"
TRAJ_NPZ = "out/circle/trajectory.npz"


def _build_ref_params(traj, k: int) -> np.ndarray:
    pos_k = traj.ref_pos[k].ravel()
    quat_k = traj.ref_quat[k].ravel()
    omega_k = traj.ref_omega[k].ravel()
    vel_world = traj.ref_vel[k].ravel()
    thrust_k = float(traj.ref_thrust[k])
    torque_k = traj.ref_torque[k].ravel()

    dq_vec = dualquat_from_pose_np(quat_k, pos_k)

    inv_rot = make_inertial_to_body_rotation()
    vel_body = np.array(inv_rot(quat_k.reshape((4, 1)), vel_world.reshape((3, 1)))).ravel()

    u_nom = np.zeros(4, dtype=np.float64)
    u_nom[control_index("thrust")] = thrust_k
    u_nom[control_index("tau_x")] = torque_k[0]
    u_nom[control_index("tau_y")] = torque_k[1]
    u_nom[control_index("tau_z")] = torque_k[2]
    return np.concatenate([dq_vec, omega_k, vel_body, u_nom])


def _load_trajectory():
    config = NMPCConfig.from_yaml(NMPC_YAML)
    traj7 = load_trajectory_npz(TRAJ_NPZ)
    traj = reinterpret_minco_trajectory(traj7, config, config.ocp.control_update_interval)
    return config, traj


@pytest.mark.acados
def test_nmpc_ref_params_shape():
    config, traj = _load_trajectory()
    p = _build_ref_params(traj, 0)
    assert p.shape == (NMPC_REF_DIM,), f"Expected ({NMPC_REF_DIM},) got {p.shape}"
    assert np.isfinite(p).all()
    dq_part = p[NMPC_REF_DQ_SLICE]
    q_real = dq_part[0:4]
    assert abs(float(np.linalg.norm(q_real)) - 1.0) < 1e-10


@pytest.mark.acados
def test_nmpc_first_step_solvable():
    config, traj = _load_trajectory()
    ocp_cfg = config.ocp
    inv_rot = make_inertial_to_body_rotation()

    k = 0
    pos = traj.ref_pos[k]
    quat_q = traj.ref_quat[k].ravel()
    dq_vec = dualquat_from_pose_np(quat_q, pos)
    v_body = np.array(
        inv_rot(quat_q.reshape((4, 1)), traj.ref_vel[k].ravel().reshape((3, 1)))
    ).ravel()
    x0 = np.concatenate([dq_vec, traj.ref_omega[k].ravel(), v_body])

    acados_solver, ocp = solver(config, codegen=False)

    N_horizon = ocp_cfg.horizon_steps
    for i in range(N_horizon):
        idx = min(i, len(traj.t) - 1)
        p_ref = _build_ref_params(traj, idx)
        acados_solver.set(i, "p", p_ref)
        acados_solver.set(i, "u", p_ref[NMPC_REF_UNOM_SLICE].copy())

    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    status = acados_solver.solve()
    residuals = acados_solver.get_stats("residuals")
    qp_iter = acados_solver.get_stats("qp_iter")
    qp_stat = acados_solver.get_stats("qp_stat")

    assert status == 0, (
        f"Solver failed (status={status}). "
        f"residuals={residuals} qp_iter={qp_iter} qp_stat={qp_stat}"
    )

    u0 = acados_solver.get(0, "u")
    assert np.isfinite(u0).all(), f"Control output non-finite: {u0}"
    assert float(u0[0]) > 0, f"Thrust must be positive, got {u0[0]:.3f}"


@pytest.mark.acados
def test_nmpc_cost_function_not_nan():
    config, traj = _load_trajectory()
    m = make_flatness_casadi()

    acc = traj.ref_acc[0]
    jerk = traj.ref_jerk[0]
    snap = traj.ref_snap[0]
    yaw = float(traj.ref_yaw[0])
    yaw_dot = float(traj.ref_yaw_dot[0])
    yaw_ddot = float(traj.ref_yaw_ddot[0])

    result = m(
        float(acc[0]),
        float(acc[1]),
        float(acc[2]),
        float(jerk[0]),
        float(jerk[1]),
        float(jerk[2]),
        float(snap[0]),
        float(snap[1]),
        float(snap[2]),
        yaw,
        yaw_dot,
        yaw_ddot,
        config.physics.mass,
        config.physics.ixx,
        config.physics.iyy,
        config.physics.izz,
        config.physics.gravity,
    )
    assert all(np.isfinite(float(r)) for r in result), f"Non-finite flatness output: {result}"
