"""NMPC solver solvability tests — no MuJoCo simulator or SHM required.

Acados-dependent; run with: ``uv run pytest -v -m acados``.
"""

from __future__ import annotations

import numpy as np
import pytest
from minco.trajectory import load_npz

from dq_nmpc.nmpc import dense_ref_from_minco, make_flatness_casadi, solver
from dq_nmpc.schema import (
    NMPC_REF_DIM,
    NMPC_REF_DQ_SLICE,
    NMPC_REF_UNOM_SLICE,
    NMPCConfig,
)

NMPC_YAML = "src/dq_nmpc/nmpc/config/default.yaml"
TRAJ_NPZ = "out/circle/trajectory.npz"


def _load_trajectory():
    config = NMPCConfig.from_yaml(NMPC_YAML)
    traj7 = load_npz(TRAJ_NPZ)
    ref_params = dense_ref_from_minco(traj7, config)
    return config, ref_params


@pytest.mark.acados
def test_nmpc_ref_params_shape():
    config, ref_params = _load_trajectory()
    p = ref_params[0]
    assert p.shape == (NMPC_REF_DIM,), f"Expected ({NMPC_REF_DIM},) got {p.shape}"
    assert np.isfinite(p).all()
    dq_part = p[NMPC_REF_DQ_SLICE]
    q_real = dq_part[0:4]
    assert abs(float(np.linalg.norm(q_real)) - 1.0) < 1e-10


@pytest.mark.acados
def test_nmpc_first_step_solvable():
    config, ref_params = _load_trajectory()
    ocp_cfg = config.ocp

    k = 0
    x0_dq = ref_params[k, 0:8]
    x0_omega = ref_params[k, 8:11]
    x0_vel_body = ref_params[k, 11:14]
    x0 = np.concatenate([x0_dq, x0_omega, x0_vel_body])

    acados_solver, ocp = solver(config, codegen=False)

    N_horizon = ocp_cfg.horizon_steps
    for i in range(N_horizon):
        idx = min(i, len(ref_params) - 1)
        acados_solver.set(i, "p", ref_params[idx])
        acados_solver.set(i, "u", ref_params[idx, NMPC_REF_UNOM_SLICE].copy())

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
    config, ref_params = _load_trajectory()

    traj7 = load_npz(TRAJ_NPZ)
    m = make_flatness_casadi()
    t = 0.0
    acc = np.array(traj7.get_acc(t)).ravel()
    jerk = np.array(traj7.get_jer(t)).ravel()
    snap = np.array(traj7.get_sna(t)).ravel()

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
        0.0,
        0.0,
        0.0,
        config.physics.mass,
        config.physics.ixx,
        config.physics.iyy,
        config.physics.izz,
        config.physics.gravity,
    )
    assert all(np.isfinite(float(r)) for r in result), f"Non-finite flatness output: {result}"
