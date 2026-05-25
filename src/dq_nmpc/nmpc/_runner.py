"""SHM-based NMPC runtime loop.

Phase 1a: SE(3) geometric controller converges to takeoff point (0, 0, 1.5).
Phase 1b: SE(3) controller converges to the first trajectory point.
Phase 2:  acados NMPC tracks the full trajectory via online re-planning.

All phases are recorded via DroneVisualizer (Rerun).

Trajectory source: minco NPZ → RefTrajectoryAsBelts via nmpc.reference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from dq_nmpc.nmpc._dq_functions import (
    dualquat_from_pose_ca_func,
    position_from_dualquat_ca_func,
    yaw_from_dualquat_ca_func,
)
from dq_nmpc.nmpc._drone_visualizer import DroneVisualizer
from dq_nmpc.nmpc._se3_controller import se3_control
from dq_nmpc.schema import (
    NMPC_REF_UNOM_SLICE,
    NMPCConfig,
    OutputPaths,
    Se3Config,
    control_index,
)

try:
    from quadrotor_sim.shm import (
        SHM_CTRL_FILE,
        SHM_STATE_FILE,
        QuadrotorControlC,
        QuadrotorStateC,
        ShmReader,
        ShmWriter,
    )

    SHM_AVAILABLE = True
except ImportError:
    SHM_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_nmpc(
    config_path: str | Path,
    trajectory_path: str | Path,
    se3_config_path: str | Path | None = None,
    max_iter: int = 0,
    rerun: bool = False,
):
    if not SHM_AVAILABLE:
        raise RuntimeError("quadrotor_sim.shm not available; ensure quadrotor_sim is on PYTHONPATH")

    config = NMPCConfig.from_yaml(config_path)
    logger.info("Config loaded: %s", config_path)

    if se3_config_path is not None:
        se3_config = Se3Config.from_yaml(se3_config_path)
        logger.info("SE3 config loaded: %s", se3_config_path)
    else:
        se3_config = Se3Config()
        logger.info("SE3 config: defaults")

    mass = config.physics.mass
    gravity = config.physics.gravity
    control_dt = config.ocp.control_update_interval
    N_horizon = config.ocp.horizon_steps
    horizon_time = config.ocp.horizon_time

    pos_fn = position_from_dualquat_ca_func()
    yaw_fn = yaw_from_dualquat_ca_func()

    data = np.load(trajectory_path)
    if "durations" not in data:
        raise ValueError(
            f"Unrecognized NPZ format: keys={list(data.files)}. "
            "Expected minco NPZ ('durations' + 'coeffs')."
        )

    from minco.trajectory import load_npz

    from dq_nmpc.nmpc._reference import belts_from_dense, dense_ref_from_minco

    traj7 = load_npz(trajectory_path)
    ref_params = dense_ref_from_minco(traj7, config)
    belts = belts_from_dense(ref_params, N_horizon)
    traj_duration = float(traj7.total_duration)
    logger.info(
        "Trajectory loaded (minco): N_c=%d, duration=%.2f s, %d pieces",
        belts.N_c,
        traj_duration,
        len(traj7),
    )

    first_tp = belts[0][0]
    first_pos = np.array(pos_fn(first_tp.dq.reshape(8, 1))).ravel()
    first_yaw = float(yaw_fn(first_tp.dq.reshape(8, 1)))

    logger.info(
        "First trajectory point: (%.3f, %.3f, %.3f), yaw=%.3f rad",
        first_pos[0],
        first_pos[1],
        first_pos[2],
        first_yaw,
    )
    logger.info(
        "Trajectory duration: %.2f s, N_c=%d belts",
        traj_duration,
        belts.N_c,
    )

    rrd_path = str(OutputPaths().se3_rrd)
    viz = DroneVisualizer(rrd_path, spawn=rerun)
    viz.log_static_trajectory(belts)
    viz.log_static_markers(
        takeoff=(0.0, 0.0, 1.5),
        first_traj=(float(first_pos[0]), float(first_pos[1]), float(first_pos[2])),
    )
    logger.info("Rerun recording to %s", rrd_path)

    state_reader = ShmReader(SHM_STATE_FILE, QuadrotorStateC, 192)
    ctrl_writer = ShmWriter(SHM_CTRL_FILE, QuadrotorControlC, 64)
    state_buf = QuadrotorStateC()

    max_attach_wait = 5.0
    attach_start = time.time()
    attached = False
    while time.time() - attach_start < max_attach_wait:
        try:
            state_reader.attach()
            attached = True
            break
        except FileNotFoundError:
            time.sleep(0.1)
    if not attached:
        raise RuntimeError(f"SHM state file {SHM_STATE_FILE} not found; is sim_core running?")

    try:
        ctrl_writer.open()
    except FileNotFoundError:
        ctrl_writer.create()

    K_p = np.array(se3_config.K_p, dtype=np.float64)
    K_v = np.array(se3_config.K_v, dtype=np.float64)
    K_R = np.array(se3_config.K_R, dtype=np.float64)
    K_w = np.array(se3_config.K_w, dtype=np.float64)
    convergence_threshold = 0.05

    dt_ns = int(control_dt * 1e9)
    next_wake = time.monotonic_ns()

    def _se3_converge(target_pos: np.ndarray, target_yaw: float, target_label: str) -> bool:
        """Run SE3 control loop until convergence or KeyboardInterrupt."""
        nonlocal next_wake
        logger.info(
            "SE3 → %s (%.2f, %.2f, %.2f), yaw=%.2f",
            target_label,
            target_pos[0],
            target_pos[1],
            target_pos[2],
            target_yaw,
        )
        viz.log_target(target_pos)

        step = 0
        while True:
            while not state_reader.read(state_buf):
                time.sleep(0.0001)

            pos = np.array(state_buf.position[:], dtype=np.float64)
            quat_wxyz = np.array(state_buf.orientation[:], dtype=np.float64)
            lin_vel_body = np.array(state_buf.linear_velocity[:], dtype=np.float64)
            ang_vel_body = np.array(state_buf.angular_velocity[:], dtype=np.float64)

            position_error = float(np.linalg.norm(pos - target_pos))
            vel_norm = float(np.linalg.norm(lin_vel_body))

            thrust, tau_x, tau_y, tau_z = se3_control(
                pos,
                quat_wxyz,
                lin_vel_body,
                ang_vel_body,
                target_pos,
                target_yaw,
                K_p,
                K_v,
                K_R,
                K_w,
                mass,
                gravity,
                np.zeros(3, dtype=np.float64),
            )

            viz.log_drone(
                pos,
                quat_wxyz,
                state_buf.time,
                error=position_error,
                thrust=thrust,
                tau_x=tau_x,
                tau_y=tau_y,
                tau_z=tau_z,
            )

            if step % 50 == 0:
                logger.info(
                    "  step=%d  pos=(%.2f, %.2f, %.2f)  err=%.3f  vel=%.2f  thrust=%.2f  yaw=%.2f",
                    step,
                    pos[0],
                    pos[1],
                    pos[2],
                    position_error,
                    vel_norm,
                    thrust,
                    target_yaw,
                )

            if position_error < convergence_threshold and vel_norm < 0.1:
                logger.info(
                    "SE3 converged to %s: pos_error=%.4f m  vel=%.3f m/s",
                    target_label,
                    position_error,
                    vel_norm,
                )
                return True

            ctrl_writer.write_control(
                thrust=thrust,
                torque_x=tau_x,
                torque_y=tau_y,
                torque_z=tau_z,
            )

            now_ns = time.monotonic_ns()
            if now_ns < next_wake:
                time.sleep((next_wake - now_ns) / 1e9)
            else:
                next_wake = now_ns
            next_wake += dt_ns
            step += 1

    try:
        takeoff_target = np.array([0.0, 0.0, 1.5], dtype=np.float64)
        if not _se3_converge(takeoff_target, first_yaw, "takeoff"):
            ctrl_writer.detach()
            state_reader.detach()
            return

        if not _se3_converge(first_pos, first_yaw, "trajectory"):
            ctrl_writer.detach()
            state_reader.detach()
            return

    except KeyboardInterrupt:
        logger.info("SE3 bootstrap stopped by user")
        ctrl_writer.detach()
        state_reader.detach()
        return

    # -- Phase 2: NMPC trajectory tracking --

    from dq_nmpc.nmpc._ocp_setup import solver as create_solver

    dq_from_pose = dualquat_from_pose_ca_func()

    def _shm_to_solver_x0(buf) -> np.ndarray:
        pos = np.array(buf.position[:], dtype=np.float64).ravel()
        quat_q = np.array(buf.orientation[:], dtype=np.float64).ravel()
        ang_vel = np.array(buf.angular_velocity[:], dtype=np.float64).ravel()
        lin_vel = np.array(buf.linear_velocity[:], dtype=np.float64).ravel()

        dq_mx = dq_from_pose(quat_q[0], quat_q[1], quat_q[2], quat_q[3], pos[0], pos[1], pos[2])
        dq_vec = np.array(dq_mx, dtype=np.float64).ravel()

        twist = np.concatenate([ang_vel, lin_vel])
        return np.concatenate([dq_vec, twist])

    logger.info("Loading acados solver from c_generated_code/ …")
    solver, _ocp = create_solver(config, codegen=False)
    logger.info("acados solver loaded (horizon=%d steps, %.2f s)", N_horizon, horizon_time)

    N_c = belts.N_c

    while not state_reader.read(state_buf):
        time.sleep(0.0001)
    x0_init = _shm_to_solver_x0(state_buf)

    for i in range(N_horizon):
        solver.set(i, "x", x0_init.copy())
        solver.set(i, "p", belts[0].points[i])
        solver.set(i, "u", belts[0].points[i, NMPC_REF_UNOM_SLICE].copy())

    MAX_INIT_SQP = 10
    INIT_SQP_TOL = 1e-3
    logger.info("=== NMPC-INIT warm-start (k=0, max %d SQP-RTI) ===", MAX_INIT_SQP)
    converged = False
    residuals = float("inf")
    for init_iter in range(MAX_INIT_SQP):
        init_start = time.monotonic()
        status = solver.solve()
        solve_ms = (time.monotonic() - init_start) * 1000.0
        residuals_raw = solver.get_stats("residuals")
        qp_iter_raw = solver.get_stats("qp_iter")
        residuals = float(np.max(np.atleast_1d(residuals_raw)))
        qp_iter = int(np.sum(np.atleast_1d(qp_iter_raw)))
        logger.info(
            "[NMPC-INIT] iter=%d/%d  solve=%.1f ms  residuals=%.2e  qp_iter=%d  status=%d",
            init_iter + 1,
            MAX_INIT_SQP,
            solve_ms,
            residuals,
            qp_iter,
            status,
        )
        if status != 0:
            logger.error("[NMPC-INIT] failed at iter %d (status=%d)", init_iter + 1, status)
            ctrl_writer.detach()
            state_reader.detach()
            return
        if residuals < INIT_SQP_TOL:
            logger.info("[NMPC-INIT] converged — residuals=%.2e < %.2e", residuals, INIT_SQP_TOL)
            converged = True
            break
    if not converged:
        logger.warning(
            "[NMPC-INIT] did not converge within %d iterations (final residuals=%.2e)",
            MAX_INIT_SQP,
            residuals,
        )

    dt_ns = int(control_dt * 1e9)
    next_wake = time.monotonic_ns()
    k = 0
    end_time = traj_duration + horizon_time
    logger.info(
        "=== NMPC-REALTIME tracking %.2f s trajectory (horizon=%.2f s) ===",
        traj_duration,
        horizon_time,
    )

    try:
        while k * control_dt < end_time:
            while not state_reader.read(state_buf):
                time.sleep(0.0001)

            x0 = _shm_to_solver_x0(state_buf)
            solver.set(0, "lbx", x0)
            solver.set(0, "ubx", x0)

            idx_belt = min(k, N_c - 1)
            belt_k = belts[idx_belt]
            for i in range(N_horizon):
                solver.set(i, "p", belt_k.points[i])

            solve_start = time.monotonic()
            status = solver.solve()
            solve_ms = (time.monotonic() - solve_start) * 1000.0

            if status != 0:
                logger.error(
                    "NMPC solver failed at step %d (status=%d, solve=%.1f ms), terminating.",
                    k,
                    status,
                    solve_ms,
                )
                break

            u_opt = solver.get(0, "u")
            pos = np.array(state_buf.position[:], dtype=np.float64)
            quat_wxyz = np.array(state_buf.orientation[:], dtype=np.float64)

            ctrl_writer.write_control(
                thrust=float(u_opt[control_index("thrust")]),
                torque_x=float(u_opt[control_index("tau_x")]),
                torque_y=float(u_opt[control_index("tau_y")]),
                torque_z=float(u_opt[control_index("tau_z")]),
            )

            tp_k = belt_k[0]
            target_pos = np.array(pos_fn(tp_k.dq.reshape(8, 1))).ravel()
            position_error = float(np.linalg.norm(pos - target_pos))

            viz.log_drone(
                pos,
                quat_wxyz,
                state_buf.time,
                error=position_error,
                thrust=float(u_opt[control_index("thrust")]),
                tau_x=float(u_opt[control_index("tau_x")]),
                tau_y=float(u_opt[control_index("tau_y")]),
                tau_z=float(u_opt[control_index("tau_z")]),
            )

            viz.log_nmpc_reference(target_pos)

            dq_batch = belt_k.points[:, :8].T  # (8, N)
            horizon_pos = np.array(pos_fn(dq_batch)).T  # (N, 3)
            horizon_pts = [
                (float(horizon_pos[i, 0]), float(horizon_pos[i, 1]), float(horizon_pos[i, 2]))
                for i in range(N_horizon)
            ]
            viz.log_nmpc_horizon(horizon_pts)
            viz.log_nmpc_stats(
                solve_ms=solve_ms,
                residuals=solver.get_stats("residuals"),
                qp_iter=solver.get_stats("qp_iter"),
                qp_stat=solver.get_stats("qp_stat"),
                ref_thrust=float(tp_k.u_nom[0]),
                pos_err_xyz=pos - target_pos,
            )

            if k % 50 == 0:
                logger.info(
                    "[NMPC-REALTIME] step=%d  pos=(%.2f,%.2f,%.2f)  err=%.3f  th=%.2f  "
                    "tx=%.4f ty=%.4f tz=%.4f  solve=%.1f ms",
                    k,
                    pos[0],
                    pos[1],
                    pos[2],
                    position_error,
                    u_opt[control_index("thrust")],
                    u_opt[control_index("tau_x")],
                    u_opt[control_index("tau_y")],
                    u_opt[control_index("tau_z")],
                    solve_ms,
                )
            else:
                logger.info(
                    "[NMPC-REALTIME] step=%d  err=%.3f  solve=%.1f ms  status=%d",
                    k,
                    position_error,
                    solve_ms,
                    status,
                )

            now_ns = time.monotonic_ns()
            if now_ns < next_wake:
                time.sleep((next_wake - now_ns) / 1e9)
            else:
                next_wake = now_ns
            next_wake += dt_ns
            k += 1

    except KeyboardInterrupt:
        logger.info("NMPC stopped by user at step %d", k)

    ctrl_writer.detach()
    state_reader.detach()
    logger.info("NMPC loop terminated after %d steps", k)
