"""SHM-based NMPC runtime loop.

Phase 1a: SE(3) geometric controller converges to takeoff point (0, 0, 1.5).
Phase 1b: SE(3) controller converges to the first trajectory point.
Phase 2:  acados NMPC tracks the full trajectory via online re-planning.

All phases are recorded via DroneVisualizer (Rerun).

Default trajectory: flatness-based planner (get_flatness_trajectory).
Pass a path to use a pre-generated NPZ (minco or planner format).
Minco NPZ paths are reinterpreted through the full flatness decomposition
(CasADi-compiled) before entering the NMPC loop.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from dq_nmpc.nmpc.drone_visualizer import DroneVisualizer
from dq_nmpc.nmpc.se3_controller import se3_control
from dq_nmpc.schema import (
    NMPC_REF_UNOM_SLICE,
    FlatnessTrajectory,
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
    trajectory_path: str | Path | None = None,
    se3_config_path: str | Path | None = None,
    max_iter: int = 0,
    rerun: bool = False,
    planner: bool = False,
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

    if planner:
        from dq_nmpc.nmpc.planner import get_flatness_trajectory

        ref = get_flatness_trajectory(
            params=config,
            initial_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            t_initial=2.0,
            t_trajectory=10.0,
            t_final=2.0,
            sample_time=control_dt,
            radius=2.0,
            angular_speed=0.5,
        )
        traj = FlatnessTrajectory(
            ref_pos=ref[0],
            ref_vel=ref[1],
            ref_acc=ref[2],
            ref_jerk=ref[3],
            ref_snap=ref[4],
            ref_quat=ref[5],
            ref_omega=ref[6],
            ref_omega_dot=ref[7],
            ref_thrust=ref[8],
            ref_torque=ref[9],
            ref_yaw=ref[10],
            ref_yaw_dot=ref[11],
            ref_yaw_ddot=ref[12],
            t=ref[13],
        )
        first_pos = traj.ref_pos[0].astype(np.float64).ravel()
        viz_traj = traj
        logger.info(
            "Trajectory generated (planner): N=%d, duration=%.2f s",
            len(traj.t),
            traj.t[-1],
        )
    elif trajectory_path is not None:
        data = np.load(trajectory_path)
        if "durations" in data:
            from dq_nmpc.minco_trajectory.loader import (
                load_trajectory_npz,
                reinterpret_minco_trajectory,
            )

            traj7 = load_trajectory_npz(trajectory_path)
            traj = reinterpret_minco_trajectory(traj7, config, control_dt)
            first_pos = traj.ref_pos[0].astype(np.float64).ravel()
            viz_traj = traj
            logger.info(
                "Trajectory loaded (minco → flatness): N=%d, duration=%.2f s, %d pieces",
                len(traj.t),
                traj.t[-1],
                len(traj7),
            )
        elif "ref_pos" in data:
            traj = FlatnessTrajectory.load_npz(trajectory_path)
            first_pos = traj.ref_pos[0].astype(np.float64).ravel()
            viz_traj = traj
            logger.info(
                "Trajectory loaded (planner NPZ): N=%d, duration=%.2f s",
                len(traj.t),
                traj.t[-1],
            )
        else:
            raise ValueError(
                f"Unrecognized NPZ format: keys={list(data.files)}. "
                "Expected 'durations' (minco) or 'ref_pos' (planner)."
            )
    else:
        raise ValueError("trajectory_path is required when planner=False")

    logger.info(
        "First trajectory point: (%.3f, %.3f, %.3f)",
        first_pos[0],
        first_pos[1],
        first_pos[2],
    )
    logger.info(
        "Trajectory yaw range: [%.2f, %.2f] rad, z range: [%.2f, %.2f] m",
        traj.ref_yaw.min(),
        traj.ref_yaw.max(),
        traj.ref_pos[:, 2].min(),
        traj.ref_pos[:, 2].max(),
    )

    rrd_path = str(OutputPaths().se3_rrd)
    viz = DroneVisualizer(rrd_path, spawn=rerun)
    viz.log_static_trajectory(viz_traj)
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

    def _se3_converge(target_pos: np.ndarray, target_label: str) -> bool:
        """Run SE3 control loop until convergence or KeyboardInterrupt.
        Returns True if converged, False if interrupted."""
        nonlocal next_wake
        logger.info(
            "SE3 → %s (%.2f, %.2f, %.2f)",
            target_label,
            target_pos[0],
            target_pos[1],
            target_pos[2],
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

            target_yaw = traj.interp_yaw(state_buf.time)
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
                remaining_ns = next_wake - now_ns
                time.sleep(remaining_ns / 1e9)
            else:
                next_wake = now_ns

            next_wake += dt_ns
            step += 1

    try:
        takeoff_target = np.array([0.0, 0.0, 1.5], dtype=np.float64)
        if not _se3_converge(takeoff_target, "takeoff"):
            ctrl_writer.detach()
            state_reader.detach()
            return

        if not _se3_converge(first_pos, "trajectory"):
            ctrl_writer.detach()
            state_reader.detach()
            return

    except KeyboardInterrupt:
        logger.info("SE3 bootstrap stopped by user")
        ctrl_writer.detach()
        state_reader.detach()
        return

    # -- Phase 2: NMPC trajectory tracking --

    from dq_nmpc.math.dq_functions import (
        dualquat_from_pose_ca_func,
        inertial_to_body_rotation_ca_func,
    )
    from dq_nmpc.nmpc.ocp_setup import solver as create_solver

    inv_rot = inertial_to_body_rotation_ca_func()
    dq_from_pose = dualquat_from_pose_ca_func()

    N_horizon = config.ocp.horizon_steps
    horizon_time = config.ocp.horizon_time
    Tsim = horizon_time / N_horizon

    def _shm_to_solver_x0(buf) -> np.ndarray:
        pos = np.array(buf.position[:], dtype=np.float64).ravel()
        quat_q = np.array(buf.orientation[:], dtype=np.float64).ravel()
        ang_vel = np.array(buf.angular_velocity[:], dtype=np.float64).ravel()
        lin_vel = np.array(buf.linear_velocity[:], dtype=np.float64).ravel()

        dq_mx = dq_from_pose(quat_q[0], quat_q[1], quat_q[2], quat_q[3], pos[0], pos[1], pos[2])
        dq_vec = np.array(dq_mx, dtype=np.float64).ravel()

        twist = np.concatenate([ang_vel, lin_vel])
        return np.concatenate([dq_vec, twist])

    def _traj_step_to_ref_params(k: int) -> np.ndarray:
        pos_k = traj.ref_pos[k].ravel()
        quat_k = traj.ref_quat[k].ravel()
        omega_k = traj.ref_omega[k].ravel()
        vel_world = traj.ref_vel[k].ravel()
        thrust_k = float(traj.ref_thrust[k])
        torque_k = traj.ref_torque[k].ravel()

        dq_mx = dq_from_pose(
            quat_k[0], quat_k[1], quat_k[2], quat_k[3], pos_k[0], pos_k[1], pos_k[2]
        )
        dq_vec = np.array(dq_mx, dtype=np.float64).ravel()

        vel_body = np.array(inv_rot(quat_k.reshape((4, 1)), vel_world.reshape((3, 1)))).ravel()

        u_nom = np.zeros(4, dtype=np.float64)
        u_nom[control_index("thrust")] = thrust_k
        u_nom[control_index("tau_x")] = torque_k[0]
        u_nom[control_index("tau_y")] = torque_k[1]
        u_nom[control_index("tau_z")] = torque_k[2]
        return np.concatenate([dq_vec, omega_k, vel_body, u_nom])

    logger.info("Loading acados solver from c_generated_code/ …")
    solver, _ocp = create_solver(config, codegen=False)
    logger.info(
        "acados solver loaded (horizon=%d steps, %.2f s)", N_horizon, config.ocp.horizon_time
    )

    traj_len = len(traj.t)

    while not state_reader.read(state_buf):
        time.sleep(0.0001)
    x0_init = _shm_to_solver_x0(state_buf)

    for i in range(N_horizon):
        solver.set(i, "x", x0_init.copy())
        idx = min(i, traj_len - 1)
        p_ref = _traj_step_to_ref_params(idx)
        solver.set(i, "p", p_ref)
        solver.set(i, "u", p_ref[NMPC_REF_UNOM_SLICE].copy())

    dt_ns = int(control_dt * 1e9)
    next_wake = time.monotonic_ns()
    k = 0
    traj_duration = float(traj.t[-1])
    end_time = traj_duration + horizon_time
    logger.info(
        "NMPC loop start — tracking %.2f s trajectory (horizon=%.2f s)", traj_duration, horizon_time
    )

    try:
        while k * control_dt < end_time:
            while not state_reader.read(state_buf):
                time.sleep(0.0001)

            x0 = _shm_to_solver_x0(state_buf)
            solver.set(0, "lbx", x0)
            solver.set(0, "ubx", x0)

            for i in range(N_horizon):
                t_shoot = k * control_dt + i * Tsim
                idx = int(t_shoot / control_dt)
                idx = min(idx, traj_len - 1)
                p_ref = _traj_step_to_ref_params(idx)
                solver.set(i, "p", p_ref)

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

            idx_k = min(k, traj_len - 1)
            target_pos = traj.ref_pos[idx_k]
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
            horizon_pts = [
                tuple(
                    traj.ref_pos[min(int((k * control_dt + i * Tsim) / control_dt), traj_len - 1)]
                )
                for i in range(N_horizon)
            ]
            viz.log_nmpc_horizon(horizon_pts)
            viz.log_nmpc_stats(
                solve_ms=solve_ms,
                residuals=solver.get_stats("residuals"),
                qp_iter=solver.get_stats("qp_iter"),
                qp_stat=solver.get_stats("qp_stat"),
                ref_thrust=float(traj.ref_thrust[idx_k]),
                pos_err_xyz=pos - target_pos,
            )

            if k % 50 == 0:
                logger.info(
                    "NMPC step=%d  pos=(%.2f,%.2f,%.2f)  err=%.3f  th=%.2f  "
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 3:
        print(
            "Usage: dq-nmpc-runner <nmpc.yaml> <trajectory.npz>"
            " [--se3-config SE3.yaml] [--max-iter N]"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    trajectory_path = sys.argv[2]
    se3_config_path = None
    max_iter = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--se3-config" and i + 1 < len(sys.argv):
            se3_config_path = sys.argv[i + 1]
        if arg == "--max-iter" and i + 1 < len(sys.argv):
            max_iter = int(sys.argv[i + 1])

    run_nmpc(
        config_path,
        trajectory_path,
        se3_config_path=se3_config_path,
        max_iter=max_iter,
    )


if __name__ == "__main__":
    main()
