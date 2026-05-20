"""SHM-based NMPC runtime loop.

Phase 1a: SE(3) geometric controller converges to takeoff point (0, 0, 1.5).
Phase 1b: SE(3) controller converges to the first trajectory point.
Phase 2:  Transition to NMPC trajectory tracking.

All SE3 segments are recorded via DroneVisualizer (Rerun).

Default trajectory: flatness-based planner (get_flatness_trajectory).
Pass a path to use a pre-generated NPZ (minco or planner format).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from dq_nmpc.nmpc.drone_visualizer import DroneVisualizer
from dq_nmpc.nmpc.se3_controller import se3_control
from dq_nmpc.schema import FlatnessTrajectory, NMPCConfig, OutputPaths, Se3Config

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
    params = config.to_params_dict()
    logger.info("Config loaded: %s", config_path)

    if se3_config_path is not None:
        se3_config = Se3Config.from_yaml(se3_config_path)
        logger.info("SE3 config loaded: %s", se3_config_path)
    else:
        se3_config = Se3Config()
        logger.info("SE3 config: defaults")

    mass = params["mass"]
    gravity = params["gravity"]
    ts = params["nmpc"]["ts"]

    if planner:
        from dq_nmpc.nmpc.planner import get_flatness_trajectory

        ref = get_flatness_trajectory(
            params=config,
            initial_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            t_initial=2.0,
            t_trajectory=10.0,
            t_final=2.0,
            sample_time=ts,
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
            from dq_nmpc.minco_trajectory.loader import load_trajectory_npz

            traj5 = load_trajectory_npz(trajectory_path)
            pts = np.array(
                [traj5.get_pos(t) for t in np.linspace(0, traj5.total_duration, 200)],
                dtype=np.float64,
            )
            traj = FlatnessTrajectory(
                ref_pos=pts,
                t=np.linspace(0, traj5.total_duration, 200),
                ref_yaw=np.zeros(200),
            )
            first_pos = np.array(traj5.get_pos(0.0), dtype=np.float64).ravel()
            viz_traj = traj5
            logger.info(
                "Trajectory loaded (minco): duration=%.2f s, %d pieces",
                traj5.total_duration,
                len(traj5),
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

    dt_ns = int(ts * 1e9)
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

    ctrl_writer.detach()
    state_reader.detach()
    logger.info("we are going to enter nmpc")


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
