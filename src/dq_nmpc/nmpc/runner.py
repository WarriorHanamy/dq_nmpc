"""SHM-based NMPC runtime loop.

Reads quadrotor state from /dev/shm/quadrotor_sim/state, solves the acados OCP,
writes control to /dev/shm/quadrotor_sim/ctrl.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from dq_nmpc.nmpc.controller import solver
from dq_nmpc.nmpc.dynamics import (
    dual_velocity_casadi,
    dualquat_quat_casadi,
    dualquat_trans_casadi,
    error_dual_aux_casadi,
    rotation_casadi,
    rotation_inverse_casadi,
    velocities_from_twist_casadi,
)
from dq_nmpc.nmpc.functions import dualquat_from_pose_casadi
from dq_nmpc.schemas.config import NMPCConfig
from dq_nmpc.trajectory.loader import load_trajectory_csv

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

dualquat_from_pose = dualquat_from_pose_casadi()
get_trans = dualquat_trans_casadi()
get_quat = dualquat_quat_casadi()
dual_twist = dual_velocity_casadi()
velocity_from_twist = velocities_from_twist_casadi()
rot = rotation_casadi()
inverse_rot = rotation_inverse_casadi()
error_dual_f = error_dual_aux_casadi()


def run_nmpc(
    config_path: str | Path,
    trajectory_path: str | Path,
    flag_build: bool = True,
    max_iter: int = 0,
):
    if not SHM_AVAILABLE:
        raise RuntimeError("quadrotor_sim.shm not available; ensure quadrotor_sim is on PYTHONPATH")

    config = NMPCConfig.from_yaml(config_path)
    params = config.to_params_dict()

    Jxx, Jyy, Jzz = params["ixx"], params["iyy"], params["izz"]
    J = np.array([[Jxx, 0.0, 0.0], [0.0, Jyy, 0.0], [0.0, 0.0, Jzz]])

    Q_nmpc = np.array(params["nmpc"]["Q"])
    Q_e_nmpc = np.array(params["nmpc"]["Q_e"])
    R_nmpc = np.array(params["nmpc"]["R"])

    N_prediction = params["nmpc"]["horizon_steps"]
    ts = params["nmpc"]["ts"]

    acados_ocp_solver, ocp = solver(params, flag_build)

    trajectory = load_trajectory_csv(trajectory_path)
    num_ref = len(trajectory.points)
    if num_ref < N_prediction:
        logger.warning("Trajectory has %d points, horizon needs %d", num_ref, N_prediction)

    X_d = np.zeros((14, N_prediction), dtype=np.float64)
    u_d = np.zeros((4, N_prediction), dtype=np.float64)

    for i in range(min(num_ref, N_prediction)):
        tp = trajectory.points[i]
        dual_d = dualquat_from_pose(tp.qw, tp.qx, tp.qy, tp.qz, tp.x, tp.y, tp.z)
        angular_linear_d = np.array([tp.wx, tp.wy, tp.wz, tp.vx, tp.vy, tp.vz])
        dual_twist_d = dual_twist(angular_linear_d, dual_d)
        X_d[0:8, i] = np.array(dual_d).ravel()
        X_d[8:14, i] = np.array(dual_twist_d).ravel()
        u_d[0, i] = tp.thrust
        w_dot = np.array([0.0, 0.0, 0.0])
        u_d[1:4, i] = J @ w_dot + np.cross(
            np.array([tp.wx, tp.wy, tp.wz]), J @ np.array([tp.wx, tp.wy, tp.wz])
        )

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

    logger.info("NMPC runner started (ts=%.3f s, horizon=%d steps)", ts, N_prediction)

    dt_ns = int(ts * 1e9)
    next_wake = time.monotonic_ns()
    iteration = 0

    try:
        while True:
            if max_iter > 0 and iteration >= max_iter:
                break

            while not state_reader.read(state_buf):
                time.sleep(0.0001)

            pos = np.array(state_buf.position[:], dtype=np.float64)
            quat_wxyz = np.array(state_buf.orientation[:], dtype=np.float64)
            lin_vel_body = np.array(state_buf.linear_velocity[:], dtype=np.float64)
            ang_vel_body = np.array(state_buf.angular_velocity[:], dtype=np.float64)

            qw, qx, qy, qz = quat_wxyz
            dual_state = dualquat_from_pose(qw, qx, qy, qz, pos[0], pos[1], pos[2])

            X = np.zeros((14, 1), dtype=np.float64)
            X[:8, 0] = np.array(dual_state).ravel()
            X[8:11, 0] = ang_vel_body
            X[11:14, 0] = lin_vel_body

            acados_ocp_solver.set(0, "lbx", X[:, 0])
            acados_ocp_solver.set(0, "ubx", X[:, 0])

            for j in range(N_prediction):
                yref = X_d[:, j]
                uref = u_d[:, j]
                aux_ref = np.hstack((yref, uref, Q_nmpc, Q_e_nmpc, R_nmpc))
                acados_ocp_solver.set(j, "p", aux_ref)

            acados_ocp_solver.set(
                N_prediction, "p", np.hstack((X_d[:, -1], u_d[:, -1], Q_nmpc, Q_e_nmpc, R_nmpc))
            )

            status = acados_ocp_solver.solve()
            if status != 0:
                logger.warning("Solver failed (status=%d), using previous control", status)

            u_control = acados_ocp_solver.get(0, "u")
            u_arr = np.array(u_control, dtype=np.float64).ravel()

            ctrl_writer.write_control(
                thrust=float(u_arr[0]),
                torque_x=float(u_arr[1]),
                torque_y=float(u_arr[2]),
                torque_z=float(u_arr[3]),
            )

            now_ns = time.monotonic_ns()
            if now_ns < next_wake:
                remaining_ns = next_wake - now_ns
                time.sleep(remaining_ns / 1e9)
            else:
                next_wake = now_ns

            next_wake += dt_ns
            iteration += 1

    except KeyboardInterrupt:
        logger.info("NMPC runner stopped by user")
    finally:
        ctrl_writer.detach()
        state_reader.detach()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 3:
        print("Usage: dq-nmpc-runner <nmpc.yaml> <trajectory.csv> [--no-build] [--max-iter N]")
        sys.exit(1)

    config_path = sys.argv[1]
    trajectory_path = sys.argv[2]
    flag_build = "--no-build" not in sys.argv
    max_iter = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--max-iter" and i + 1 < len(sys.argv):
            max_iter = int(sys.argv[i + 1])

    run_nmpc(config_path, trajectory_path, flag_build=flag_build, max_iter=max_iter)


if __name__ == "__main__":
    main()
