"""Static Rerun visualization of NMPC reinterpreted reference trajectory.

Takes dense ``ref_params (N, 18)`` and writes a ``.rrd`` file with
3D path, body axes, position, angular velocity, body velocity, thrust,
and torque time series.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.blueprint import Blueprint, Horizontal, Spatial3DView, TimeSeriesView, Vertical

from dq_nmpc.nmpc._dq_functions import position_from_dualquat_ca_func
from dq_nmpc.schema import (
    NMPC_REF_DQ_SLICE,
    NMPC_REF_OMEGA_SLICE,
    NMPC_REF_UNOM_SLICE,
    NMPC_REF_VEL_SLICE,
)

logger = logging.getLogger(__name__)

_REF_APP_ID = "dq_nmpc_ref"
_BODY_AXIS_LEN = 0.25
_BODY_AXES_STRIDE = 10


def visualize_ref_params(
    ref_params: np.ndarray,
    output_path: str | Path,
    *,
    dt: float,
    spawn: bool = False,
) -> Path:
    """Write a static .rrd file visualizing the reinterpreted reference.

    @param[in] ref_params   ``(N, 18)`` dense reference parameters
    @param[in] output_path  Output ``.rrd`` file path
    @param[in] dt           Time step between ref_params rows [s]
    @param[in] spawn        Spawn Rerun viewer after writing
    @return                 Path to the written ``.rrd`` file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(_REF_APP_ID, spawn=spawn)
    rr.save(str(output_path))

    blueprint = Blueprint(
        Vertical(
            Spatial3DView(origin="/reference/3d", name="Reference Trajectory"),
            Horizontal(
                TimeSeriesView(
                    origin="/reference/position",
                    contents=[
                        "/reference/position/x",
                        "/reference/position/y",
                        "/reference/position/z",
                    ],
                    name="Position [m]",
                ),
                TimeSeriesView(
                    origin="/reference/angular_velocity",
                    contents=[
                        "/reference/omega/wx",
                        "/reference/omega/wy",
                        "/reference/omega/wz",
                    ],
                    name="Angular Velocity [rad/s]",
                ),
                name="States",
                column_shares=[1, 1],
            ),
            Horizontal(
                TimeSeriesView(
                    origin="/reference/linear_velocity",
                    contents=[
                        "/reference/vel/vx",
                        "/reference/vel/vy",
                        "/reference/vel/vz",
                    ],
                    name="Body Velocity [m/s]",
                ),
                TimeSeriesView(
                    origin="/reference/control",
                    contents=[
                        "/reference/control/thrust",
                        "/reference/control/tau_x",
                        "/reference/control/tau_y",
                        "/reference/control/tau_z",
                    ],
                    name="Control",
                ),
                name="Controls",
                column_shares=[1, 1],
            ),
            row_shares=[2, 1, 1],
        ),
    )
    rr.send_blueprint(blueprint, make_default=True)

    N = ref_params.shape[0]
    t_vec = np.arange(N, dtype=np.float64) * dt

    _log_3d_trajectory(ref_params)
    _log_time_series(ref_params, t_vec)

    logger.info("Reference visualization written: %s", output_path)
    return output_path


def _log_3d_trajectory(ref_params: np.ndarray) -> None:
    """Log 3D trajectory path and periodic body-axis arrows."""
    dq_to_pos = position_from_dualquat_ca_func()
    dq_all = ref_params[:, NMPC_REF_DQ_SLICE].T  # (8, N)
    pos_all = np.array(dq_to_pos(dq_all)).T  # (N, 3)

    N = ref_params.shape[0]
    t_dt = np.arange(N, dtype=np.float64)

    points = [(float(p[0]), float(p[1]), float(p[2])) for p in pos_all]
    rr.log(
        "reference/3d/path",
        rr.LineStrips3D([points], colors=[(128, 128, 128)]),
        static=True,
    )

    start = pos_all[0]
    rr.log(
        "reference/3d/start",
        rr.Points3D(
            [(float(start[0]), float(start[1]), float(start[2]))],
            radii=[0.08],
            colors=[(0, 255, 0)],
        ),
        static=True,
    )
    end = pos_all[-1]
    rr.log(
        "reference/3d/end",
        rr.Points3D(
            [(float(end[0]), float(end[1]), float(end[2]))],
            radii=[0.08],
            colors=[(255, 0, 0)],
        ),
        static=True,
    )

    for i in range(0, N, _BODY_AXES_STRIDE):
        qw, qx, qy, qz = (
            float(ref_params[i, 0]),
            float(ref_params[i, 1]),
            float(ref_params[i, 2]),
            float(ref_params[i, 3]),
        )
        px, py, pz = float(pos_all[i, 0]), float(pos_all[i, 1]), float(pos_all[i, 2])

        x_body = (
            float(qw * qw + qx * qx - qy * qy - qz * qz),
            float(2.0 * (qx * qy + qw * qz)),
            float(2.0 * (qx * qz - qw * qy)),
        )
        y_body = (
            float(2.0 * (qx * qy - qw * qz)),
            float(qw * qw - qx * qx + qy * qy - qz * qz),
            float(2.0 * (qy * qz + qw * qx)),
        )
        z_body = (
            float(2.0 * (qx * qz + qw * qy)),
            float(2.0 * (qy * qz - qw * qx)),
            float(qw * qw - qx * qx - qy * qy + qz * qz),
        )
        rr.set_time_seconds("time", float(t_dt[i]))
        rr.log(
            "reference/3d/body_axes",
            rr.Arrows3D(
                origins=[(px, py, pz), (px, py, pz), (px, py, pz)],
                vectors=[
                    (
                        x_body[0] * _BODY_AXIS_LEN,
                        x_body[1] * _BODY_AXIS_LEN,
                        x_body[2] * _BODY_AXIS_LEN,
                    ),
                    (
                        y_body[0] * _BODY_AXIS_LEN,
                        y_body[1] * _BODY_AXIS_LEN,
                        y_body[2] * _BODY_AXIS_LEN,
                    ),
                    (
                        z_body[0] * _BODY_AXIS_LEN,
                        z_body[1] * _BODY_AXIS_LEN,
                        z_body[2] * _BODY_AXIS_LEN,
                    ),
                ],
                colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            ),
        )


def _log_time_series(ref_params: np.ndarray, t_vec: np.ndarray) -> None:
    """Log scalar time series for position, omega, body velocity, and control."""
    dq_to_pos = position_from_dualquat_ca_func()
    dq_all = ref_params[:, NMPC_REF_DQ_SLICE].T  # (8, N)
    pos_all = np.array(dq_to_pos(dq_all)).T  # (N, 3)

    omega = ref_params[:, NMPC_REF_OMEGA_SLICE]  # (N, 3)
    vel_body = ref_params[:, NMPC_REF_VEL_SLICE]  # (N, 3)
    thrust = ref_params[:, NMPC_REF_UNOM_SLICE.start]  # (N,)
    torque = ref_params[:, (NMPC_REF_UNOM_SLICE.start + 1) : NMPC_REF_UNOM_SLICE.stop]  # (N, 3)

    for i in range(len(t_vec)):
        t = float(t_vec[i])
        rr.set_time_seconds("time", t)
        rr.log("reference/position/x", rr.Scalars([float(pos_all[i, 0])]))
        rr.log("reference/position/y", rr.Scalars([float(pos_all[i, 1])]))
        rr.log("reference/position/z", rr.Scalars([float(pos_all[i, 2])]))
        rr.log("reference/omega/wx", rr.Scalars([float(omega[i, 0])]))
        rr.log("reference/omega/wy", rr.Scalars([float(omega[i, 1])]))
        rr.log("reference/omega/wz", rr.Scalars([float(omega[i, 2])]))
        rr.log("reference/vel/vx", rr.Scalars([float(vel_body[i, 0])]))
        rr.log("reference/vel/vy", rr.Scalars([float(vel_body[i, 1])]))
        rr.log("reference/vel/vz", rr.Scalars([float(vel_body[i, 2])]))
        rr.log("reference/control/thrust", rr.Scalars([float(thrust[i])]))
        rr.log("reference/control/tau_x", rr.Scalars([float(torque[i, 0])]))
        rr.log("reference/control/tau_y", rr.Scalars([float(torque[i, 1])]))
        rr.log("reference/control/tau_z", rr.Scalars([float(torque[i, 2])]))
