"""Rerun-based live visualizer for drone guidance and NMPC tracking.

Records drone pose, target markers, trajectory path,
position error, thrust, body torques, and NMPC solver diagnostics.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
from rerun.blueprint import Blueprint, Horizontal, Spatial3DView, TimeSeriesView, Vertical

__all__ = ["DroneVisualizer"]


class DroneVisualizer:
    """Live + offline Rerun recorder.

    Usage::

        viz = DroneVisualizer("out/boot.rrd")
        viz.log_static_trajectory(traj5)
        viz.log_static_markers(takeoff=(0, 0, 1.5), first_traj=(2, 0, 2))
        viz.log_drone(pos, quat, sim_time, error=pos_error, thrust=thrust,
                      tau_x=tx, tau_y=ty, tau_z=tz)
        viz.log_nmpc_reference(ref_pos)
        viz.log_nmpc_stats(solve_ms=..., residuals=..., qp_iter=..., qp_stat=...)
    """

    def __init__(self, rrd_path: str, application_id: str = "dq_nmpc", spawn: bool = False) -> None:
        rr.init(application_id, spawn=spawn)
        rr.save(rrd_path)

        blueprint = Blueprint(
            Vertical(
                Spatial3DView(origin="/", name="3D View"),
                Horizontal(
                    TimeSeriesView(
                        origin="/drone",
                        contents=[
                            "/drone/pos_x",
                            "/drone/pos_y",
                            "/drone/pos_z",
                            "/nmpc/ref_x",
                            "/nmpc/ref_y",
                            "/nmpc/ref_z",
                        ],
                        name="Position",
                    ),
                    TimeSeriesView(
                        origin="/control",
                        contents=[
                            "/control/thrust",
                            "/control/torque_x",
                            "/control/torque_y",
                            "/control/torque_z",
                            "/nmpc/ref_thrust",
                        ],
                        name="Control",
                    ),
                    TimeSeriesView(
                        origin="/error",
                        contents=[
                            "/error/position",
                            "/nmpc/pos_err_x",
                            "/nmpc/pos_err_y",
                            "/nmpc/pos_err_z",
                        ],
                        name="Error",
                    ),
                    TimeSeriesView(
                        origin="/nmpc",
                        contents=[
                            "/nmpc/solve_ms",
                            "/nmpc/qp_iter",
                            "/nmpc/res_eq",
                            "/nmpc/res_ineq",
                            "/nmpc/res_comp",
                            "/nmpc/res_stat",
                        ],
                        name="Solver",
                    ),
                    name="Scalars",
                    column_shares=[2, 2, 2, 1],
                ),
                row_shares=[3, 1],
            ),
        )
        rr.send_blueprint(blueprint, make_default=True)

    def log_static_trajectory(self, traj, num_samples: int = 200) -> None:
        """Log the full reference trajectory path as a static line strip.

        Accepts ``RefTrajectoryAsBelts`` (extracts position from first point
        of each belt via CasADi), or a minco ``Trajectory7`` (samples via
        ``get_pos``), or a legacy object with ``ref_pos`` attribute.
        """
        from dq_nmpc.nmpc.dq_functions import position_from_dualquat_ca_func

        if hasattr(traj, "belts"):
            dq_to_pos = position_from_dualquat_ca_func()
            dq_all = traj.belts[:, 0, :8].T  # (8, N_c)
            pts_all = np.array(dq_to_pos(dq_all)).T  # (N_c, 3)
            points = [(float(p[0]), float(p[1]), float(p[2])) for p in pts_all]
        elif hasattr(traj, "ref_pos"):
            pts = traj.ref_pos
            points = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
        else:
            duration = traj.total_duration
            dt = duration / max(num_samples - 1, 1)
            points = []
            for i in range(num_samples):
                t = i * dt
                p = np.array(traj.get_pos(min(t, duration)), dtype=np.float64).ravel()
                points.append((float(p[0]), float(p[1]), float(p[2])))
        rr.log(
            "trajectory/path",
            rr.LineStrips3D([points], colors=[(128, 128, 128)]),
            static=True,
        )

    def log_static_markers(
        self,
        takeoff: tuple[float, float, float] = (0.0, 0.0, 1.5),
        first_traj: tuple[float, float, float] = (2.0, 0.0, 2.0),
    ) -> None:
        """Log static marker points for takeoff and first trajectory point."""
        rr.log(
            "target/takeoff",
            rr.Points3D([takeoff], radii=[0.06], colors=[(0, 128, 255)]),
            static=True,
        )
        rr.log(
            "trajectory/first",
            rr.Points3D([first_traj], radii=[0.08], colors=[(255, 0, 0)]),
            static=True,
        )

    def log_target(self, pos: np.ndarray) -> None:
        """Log current SE3 target as a green sphere."""
        rr.log(
            "target/current",
            rr.Points3D(
                [(float(pos[0]), float(pos[1]), float(pos[2]))],
                radii=[0.05],
                colors=[(0, 255, 0)],
            ),
        )

    def log_drone(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        sim_time: float,
        error: float | None = None,
        thrust: float | None = None,
        tau_x: float | None = None,
        tau_y: float | None = None,
        tau_z: float | None = None,
    ) -> None:
        """Log drone pose, position scalars, error, thrust, and torques."""
        rr.set_time("sim_time", timestamp=sim_time)

        rr.log(
            "drone/pose",
            rr.Transform3D(
                translation=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation=rr.Quaternion(
                    xyzw=(
                        float(quat_wxyz[1]),
                        float(quat_wxyz[2]),
                        float(quat_wxyz[3]),
                        float(quat_wxyz[0]),
                    )
                ),
            ),
        )

        rr.log(
            "drone/pose/body_axes",
            rr.Arrows3D(
                origins=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                vectors=[(0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.3)],
                colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            ),
        )

        self._log_pos(pos)

        if error is not None:
            self._log_error(error)

        if thrust is not None and tau_x is not None and tau_y is not None and tau_z is not None:
            self._log_control(thrust, tau_x, tau_y, tau_z)

    def log_nmpc_reference(self, ref_pos: np.ndarray) -> None:
        """Log NMPC reference position as a bright red sphere and scalars."""
        x, y, z = float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])
        rr.log(
            "nmpc/ref_point",
            rr.Points3D([(x, y, z)], radii=[0.04], colors=[(255, 50, 50)]),
        )
        rr.log("nmpc/ref_x", rr.Scalars([x]))
        rr.log("nmpc/ref_y", rr.Scalars([y]))
        rr.log("nmpc/ref_z", rr.Scalars([z]))

    def log_nmpc_horizon(
        self,
        positions: list[tuple[float, float, float]],
    ) -> None:
        """Log NMPC horizon preview: blue line strip + red gradient spheres.

        The first point (i=0) is the current-step reference (bright red,
        larger radius).  Later points are future references (dimmer red,
        smaller radius).

        @param[in] positions  N horizon reference positions [(x, y, z), ...]
        """
        rr.log(
            "nmpc/horizon/path",
            rr.LineStrips3D([positions], colors=[(60, 120, 255)]),
        )
        for i, pt in enumerate(positions):
            color = (255, 0, 0) if i == 0 else (140, 30, 30)
            radius = 0.035 if i == 0 else 0.022
            rr.log(
                f"nmpc/horizon/s{i}",
                rr.Points3D([pt], radii=[radius], colors=[color]),
            )

    def log_nmpc_stats(
        self,
        solve_ms: float,
        residuals: np.ndarray,
        qp_iter: np.ndarray,
        qp_stat: np.ndarray | None = None,
        ref_thrust: float | None = None,
        pos_err_xyz: np.ndarray | None = None,
    ) -> None:
        """Log NMPC solver diagnostics as scalar time series.

        @param[in] solve_ms    Wall-clock solve time [ms]
        @param[in] residuals   (4,) array: eq, ineq, comp, stat residual norms
        @param[in] qp_iter     (2,) array: QP iterations [outer, inner]
        @param[in] qp_stat     (2,) array: QP solver status codes (optional)
        @param[in] ref_thrust  Feedforward reference thrust [N] (optional)
        @param[in] pos_err_xyz (3,) per-axis position error [m] (optional)
        """
        rr.log("nmpc/solve_ms", rr.Scalars([solve_ms]))
        rr.log("nmpc/res_eq", rr.Scalars([float(residuals[0])]))
        rr.log("nmpc/res_ineq", rr.Scalars([float(residuals[1])]))
        rr.log("nmpc/res_comp", rr.Scalars([float(residuals[2])]))
        rr.log("nmpc/res_stat", rr.Scalars([float(residuals[3])]))
        rr.log("nmpc/qp_iter", rr.Scalars([float(qp_iter[0])]))

        if qp_stat is not None and len(qp_stat) >= 2:
            rr.log("nmpc/qp_stat", rr.Scalars([float(qp_stat[0])]))

        if ref_thrust is not None:
            rr.log("nmpc/ref_thrust", rr.Scalars([ref_thrust]))

        if pos_err_xyz is not None:
            rr.log("nmpc/pos_err_x", rr.Scalars([float(pos_err_xyz[0])]))
            rr.log("nmpc/pos_err_y", rr.Scalars([float(pos_err_xyz[1])]))
            rr.log("nmpc/pos_err_z", rr.Scalars([float(pos_err_xyz[2])]))

    def _log_pos(self, pos: np.ndarray) -> None:
        rr.log("drone/pos_x", rr.Scalars([float(pos[0])]))
        rr.log("drone/pos_y", rr.Scalars([float(pos[1])]))
        rr.log("drone/pos_z", rr.Scalars([float(pos[2])]))

    def _log_control(self, thrust: float, tau_x: float, tau_y: float, tau_z: float) -> None:
        rr.log("control/thrust", rr.Scalars([thrust]))
        rr.log("control/torque_x", rr.Scalars([tau_x]))
        rr.log("control/torque_y", rr.Scalars([tau_y]))
        rr.log("control/torque_z", rr.Scalars([tau_z]))

    def _log_error(self, position_error: float) -> None:
        rr.log("error/position", rr.Scalars([position_error]))
