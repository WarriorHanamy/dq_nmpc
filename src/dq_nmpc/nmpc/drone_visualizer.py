"""Rerun-based live visualizer for drone guidance and NMPC tracking.

Records drone pose, body axes, target markers, trajectory path,
position error, thrust, and body torques.
"""

from __future__ import annotations

import numpy as np
import rerun as rr

from dq_nmpc.nmpc.se3_controller import quat_to_rotmat


class DroneVisualizer:
    """Live + offline Rerun recorder for drone guidance phases.

    Usage::

        viz = DroneVisualizer("out/se3_bootstrap.rrd")
        viz.log_static_trajectory(traj5)
        viz.log_static_markers(takeoff=(0, 0, 1.5), first_traj=(2, 0, 2))
        while not converged:
            viz.log_drone(pos, quat, sim_time)
            viz.log_control(thrust, tx, ty, tz)
            viz.log_error(pos_error)
    """

    def __init__(self, rrd_path: str, application_id: str = "dq_nmpc_se3") -> None:
        rr.init(application_id, spawn=True)
        rr.save(rrd_path)

    def log_static_trajectory(self, traj5, num_samples: int = 200) -> None:
        """Log the full reference trajectory path as a static line strip."""
        duration = traj5.total_duration
        dt = duration / max(num_samples - 1, 1)
        points = []
        for i in range(num_samples):
            t = i * dt
            p = np.array(traj5.get_pos(min(t, duration)), dtype=np.float64).ravel()
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

    def log_drone(self, pos: np.ndarray, quat_wxyz: np.ndarray, sim_time: float) -> None:
        """Log drone world ENU pose (Transform3D) and body FLU axes (Arrows3D)."""
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

        R = quat_to_rotmat(
            float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
        )
        origin = (0.0, 0.0, 0.0)
        length = 0.3
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        labels = ["x_body", "y_body", "z_body"]
        vectors = [
            (float(R[0, 0]) * length, float(R[0, 1]) * length, float(R[0, 2]) * length),
            (float(R[1, 0]) * length, float(R[1, 1]) * length, float(R[1, 2]) * length),
            (float(R[2, 0]) * length, float(R[2, 1]) * length, float(R[2, 2]) * length),
        ]
        rr.log(
            "drone/body_axes",
            rr.Arrows3D(
                origins=[origin, origin, origin],
                vectors=vectors,
                colors=colors,
                labels=labels,
            ),
        )

    def log_control(self, thrust: float, tau_x: float, tau_y: float, tau_z: float) -> None:
        """Log thrust and body torques as scalar time series."""
        rr.log("control/thrust", rr.Scalars([thrust]))
        rr.log("control/torque_x", rr.Scalars([tau_x]))
        rr.log("control/torque_y", rr.Scalars([tau_y]))
        rr.log("control/torque_z", rr.Scalars([tau_z]))

    def log_error(self, position_error: float) -> None:
        """Log position error as scalar time series."""
        rr.log("error/position", rr.Scalars([position_error]))
