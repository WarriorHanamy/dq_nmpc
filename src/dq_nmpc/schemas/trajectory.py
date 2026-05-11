"""Reference trajectory schemas."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class TrajectoryPoint(BaseModel):
    """Single point on the reference trajectory (13D state + 4D control)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    thrust: float = 0.0
    torque_x: float = 0.0
    torque_y: float = 0.0
    torque_z: float = 0.0

    def state_as_array(self) -> np.ndarray:
        """Return (13,) state array."""
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.qw,
                self.qx,
                self.qy,
                self.qz,
                self.wx,
                self.wy,
                self.wz,
            ],
            dtype=np.float64,
        )

    def control_as_array(self) -> np.ndarray:
        """Return (4,) control array."""
        return np.array(
            [self.thrust, self.torque_x, self.torque_y, self.torque_z],
            dtype=np.float64,
        )


class ReferenceTrajectory(BaseModel):
    """Full reference trajectory over the prediction horizon."""

    points: list[TrajectoryPoint] = Field(default_factory=list)
    horizon_steps: int = Field(default=10, gt=0)
    cost_weights: list[float] = Field(default_factory=lambda: [1.0] * 30)

    def state_matrix(self) -> np.ndarray:
        """Return (N, 13) reference state matrix."""
        return np.column_stack([p.state_as_array() for p in self.points])

    def control_matrix(self) -> np.ndarray:
        """Return (N, 4) reference control matrix."""
        return np.column_stack([p.control_as_array() for p in self.points])
