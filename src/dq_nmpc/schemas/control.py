"""NMPC control output schema."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class ControlCommand(BaseModel):
    """4D control: body-frame thrust + torques."""

    thrust: float = Field(default=0.0, ge=0.0, description="Thrust [N]")
    torque_x: float = Field(default=0.0, description="Torque about body x [N·m]")
    torque_y: float = Field(default=0.0, description="Torque about body y [N·m]")
    torque_z: float = Field(default=0.0, description="Torque about body z [N·m]")

    def to_array(self) -> np.ndarray:
        """Return (4,) numpy array [thrust, tau_x, tau_y, tau_z]."""
        return np.array(
            [self.thrust, self.torque_x, self.torque_y, self.torque_z],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ControlCommand:
        """Construct from (4,) array."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        return cls(
            thrust=arr[0],
            torque_x=arr[1],
            torque_y=arr[2],
            torque_z=arr[3],
        )
