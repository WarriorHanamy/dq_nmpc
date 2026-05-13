"""State schemas: dual quaternion (14D) and classical (13D) representations."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class DualQuaternionState(BaseModel):
    """14D state: dual quaternion (8) + body-frame twist (6).

    Layout: [qw,qx,qy,qz, dw,dx,dy,dz, wx,wy,wz, vx,vy,vz]
    - qw,qx,qy,qz: primary quaternion (world ENU orientation)
    - dw,dx,dy,dz: dual part (encodes world ENU position)
    - wx,wy,wz:     body FLU angular velocity [rad/s]
    - vx,vy,vz:     body FLU linear velocity [m/s]
    """

    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    dw: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return (14,) numpy array for acados solver."""
        return np.array(
            [
                self.qw,
                self.qx,
                self.qy,
                self.qz,
                self.dw,
                self.dx,
                self.dy,
                self.dz,
                self.wx,
                self.wy,
                self.wz,
                self.vx,
                self.vy,
                self.vz,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> DualQuaternionState:
        """Construct from (14,) array."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        return cls(
            qw=arr[0],
            qx=arr[1],
            qy=arr[2],
            qz=arr[3],
            dw=arr[4],
            dx=arr[5],
            dy=arr[6],
            dz=arr[7],
            wx=arr[8],
            wy=arr[9],
            wz=arr[10],
            vx=arr[11],
            vy=arr[12],
            vz=arr[13],
        )

    def split_dual(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (dual_quat (8,), twist (6,))."""
        arr = self.to_array()
        return arr[:8], arr[8:]


class ClassicalState(BaseModel):
    """13D classical state for ROS / SHM compatibility.

    Layout: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    - x, y, z:        world ENU position [m]
    - vx, vy, vz:      body FLU linear velocity [m/s]
    - qw, qx, qy, qz:  world ENU orientation quaternion
    - wx, wy, wz:      body FLU angular velocity [rad/s]
    """

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

    def to_array(self) -> np.ndarray:
        """Return (13,) numpy array."""
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

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ClassicalState:
        """Construct from (13,) array."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        return cls(
            x=arr[0],
            y=arr[1],
            z=arr[2],
            vx=arr[3],
            vy=arr[4],
            vz=arr[5],
            qw=arr[6],
            qx=arr[7],
            qy=arr[8],
            qz=arr[9],
            wx=arr[10],
            wy=arr[11],
            wz=arr[12],
        )
