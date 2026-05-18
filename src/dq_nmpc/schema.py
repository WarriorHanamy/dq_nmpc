"""Single-source Pydantic models for dq_nmpc — configs, state, control, trajectory.

All models are frozen: invalid state is rejected at construction, not discovered at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Layout constants — single source of truth for I/O contracts
# ---------------------------------------------------------------------------

TRAJECTORY_CSV_COLUMNS: tuple[str, ...] = (
    "t",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "jx",
    "jy",
    "jz",
    "qw",
    "qx",
    "qy",
    "qz",
    "wx",
    "wy",
    "wz",
    "thrust",
)

ARTIFACTS_DIR = "out"


def csv_column_index(name: str) -> int:
    """Return 0-based column index for a name in TRAJECTORY_CSV_COLUMNS.

    Raises ValueError if the name is not a recognized column.
    """
    return TRAJECTORY_CSV_COLUMNS.index(name)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class NMPCParams(BaseModel):
    """NMPC solver hyperparameters.

    Every fixed-size array uses list[float] with exact max_length.
    Cross-field validation ensures Q/Q_e match nx and R/lbu/ubu match nu.
    """

    model_config = ConfigDict(frozen=True)

    Q: list[float] = Field(min_length=14, max_length=14)
    Q_e: list[float] = Field(min_length=14, max_length=14)
    R: list[float] = Field(min_length=4, max_length=4)
    nx: int = Field(default=14, gt=0)
    nu: int = Field(default=4, gt=0)
    lbu: list[float] = Field(min_length=4, max_length=4)
    ubu: list[float] = Field(min_length=4, max_length=4)
    horizon_steps: int = Field(default=10, gt=0)
    horizon_time: float = Field(default=1.0, gt=0.0)
    ts: float = Field(default=0.01, gt=0.0, description="Sample time [s]")

    @model_validator(mode="after")
    def _check_lengths_match_dims(self):
        if len(self.Q) != self.nx:
            raise ValueError(f"len(Q)={len(self.Q)} must equal nx={self.nx}")
        if len(self.Q_e) != self.nx:
            raise ValueError(f"len(Q_e)={len(self.Q_e)} must equal nx={self.nx}")
        if len(self.R) != self.nu:
            raise ValueError(f"len(R)={len(self.R)} must equal nu={self.nu}")
        if len(self.lbu) != self.nu:
            raise ValueError(f"len(lbu)={len(self.lbu)} must equal nu={self.nu}")
        if len(self.ubu) != self.nu:
            raise ValueError(f"len(ubu)={len(self.ubu)} must equal nu={self.nu}")
        return self


class NMPCConfig(BaseModel):
    """Validated NMPC configuration, loadable from YAML."""

    model_config = ConfigDict(frozen=True)

    mass: float = Field(gt=0.0, description="Mass [kg]")
    gravity: float = Field(default=9.80665, gt=0.0, description="Gravity [m/s^2]")
    ixx: float = Field(gt=0.0, description="Inertia about x [kg·m^2]")
    iyy: float = Field(gt=0.0, description="Inertia about y [kg·m^2]")
    izz: float = Field(gt=0.0, description="Inertia about z [kg·m^2]")
    mav_name: str = Field(default="quadrotor")
    nmpc: NMPCParams

    @classmethod
    def from_yaml(cls, path: str | Path) -> NMPCConfig:
        """Load and validate configuration from a YAML file.

        Handles the ROS2 '/**/ros__parameters' nesting convention.
        """
        with open(path, "r") as stream:
            raw = yaml.safe_load(stream)

        if "/**" in raw:
            raw = raw["/**"]["ros__parameters"]

        return cls(**raw)

    def to_params_dict(self) -> dict:
        """Return a dict structured for the NMPC solver (backward-compatible)."""
        return {
            "mass": self.mass,
            "gravity": self.gravity,
            "ixx": self.ixx,
            "iyy": self.iyy,
            "izz": self.izz,
            "mav_name": self.mav_name,
            "nmpc": {
                "Q": self.nmpc.Q,
                "Q_e": self.nmpc.Q_e,
                "R": self.nmpc.R,
                "ubu": self.nmpc.ubu,
                "lbu": self.nmpc.lbu,
                "horizon_steps": self.nmpc.horizon_steps,
                "ts": self.nmpc.ts,
                "horizon_time": self.nmpc.horizon_time,
                "nx": self.nmpc.nx,
                "nu": self.nmpc.nu,
            },
        }


class ControlCommand(BaseModel):
    """4D control: body-frame thrust + torques."""

    model_config = ConfigDict(frozen=True)

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


class DualQuaternionState(BaseModel):
    """14D state: dual quaternion (8) + body-frame twist (6).

    Layout: [qw,qx,qy,qz, dw,dx,dy,dz, wx,wy,wz, vx,vy,vz]
    - qw,qx,qy,qz: primary quaternion (world ENU orientation)
    - dw,dx,dy,dz: dual part (encodes world ENU position)
    - wx,wy,wz:     body FLU angular velocity [rad/s]
    - vx,vy,vz:     body FLU linear velocity [m/s]
    """

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(frozen=True)

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


class TrajectoryPoint(BaseModel):
    """Single reference trajectory point (13D state + 4D control).

    - x, y, z:       world ENU position [m]
    - vx, vy, vz:     world ENU linear velocity [m/s]
    - qw,qx,qy,qz:    world ENU orientation quaternion
    - wx, wy, wz:     body FLU angular velocity [rad/s]
    - thrust:         body FLU thrust [N]
    - torque_x/y/z:   body FLU torque [Nm] (typically zero for ref)
    """

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(frozen=True)

    points: list[TrajectoryPoint] = Field(default_factory=list)
    horizon_steps: int = Field(default=10, gt=0)
    cost_weights: list[float] = Field(default_factory=lambda: [1.0] * 30)

    def state_matrix(self) -> np.ndarray:
        """Return (N, 13) reference state matrix."""
        return np.column_stack([p.state_as_array() for p in self.points])

    def control_matrix(self) -> np.ndarray:
        """Return (N, 4) reference control matrix."""
        return np.column_stack([p.control_as_array() for p in self.points])


class SHMConfig(BaseModel):
    """POSIX shared memory interface configuration."""

    model_config = ConfigDict(frozen=True)

    state_file: str = Field(default="/dev/shm/quadrotor_sim/state")
    ctrl_file: str = Field(default="/dev/shm/quadrotor_sim/ctrl")
    state_size: int = Field(default=192, gt=0, description="State segment size [bytes]")
    ctrl_size: int = Field(default=64, gt=0, description="Control segment size [bytes]")
    attach_timeout: float = Field(default=5.0, gt=0.0, description="SHM attach timeout [s]")

    @classmethod
    def default(cls) -> SHMConfig:
        """Return instance with default paths."""
        return cls()
