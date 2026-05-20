"""Single-source Pydantic models for dq_nmpc — configs, state, control, trajectory.

All models are frozen: invalid state is rejected at construction, not discovered at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class FlatnessTrajectory(BaseModel):
    """Dense flatness-based reference trajectory from get_flatness_trajectory.

    All arrays are (N,) or (N, D) with time-aligned row-major layout.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    ref_pos: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_vel: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_acc: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_jerk: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_snap: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_quat: np.ndarray = Field(default_factory=lambda: np.empty((0, 4)))
    ref_omega: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_omega_dot: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_thrust: np.ndarray = Field(default_factory=lambda: np.empty(0))
    ref_torque: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)))
    ref_yaw: np.ndarray = Field(default_factory=lambda: np.empty(0))
    ref_yaw_dot: np.ndarray = Field(default_factory=lambda: np.empty(0))
    ref_yaw_ddot: np.ndarray = Field(default_factory=lambda: np.empty(0))
    t: np.ndarray = Field(default_factory=lambda: np.empty(0))

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_ndarray(cls, v: object) -> np.ndarray:
        if isinstance(v, list):
            return np.asarray(v, dtype=np.float64)
        return v

    def interp_pos(self, t_query: float) -> np.ndarray:
        """Interpolate position at a query time.

        @param[in] t_query  Time [s]
        @return (3,) ndarray [m]
        """
        return np.array(
            [np.interp(t_query, self.t, self.ref_pos[:, i]) for i in range(3)],
            dtype=np.float64,
        )

    def interp_yaw(self, t_query: float) -> float:
        """Interpolate yaw angle at a query time.

        @param[in] t_query  Time [s]
        @return Yaw angle [rad]
        """
        return float(np.interp(t_query, self.t, self.ref_yaw))

    def save_npz(self, path: str | Path) -> None:
        """Save trajectory arrays to a compressed NPZ file."""
        np.savez(
            path,
            ref_pos=self.ref_pos,
            ref_vel=self.ref_vel,
            ref_acc=self.ref_acc,
            ref_jerk=self.ref_jerk,
            ref_snap=self.ref_snap,
            ref_quat=self.ref_quat,
            ref_omega=self.ref_omega,
            ref_omega_dot=self.ref_omega_dot,
            ref_thrust=self.ref_thrust,
            ref_torque=self.ref_torque,
            ref_yaw=self.ref_yaw,
            ref_yaw_dot=self.ref_yaw_dot,
            ref_yaw_ddot=self.ref_yaw_ddot,
            t=self.t,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> FlatnessTrajectory:
        """Load trajectory from a compressed NPZ file."""
        data = np.load(path)
        return cls(**{k: data[k] for k in data.files})


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


class TrajectoryConfig(BaseModel):
    """Trajectory generation parameters — single source of truth."""

    model_config = ConfigDict(frozen=True)

    shape: str = Field(default="circle", description="Trajectory shape")
    ts: float = Field(gt=0.0, description="Sample time, shared with NMPC [s]")
    mass: float = Field(gt=0.0, description="Flatness model mass [kg]")
    gravity: float = Field(default=9.80665, gt=0.0, description="Gravity [m/s²]")
    num_waypoints: int = Field(default=10, gt=1, description="Number of intermediate waypoints")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrajectoryConfig:
        """Load trajectory generation config from YAML."""
        with open(path, "r") as stream:
            raw = yaml.safe_load(stream)
        return cls(**raw)


class Se3Config(BaseModel):
    """SE(3) geometric controller gains (Lee et al. 2010)."""

    model_config = ConfigDict(frozen=True)

    K_p: list[float] = Field(default=[4.0, 4.0, 6.0], min_length=3, max_length=3)
    K_v: list[float] = Field(default=[3.0, 3.0, 4.0], min_length=3, max_length=3)
    K_R: list[float] = Field(default=[8.0, 8.0, 4.0], min_length=3, max_length=3)
    K_w: list[float] = Field(default=[1.0, 1.0, 0.5], min_length=3, max_length=3)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Se3Config:
        """Load SE3 controller gains from a YAML file."""
        with open(path, "r") as stream:
            raw = yaml.safe_load(stream)
        return cls(**raw)


class OutputPaths(BaseModel):
    """Resolved output artifact paths for a trajectory shape.

    Directories are created lazily on first property access.
    """

    model_config = ConfigDict(frozen=False)

    base_dir: str = "out"
    shape: str = "circle"

    @field_validator("shape")
    @classmethod
    def _shape_known(cls, v: str) -> str:
        from dq_nmpc.minco_trajectory.waypoints import SHAPES

        if v not in SHAPES:
            raise ValueError(f"Unknown shape '{v}'. Choose from: {SHAPES}")
        return v

    def _shape_dir(self) -> Path:
        return Path(self.base_dir) / self.shape

    def _mkdir(self) -> None:
        self._shape_dir().mkdir(parents=True, exist_ok=True)

    @property
    def trajectory_csv(self) -> Path:
        self._mkdir()
        return self._shape_dir() / "trajectory.csv"

    @property
    def trajectory_npz(self) -> Path:
        self._mkdir()
        return self._shape_dir() / "trajectory.npz"

    @property
    def trajectory_html(self) -> Path:
        self._mkdir()
        return self._shape_dir() / "trajectory.html"

    @property
    def se3_rrd(self) -> Path:
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.base_dir) / "se3_bootstrap.rrd"

    @classmethod
    def from_trajectory_config(cls, tc: TrajectoryConfig) -> OutputPaths:
        return cls(shape=tc.shape)
