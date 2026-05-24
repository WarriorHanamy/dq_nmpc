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

# NMPC OCP parameter vector layout
NMPC_REF_DIM = 18  # nx(14) + nu(4) — per shooting-node reference
NMPC_OCP_P_DIM = 50  # NMPC_REF_DIM + COST_PARAMS_DIM — full runtime p vector

# ref_params[18] sub-layout
NMPC_REF_DQ_SLICE = slice(0, 8)
NMPC_REF_OMEGA_SLICE = slice(8, 11)
NMPC_REF_VEL_SLICE = slice(11, 14)
NMPC_REF_UNOM_SLICE = slice(14, 18)


# Ordered control input layout — defines u[0], u[1], u[2], u[3] semantics.
# Must be kept in sync with ControlCommand.to_array() order and
# dynamics.py CasADi symbol ordering.
CONTROL_INPUT: tuple[str, ...] = (
    "thrust",
    "tau_x",
    "tau_y",
    "tau_z",
)

CONTROL_SYM_NAMES: tuple[str, ...] = (
    "F_ref",
    "tau_1_ref",
    "tau_2_ref",
    "tau_3_ref",
)


def control_index(name: str) -> int:
    """Return 0-based index of a control channel name in CONTROL_INPUT."""
    return CONTROL_INPUT.index(name)


def control_name(idx: int) -> str:
    """Return the CasADi symbol name for the control channel at index idx."""
    return CONTROL_SYM_NAMES[idx]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PhysicsParams(BaseModel):
    """Physical parameters shared across NMPC config and trajectory config."""

    model_config = ConfigDict(frozen=True)

    mass: float = Field(gt=0.0, description="Mass [kg]")
    gravity: float = Field(default=9.80665, gt=0.0, description="Gravity [m/s^2]")
    ixx: float = Field(gt=0.0, description="Inertia about body X [kg·m^2]")
    iyy: float = Field(gt=0.0, description="Inertia about body Y [kg·m^2]")
    izz: float = Field(gt=0.0, description="Inertia about body Z [kg·m^2]")


class OCPParams(BaseModel):
    """OCP solver hyperparameters.

    Every fixed-size array uses list[float] with exact max_length.
    Cross-field validation ensures Q/Q_e match nx and R/lbu/ubu match nu.
    """

    model_config = ConfigDict(frozen=True)

    Q: list[float] = Field(
        min_length=12,
        max_length=12,
        description="12D state-cost weights [rot(3), trans(3), angvel(3), vel(3)]",
    )
    Q_e: list[float] = Field(
        min_length=12, max_length=12, description="12D terminal state-cost weights"
    )
    R: list[float] = Field(min_length=4, max_length=4)
    nx: int = Field(default=14, gt=0)
    nu: int = Field(default=4, gt=0)
    lbu: list[float] = Field(min_length=4, max_length=4)
    ubu: list[float] = Field(min_length=4, max_length=4)
    horizon_steps: int = Field(default=10, gt=0)
    horizon_time: float = Field(default=1.0, gt=0.0)
    control_update_interval: float = Field(
        default=0.01, gt=0.0, description="MPC control-loop period [s]"
    )

    @model_validator(mode="after")
    def _check_lengths_match_dims(self):
        if len(self.Q) != 12:
            raise ValueError(f"len(Q)={len(self.Q)} must equal 12 (state-cost dimension)")
        if len(self.Q_e) != 12:
            raise ValueError(f"len(Q_e)={len(self.Q_e)} must equal 12")
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

    mav_name: str = Field(default="quadrotor")
    physics: PhysicsParams
    ocp: OCPParams

    @classmethod
    def from_yaml(cls, path: str | Path) -> NMPCConfig:
        """Load and validate configuration from a YAML file."""
        with open(path, "r") as stream:
            raw = yaml.safe_load(stream)
        return cls(**raw)


class ControlCommand(BaseModel):
    """4D control: body-frame thrust + torques."""

    model_config = ConfigDict(frozen=True)

    thrust: float = Field(default=0.0, ge=0.0, description="Thrust [N]")
    torque_x: float = Field(default=0.0, description="Torque about body x [N·m]")
    torque_y: float = Field(default=0.0, description="Torque about body y [N·m]")
    torque_z: float = Field(default=0.0, description="Torque about body z [N·m]")

    def to_array(self) -> np.ndarray:
        """Return (4,) numpy array ordered per CONTROL_INPUT."""
        return np.array(
            [self.thrust, self.torque_x, self.torque_y, self.torque_z],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ControlCommand:
        """Construct from (4,) array ordered per CONTROL_INPUT."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        _i = control_index
        return cls(
            thrust=arr[_i("thrust")],
            torque_x=arr[_i("tau_x")],
            torque_y=arr[_i("tau_y")],
            torque_z=arr[_i("tau_z")],
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
    """Single NMPC reference point — 18D array for one OCP stage parameter.

    Layout: [dq(8) | omega(3) | vel_body(3) | u_nom(4)]
    - dq[0:4]:    real quaternion [qw, qx, qy, qz]   (identity = [1,0,0,0])
    - dq[4:8]:    dual part [dw, dx, dy, dz]          (encodes world position)
    - omega(3):   body-frame angular velocity [rad/s]
    - vel_body(3): body-frame linear velocity [m/s]
    - u_nom(4):   nominal control [thrust(N), tau_x, tau_y, tau_z] [N·m]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dq: np.ndarray = Field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    omega: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vel_body: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    u_nom: np.ndarray = Field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_ndarray(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)

    def to_array(self) -> np.ndarray:
        """Return (18,) numpy array [dq(8), omega(3), vel_body(3), u_nom(4)]."""
        return np.concatenate([self.dq, self.omega, self.vel_body, self.u_nom])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> TrajectoryPoint:
        """Construct from (18,) array."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        return cls(
            dq=arr[0:8],
            omega=arr[8:11],
            vel_body=arr[11:14],
            u_nom=arr[14:18],
        )


class RefTrajBelt(BaseModel):
    """Horizon-window reference for a single NMPC solve — (N, 18) array.

    N = horizon_steps (number of shooting-interval reference nodes).
    Each row k corresponds to acados stage k: ``solver.set(k, "p", belt[k])``.
    Mutable (frozen=False) so consumer can update points in-place.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    points: np.ndarray  # (horizon_steps, 18)
    horizon_steps: int = Field(gt=0)

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_belt(cls, v: object) -> np.ndarray | int:
        if isinstance(v, (list, np.ndarray)):
            return np.asarray(v, dtype=np.float64)
        return v

    @model_validator(mode="after")
    def _check_shape(self) -> RefTrajBelt:
        if self.points.ndim != 2 or self.points.shape != (self.horizon_steps, 18):
            raise ValueError(f"points shape {self.points.shape} != ({self.horizon_steps}, 18)")
        return self

    def to_array(self) -> np.ndarray:
        """Return (horizon_steps, 18) numpy array."""
        return self.points

    @classmethod
    def from_array(cls, arr: np.ndarray) -> RefTrajBelt:
        """Construct from (N, 18) array."""
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        return cls(points=arr, horizon_steps=arr.shape[0])

    def __getitem__(self, k: int) -> TrajectoryPoint:
        """Return k-th reference point as TrajectoryPoint."""
        return TrajectoryPoint.from_array(self.points[k])

    def __len__(self) -> int:
        return int(self.horizon_steps)


class RefTrajectoryAsBelts(BaseModel):
    """Full-length belt set: N_c belts, each (N, 18), for every control step.

    Shape: belts[N_c, N, 18]
    - N_c: total number of MPC control update steps (= len(full trajectory))
    - N:   horizon_steps (= acados dims.N)
    - 18:  TrajectoryPoint.to_array() layout

    Mutable (frozen=False) — consumer may slice or update belts in-place.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    belts: np.ndarray  # (N_c, N, 18)
    N_c: int = Field(gt=0, description="Total control steps")
    horizon_steps: int = Field(gt=0, description="Steps per belt (acados dims.N)")

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_belts(cls, v: object) -> np.ndarray | int:
        if isinstance(v, (list, np.ndarray)):
            return np.asarray(v, dtype=np.float64)
        return v

    @model_validator(mode="after")
    def _check_shape(self) -> RefTrajectoryAsBelts:
        if self.belts.ndim != 3 or self.belts.shape != (self.N_c, self.horizon_steps, 18):
            raise ValueError(
                f"belts shape {self.belts.shape} != ({self.N_c}, {self.horizon_steps}, 18)"
            )
        return self

    def __getitem__(self, k: int) -> RefTrajBelt:
        """Return k-th belt as RefTrajBelt."""
        return RefTrajBelt(points=self.belts[k], horizon_steps=self.horizon_steps)

    def __len__(self) -> int:
        return int(self.N_c)


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
    control_update_interval: float = Field(gt=0.0, description="MPC control-loop period [s]")
    num_waypoints: int = Field(default=10, gt=1, description="Number of intermediate waypoints")
    sfc_half_extents: list[float] = Field(
        default=[0.5, 0.5, 0.5],
        min_length=3,
        max_length=3,
        description="SFC box half-extents (x, y, z) [m]",
    )

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
        from dq_nmpc.minco_trajectory import SHAPES

        if v not in SHAPES:
            raise ValueError(f"Unknown shape '{v}'. Choose from: {SHAPES}")
        return v

    def _shape_dir(self) -> Path:
        return Path(self.base_dir) / self.shape

    def _mkdir(self) -> None:
        self._shape_dir().mkdir(parents=True, exist_ok=True)

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

    @property
    def nmpc_rrd(self) -> Path:
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.base_dir) / "nmpc_tracking.rrd"

    @classmethod
    def from_trajectory_config(cls, tc: TrajectoryConfig) -> OutputPaths:
        return cls(shape=tc.shape)
