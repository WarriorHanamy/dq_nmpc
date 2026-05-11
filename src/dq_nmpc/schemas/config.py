"""NMPC configuration schema, validated from YAML parameters."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class NMPCParams(BaseModel):
    """NMPC solver hyperparameters."""

    Q: list[float] = Field(min_length=1)
    Q_e: list[float] = Field(min_length=1)
    R: list[float] = Field(min_length=1)
    nx: int = Field(default=13, gt=0)
    nu: int = Field(default=4, gt=0)
    lbu: list[float] = Field(min_length=1)
    ubu: list[float] = Field(min_length=1)
    horizon_steps: int = Field(default=10, gt=0)
    horizon_time: float = Field(default=1.0, gt=0.0)
    ts: float = Field(default=0.01, gt=0.0, description="Sample time [s]")


class NMPCConfig(BaseModel):
    """Validated NMPC configuration, loadable from YAML."""

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
