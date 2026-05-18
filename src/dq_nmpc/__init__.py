"""Dual Quaternion NMPC for Quadrotor.

Subpackages:
    math       -- Pure quaternion/dual-quaternion math (numpy + casadi, no acados/ROS)
    schema     -- Single-source Pydantic models + layout constants
    core       -- Stateless primitives (paths, subprocess, SHM helpers)
    cli        -- Zero-logic argument dispatch
    workflows  -- Domain-specific multi-step pipelines
    nmpc       -- NMPC controller (requires acados)
    ros        -- ROS2 compatibility layer (optional)
"""

from dq_nmpc.math.dual_quaternion import DualQuaternion
from dq_nmpc.math.quaternion import Quaternion
from dq_nmpc.schema import (  # noqa: F401
    ClassicalState,
    ControlCommand,
    DualQuaternionState,
    NMPCConfig,
    NMPCParams,
    ReferenceTrajectory,
    SHMConfig,
    TrajectoryPoint,
)

__all__ = [
    "ClassicalState",
    "ControlCommand",
    "DualQuaternion",
    "DualQuaternionState",
    "NMPCConfig",
    "NMPCParams",
    "Quaternion",
    "ReferenceTrajectory",
    "SHMConfig",
    "TrajectoryPoint",
]
