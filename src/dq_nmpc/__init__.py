"""Dual Quaternion NMPC for Quadrotor.

Subpackages:
    math   -- Pure quaternion/dual-quaternion math (numpy + casadi, no acados/ROS)
    schemas  -- Pydantic models for I/O data contracts
    nmpc   -- NMPC controller (requires acados)
    ros    -- ROS2 compatibility layer (optional)
"""

from dq_nmpc.math.dual_quaternion import DualQuaternion
from dq_nmpc.math.quaternion import Quaternion

__all__ = ["DualQuaternion", "Quaternion"]

# Lazy imports for optional acados-dependent modules.
# Import them explicitly when needed:
#   from dq_nmpc.nmpc import solver, export_model, ...
#   from dq_nmpc.nmpc.functions import dualquat_from_pose_casadi
