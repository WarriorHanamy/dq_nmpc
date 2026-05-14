"""Type aliases for domain-specific vector spaces and numeric backends.

Quaternion     — S^3, shape 4   [qw, qx, qy, qz]
Twist          — se(3), shape 6  [wx, wy, wz, vx, vy, vz]
DualQuaternion — SE(3), shape 8  [qw,qx,qy,qz, dw,dx,dy,dz]
Vec3           — R^3, shape 3   [x, y, z]
VecN           — arbitrary-length vector
"""

from typing import Union

import casadi as cs
import numpy as np

Scalar = Union[float, int, cs.MX, cs.SX]
"""A single numeric value: float, int, or CasADi symbolic scalar."""

Vector = Union[np.ndarray, cs.MX, cs.SX]
"""Backward compat — prefer Quaternion / Twist / DualQuaternion / Vec3 / VecN."""

VecN = Union[np.ndarray, cs.MX, cs.SX]
"""Arbitrary-length numeric vector (e.g. cost weights)."""

Vec3 = Union[np.ndarray, cs.MX, cs.SX]
"""Element of R^3 — position, linear velocity, angular velocity."""

Quaternion = Union[np.ndarray, cs.MX, cs.SX]
"""Element of S^3 — quaternion [qw, qx, qy, qz]."""

Twist = Union[np.ndarray, cs.MX, cs.SX]
"""Element of se(3) — body-frame twist [wx, wy, wz, vx, vy, vz]."""

DualQuaternion = Union[np.ndarray, cs.MX, cs.SX]
"""Element of SE(3) — dual quaternion [qw,qx,qy,qz, dw,dx,dy,dz]."""
