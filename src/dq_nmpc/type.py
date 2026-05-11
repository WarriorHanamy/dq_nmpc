"""Type aliases for dual quaternion math module."""

from typing import Union

import casadi as cs
import numpy as np

Scalar = Union[float, int, cs.MX, cs.SX]
"""A single numeric value: float, int, or CasADi symbolic scalar."""

Vector = Union[np.ndarray, cs.MX, cs.SX]
"""A column vector: NumPy array or CasADi symbolic vector."""
