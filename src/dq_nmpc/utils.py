"""Re-export shim — imports moved to dq_nmpc.math.casadi_helpers."""

from dq_nmpc.math.casadi_helpers import (  # noqa: F401
    calc_quat_cost,
    calc_vec_cost,
    conjugate_quaternion,
    multiply_quaternions,
    normalize_quaternion,
    rotate_vector_by_quaternion,
    yaml_to_dict,
)
