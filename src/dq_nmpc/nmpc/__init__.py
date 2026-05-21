"""NMPC solver subpackage. Requires acados to be installed."""

try:
    from dq_nmpc.math.dq_functions import (  # noqa: F401
        make_body_to_inertial_rotation,
        make_dualquat_from_pose,
        make_dualquat_mul_conj,
        make_inertial_to_body_rotation,
        make_quat_error_cost,
        make_translation_error_cost,
    )
    from dq_nmpc.nmpc.dynamics import (  # noqa: F401
        apply_noise,
        export_acados_model,
        make_body_velocity_from_twist,
        make_get_quaternion,
        make_get_translation,
        make_inertial_velocity_from_twist,
        make_quadrotor_model,
    )
    from dq_nmpc.nmpc.ocp_setup import solver  # noqa: F401
    from dq_nmpc.nmpc.planner import get_flatness_trajectory  # noqa: F401
except ImportError:
    pass
