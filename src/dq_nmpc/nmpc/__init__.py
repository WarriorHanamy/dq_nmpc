"""NMPC solver subpackage. Requires acados to be installed."""

try:
    from dq_nmpc.math.dq_algebra import (  # noqa: F401
        adjoint_map,
        dualquat_conjugate,
        dualquat_mul_conj,
        log_error_dualquat,
        log_map_dualquat,
    )
    from dq_nmpc.math.dq_functions import (  # noqa: F401
        make_dualquat_from_pose,
        make_dualquat_mul_conj,
        make_quat_error_cost,
        make_translation_error_cost,
    )
    from dq_nmpc.nmpc.dynamics import (  # noqa: F401
        apply_noise,
        export_acados_model,
        make_body_to_inertial_rotation,
        make_body_velocity_from_twist,
        make_get_quaternion,
        make_get_translation,
        make_inertial_to_body_rotation,
        make_inertial_velocity_from_twist,
        make_quadrotor_model,
        rotate_vector_body_to_inertial,
    )
    from dq_nmpc.nmpc.ocp_setup import create_ocp_solver, solver  # noqa: F401
    from dq_nmpc.nmpc.planner import compute_flatness_states  # noqa: F401
except ImportError:
    pass
