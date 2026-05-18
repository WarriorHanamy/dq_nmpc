"""NMPC solver subpackage. Requires acados to be installed."""

try:
    from dq_nmpc.nmpc.controller import solver  # noqa: F401
    from dq_nmpc.nmpc.dynamics import (  # noqa: F401
        adjoint_map,
        apply_noise,
        compute_flatness_states,
        dualquat_conjugate,
        dualquat_mul_conj,
        export_acados_model,
        log_error_dualquat,
        log_map_dualquat,
        make_body_to_inertial_rotation,
        make_body_velocity_from_twist,
        make_dualquat_mul_conj,
        make_get_quaternion,
        make_get_translation,
        make_inertial_to_body_rotation,
        make_inertial_velocity_from_twist,
        make_quadrotor_model,
        make_quat_error_cost,
        make_translation_error_cost,
        rotate_vector_body_to_inertial,
    )
    from dq_nmpc.nmpc.functions import dualquat_from_pose_casadi  # noqa: F401
    from dq_nmpc.nmpc.ocp_setup import create_ocp_solver  # noqa: F401
except ImportError:
    pass
