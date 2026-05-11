"""NMPC solver subpackage. Requires acados to be installed."""

try:
    from dq_nmpc.nmpc.dynamics import (  # noqa: F401
        Ad,
        compute_flatness_states,
        conjugate_dual,
        cost_quaternion_casadi,
        cost_translation_casadi,
        dual_velocity_casadi,
        dualquat_quat_casadi,
        dualquat_trans_casadi,
        error_dual,
        error_dual_aux_casadi,
        error_lie,
        export_model,
        ln_dual,
        noise,
        quadrotorModel,
        rotation,
        rotation_casadi,
        rotation_inverse_casadi,
        velocities_from_twist_casadi,
    )
    from dq_nmpc.nmpc.functions import dualquat_from_pose_casadi  # noqa: F401
    from dq_nmpc.nmpc.ocp_setup import create_ocp_solver  # noqa: F401
    from dq_nmpc.nmpc.controller import solver  # noqa: F401
except ImportError:
    pass
