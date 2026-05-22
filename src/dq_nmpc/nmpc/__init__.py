"""NMPC solver subpackage. Requires acados to be installed."""

try:
    from dq_nmpc.math.dq_functions import (  # noqa: F401
        dualquat_from_pose_ca_func,
        inertial_to_body_rotation_ca_func,
    )
    from dq_nmpc.nmpc.dynamics import export_acados_model  # noqa: F401
    from dq_nmpc.nmpc.ocp_setup import solver  # noqa: F401
    from dq_nmpc.nmpc.planner import get_flatness_trajectory  # noqa: F401
except ImportError:
    pass
