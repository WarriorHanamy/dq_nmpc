"""NMPC solver subpackage. Requires acados to be installed."""

__all__ = [
    "dense_ref_from_minco",
    "export_acados_model",
    "make_flatness_casadi",
    "run_nmpc",
    "solver",
]

try:
    from dq_nmpc.nmpc.dynamics import export_acados_model  # noqa: F401
    from dq_nmpc.nmpc.ocp_setup import solver  # noqa: F401
except ImportError:
    pass

from dq_nmpc.nmpc.flatness import make_flatness_casadi
from dq_nmpc.nmpc.reference import dense_ref_from_minco
from dq_nmpc.nmpc.runner import run_nmpc
