"""NMPC solver: DQ math, flatness, OCP, runtime."""

from ._dq_functions import dualquat_from_pose_ca_func
from ._flatness import make_flatness_casadi
from ._ref_visualizer import visualize_ref_params
from ._reference import belts_from_dense, dense_ref_from_minco

__all__ = [
    "belts_from_dense",
    "dense_ref_from_minco",
    "dualquat_from_pose_ca_func",
    "export_acados_model",
    "make_flatness_casadi",
    "run_nmpc",
    "solver",
    "visualize_ref_params",
]


def __getattr__(name: str):
    if name == "solver":
        from ._ocp_setup import solver

        return solver
    if name == "export_acados_model":
        from ._dynamics import export_acados_model

        return export_acados_model
    if name == "run_nmpc":
        from ._runner import run_nmpc

        return run_nmpc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
