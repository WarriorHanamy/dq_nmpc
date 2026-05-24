"""Trajectory generation (minco-python) and I/O (CSV)."""

from ._waypoints import SHAPES

__all__ = ["SHAPES", "generate_trajectory", "load_trajectory_npz"]


def __getattr__(name: str):
    if name == "generate_trajectory":
        from ._generator import generate_trajectory

        return generate_trajectory
    if name == "load_trajectory_npz":
        from ._loader import load_trajectory_npz

        return load_trajectory_npz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
