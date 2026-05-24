"""Waypoint generation and SFC construction for trajectory planning."""

import numpy as np

SHAPES: tuple[str, ...] = ("circle", "line", "fig8")


def make_sfc_box(
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> np.ndarray:
    """Create a safe-flight corridor as an axis-aligned box.

    Returns (M, 4) half-space matrix where each row [A,B,C,D] encodes
    Ax + By + Cz + D <= 0.
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents
    A = np.array(
        [
            [1, 0, 0, -(cx + hx)],
            [-1, 0, 0, cx - hx],
            [0, 1, 0, -(cy + hy)],
            [0, -1, 0, cy - hy],
            [0, 0, 1, -(cz + hz)],
            [0, 0, -1, cz - hz],
        ],
        dtype=np.float64,
    )
    return A


def waypoints_for_shape(shape: str, num_waypoints: int = 10) -> np.ndarray:
    if shape == "line":
        start = np.array([0.0, 0.0, 1.5])
        end = np.array([5.0, 0.0, 1.5])
        return np.linspace(start, end, num_waypoints - 1).T
    elif shape == "circle":
        angles = np.linspace(0, 2 * np.pi, num_waypoints - 1, endpoint=False)
        radius = 2.0
        z = 2.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.vstack([x, y, np.full_like(x, z)])
    elif shape == "fig8":
        t = np.linspace(0, 2 * np.pi, num_waypoints - 1)
        a = 2.0
        x = a * np.sin(t)
        y = a * np.sin(t) * np.cos(t)
        z = np.full_like(x, 2.0)
        return np.vstack([x, y, z])
    else:
        raise ValueError(f"Unknown shape: {shape}")
