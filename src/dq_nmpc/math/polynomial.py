"""Polynomial basis functions for minimum-snap trajectory generation (order-9)."""

import numpy as np


def position_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack([1 * np.ones(t.shape), t, t**2, t**3, t**4, t**5, t**6, t**7, t**8, t**9])
    return vector


def velocity_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack(
        [
            0 * np.ones(t.shape),
            1 * np.ones(t.shape),
            2 * t,
            3 * t**2,
            4 * t**3,
            5 * t**4,
            6 * t**5,
            7 * t**6,
            8 * t**7,
            9 * t**8,
        ]
    )
    return vector


def acceleration_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack(
        [
            0 * np.ones(t.shape),
            0 * np.ones(t.shape),
            2 * np.ones(t.shape),
            6 * t,
            12 * t**2,
            20 * t**3,
            30 * t**4,
            42 * t**5,
            56 * t**6,
            72 * t**7,
        ]
    )
    return vector


def jerk_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack(
        [
            0 * np.ones(t.shape),
            0 * np.ones(t.shape),
            0 * np.ones(t.shape),
            6 * np.ones(t.shape),
            24 * t,
            60 * t**2,
            120 * t**3,
            210 * t**4,
            336 * t**5,
            504 * t**6,
        ]
    )
    return vector


def snap_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack(
        [
            0 * np.ones(t.shape),
            0 * np.ones(t.shape),
            0 * np.ones(t.shape),
            0 * np.ones(t.shape),
            24 * np.ones(t.shape),
            120 * t,
            360 * t**2,
            840 * t**3,
            1680 * t**4,
            3024 * t**5,
        ]
    )
    return vector
