"""Load minco NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_trajectory_npz(path: str | Path) -> Any:
    """Reconstruct a minco Trajectory7 from a .npz coefficient file.

    @param[in] path  Path to trajectory.npz
    @return          minco.poly_traj.Trajectory7 instance
    """
    import minco

    data = np.load(path)
    durations = data["durations"].tolist()
    coeff_mats = [data["coeffs"][i] for i in range(len(durations))]
    return minco.poly_traj.Trajectory7(durations, coeff_mats)
