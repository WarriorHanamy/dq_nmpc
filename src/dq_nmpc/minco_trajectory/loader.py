"""Load minco NPZ files and validate trajectory metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dq_nmpc.schema import NMPCConfig


def _parse_csv_meta(path: Path) -> dict[str, str]:
    """Read '# key=value' comment lines from top of CSV."""
    meta: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                stripped = stripped[1:].strip()
                if "=" in stripped:
                    key, _, val = stripped.partition("=")
                    meta[key.strip()] = val.strip()
            else:
                break
    return meta


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


def load_trajectory_meta(path: str | Path) -> dict[str, str]:
    """Parse metadata from trajectory CSV comment header."""
    return _parse_csv_meta(Path(path))


def validate_trajectory_ts(csv_path: str | Path, config: NMPCConfig) -> None:
    """Verify trajectory control update interval matches NMPC config."""
    meta = _parse_csv_meta(Path(csv_path))
    dt_csv = float(meta.get("control_update_interval", 0.0))
    dt_nmpc = config.ocp.control_update_interval
    if abs(dt_csv - dt_nmpc) > 1e-6:
        raise ValueError(
            f"control_update_interval mismatch: trajectory={dt_csv} vs nmpc config={dt_nmpc}"
        )
