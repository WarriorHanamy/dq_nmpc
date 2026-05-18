"""Load trajectory CSV into ReferenceTrajectory for NMPC consumption."""

from __future__ import annotations

import csv
from pathlib import Path

from dq_nmpc.schema import TRAJECTORY_CSV_COLUMNS, NMPCConfig, ReferenceTrajectory, TrajectoryPoint


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


def load_trajectory_csv(path: str | Path) -> ReferenceTrajectory:
    path = Path(path)
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None:
            missing = set(TRAJECTORY_CSV_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV header missing columns: {sorted(missing)}. "
                    f"Expected: {list(TRAJECTORY_CSV_COLUMNS)}"
                )
        for row in reader:
            tp = TrajectoryPoint(
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                vz=float(row["vz"]),
                qw=float(row["qw"]),
                qx=float(row["qx"]),
                qy=float(row["qy"]),
                qz=float(row["qz"]),
                wx=float(row["wx"]),
                wy=float(row["wy"]),
                wz=float(row["wz"]),
                thrust=float(row["thrust"]),
            )
            points.append(tp)
    return ReferenceTrajectory(points=points, horizon_steps=len(points))


def load_trajectory_meta(path: str | Path) -> dict[str, str]:
    """Parse metadata from trajectory CSV comment header."""
    return _parse_csv_meta(Path(path))


def validate_trajectory_ts(csv_path: str | Path, config: NMPCConfig) -> None:
    """Verify trajectory sample time matches NMPC config."""
    meta = _parse_csv_meta(Path(csv_path))
    ts_csv = float(meta.get("ts", 0.0))
    ts_nmpc = config.nmpc.ts
    if abs(ts_csv - ts_nmpc) > 1e-6:
        raise ValueError(f"ts mismatch: trajectory={ts_csv} vs nmpc config={ts_nmpc}")
