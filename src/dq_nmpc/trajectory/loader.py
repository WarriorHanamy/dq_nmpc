"""Load trajectory CSV into ReferenceTrajectory for NMPC consumption."""

from __future__ import annotations

import csv
from pathlib import Path

from dq_nmpc.schema import TRAJECTORY_CSV_COLUMNS, ReferenceTrajectory, TrajectoryPoint


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
