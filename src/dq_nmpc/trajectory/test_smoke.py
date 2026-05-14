"""Smoke test for trajectory/ — import loader, basic roundtrip.

Usage:
    uv run python src/dq_nmpc/trajectory/test_smoke.py           # full
    uv run python src/dq_nmpc/trajectory/test_smoke.py --quick   # sub-second
"""

import sys
import tempfile
from pathlib import Path


def _test_import():
    from dq_nmpc.trajectory.loader import load_trajectory_csv

    return load_trajectory_csv


def test_loader_import():
    load_trajectory_csv = _test_import()
    assert load_trajectory_csv is not None
    return True


def test_loader_csv_roundtrip():
    """Write a minimal CSV and load it back."""
    import csv

    load_trajectory_csv = _test_import()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "qw",
                "qx",
                "qy",
                "qz",
                "wx",
                "wy",
                "wz",
                "thrust",
            ]
        )
        writer.writerow(
            ["0.0", "0", "0", "2", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "10.0"]
        )
        tmp_path = Path(f.name)

    try:
        traj = load_trajectory_csv(tmp_path)
        assert len(traj.points) == 1
        assert traj.points[0].z == 2.0
        assert traj.points[0].thrust == 10.0
    finally:
        tmp_path.unlink()

    return True


def main():
    quick = "--quick" in sys.argv

    print("trajectory/test_smoke.py ... ", end="")
    _test_import()
    if quick:
        print("quick OK")
        return 0

    test_loader_import()
    test_loader_csv_roundtrip()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
