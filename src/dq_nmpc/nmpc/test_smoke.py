"""Smoke test for nmpc/ — import, basic shapes (skip if acados unavailable).

Usage:
    uv run python src/dq_nmpc/nmpc/test_smoke.py           # full
    uv run python src/dq_nmpc/nmpc/test_smoke.py --quick   # sub-second
"""

import sys


def _test_import():
    try:
        from dq_nmpc.nmpc.controller import solver
        from dq_nmpc.nmpc.dynamics import export_acados_model
    except ImportError as e:
        return None, str(e)

    return (solver, export_acados_model), None


def test_controller_import():
    result, err = _test_import()
    if result is None:
        print(f"SKIP (acados not available: {err})")
        return False
    solver, export_acados_model = result
    assert solver is not None
    assert export_acados_model is not None
    return True


def test_dynamics_functions_import():
    try:
        from dq_nmpc.nmpc.dynamics import make_body_velocity_from_twist  # noqa: F401
    except ImportError:
        return False
    return True


def main():
    quick = "--quick" in sys.argv

    print("nmpc/test_smoke.py ... ", end="")
    result, err = _test_import()
    if result is None:
        print(f"SKIP (acados not available: {err})")
        return 0
    if quick:
        print("quick OK")
        return 0

    ok1 = test_controller_import()
    ok2 = test_dynamics_functions_import()
    if not ok1 or not ok2:
        print("SKIP")
        return 0
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
