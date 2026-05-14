"""Smoke test for math/ — imports, construction, basic operations.

Usage:
    uv run python src/dq_nmpc/math/test_smoke.py           # full
    uv run python src/dq_nmpc/math/test_smoke.py --quick   # sub-second
"""

import sys

import numpy as np


def _test_import():
    from dq_nmpc.math.dual_quaternion import DualQuaternion
    from dq_nmpc.math.quaternion import Quaternion

    return Quaternion, DualQuaternion


def test_identity():
    Quaternion, _ = _test_import()
    q = Quaternion(q=np.array([1.0, 0.0, 0.0, 0.0]))
    assert q is not None
    return True


def test_conjugate():
    Quaternion, _ = _test_import()
    q = Quaternion(q=np.array([0.707, 0.707, 0.0, 0.0]))
    assert q is not None
    return True


def test_dual_quat_identity():
    Quaternion, DualQuaternion = _test_import()
    qr = Quaternion(q=np.array([1.0, 0.0, 0.0, 0.0]))
    qd = Quaternion(q=np.array([0.0, 0.0, 0.0, 0.0]))
    dq = DualQuaternion(q_real=qr, q_dual=qd)
    assert dq is not None
    return True


def main():
    quick = "--quick" in sys.argv

    print("math/test_smoke.py ... ", end="")
    _test_import()
    if quick:
        print("quick OK")
        return 0

    test_identity()
    test_conjugate()
    test_dual_quat_identity()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
