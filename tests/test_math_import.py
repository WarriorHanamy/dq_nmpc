"""Basic import and smoke tests for the math module."""



def test_import_quaternion():
    import numpy as np

    from dq_nmpc.math.quaternion import Quaternion

    q = Quaternion(q=np.array([1.0, 0.0, 0.0, 0.0]))
    assert q is not None


def test_import_dual_quaternion():
    import numpy as np

    from dq_nmpc.math.dual_quaternion import DualQuaternion
    from dq_nmpc.math.quaternion import Quaternion

    qr = Quaternion(q=np.array([1.0, 0.0, 0.0, 0.0]))
    qd = Quaternion(q=np.array([0.0, 0.0, 0.0, 0.0]))
    dq = DualQuaternion(q_real=qr, q_dual=qd)
    assert dq is not None


def test_import_via_package():
    import numpy as np

    from dq_nmpc import DualQuaternion, Quaternion

    qr = Quaternion(q=np.array([1.0, 0.0, 0.0, 0.0]))
    qd = Quaternion(q=np.array([0.0, 0.0, 0.0, 0.0]))
    dq = DualQuaternion(q_real=qr, q_dual=qd)
    assert dq is not None
