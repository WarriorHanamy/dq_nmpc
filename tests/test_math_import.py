"""Basic import and smoke tests for the math module."""

import numpy as np


def test_dualquat_from_pose_ca_func_np_inputs():
    """Verify _ca_func works with numpy array elements."""
    from dq_nmpc.math.dq_functions import dualquat_from_pose_ca_func

    fn = dualquat_from_pose_ca_func()
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    tx, ty, tz = 1.0, 2.0, 3.0
    result = np.array(fn(qw, qx, qy, qz, tx, ty, tz)).ravel()
    assert result.shape == (8,)
    assert result[0] == 1.0 and result[1] == 0.0 and result[2] == 0.0 and result[3] == 0.0
    assert abs(result[5] - 0.5) < 0.01


def test_import_casadi_dq_from_pose():
    from dq_nmpc.math.dq_functions import dualquat_from_pose_ca_func

    fn = dualquat_from_pose_ca_func()
    result = fn(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    assert float(result[0]) == 1.0
    assert abs(float(result[5]) - 0.5) < 0.01


def test_import_via_package():
    from dq_nmpc.math.dq_functions import dualquat_from_pose_ca_func

    assert callable(dualquat_from_pose_ca_func)
