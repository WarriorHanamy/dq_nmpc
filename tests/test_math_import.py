"""Basic import and smoke tests for the math module."""

import numpy as np


def test_import_dualquat_from_pose_np():
    from dq_nmpc.math.dq_functions import dualquat_from_pose_np

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    pos = np.array([1.0, 2.0, 3.0])
    dq = dualquat_from_pose_np(quat, pos)
    assert dq.shape == (8,)
    assert dq[0] == 1.0 and dq[1] == 0.0 and dq[2] == 0.0 and dq[3] == 0.0
    assert abs(dq[5] - 0.5) < 0.01  # 0.5 * ty


def test_import_casadi_dq_from_pose():
    from dq_nmpc.math.dq_functions import make_dualquat_from_pose

    fn = make_dualquat_from_pose()
    result = fn(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    assert float(result[0]) == 1.0  # qw
    assert abs(float(result[5]) - 0.5) < 0.01  # dual y


def test_import_via_package():
    from dq_nmpc.math.dq_functions import dualquat_from_pose_np, make_dualquat_from_pose

    assert callable(make_dualquat_from_pose)
    assert callable(dualquat_from_pose_np)
