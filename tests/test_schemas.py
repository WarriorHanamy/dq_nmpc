"""Tests for Pydantic schema models."""

import numpy as np
import pytest
from pydantic import ValidationError

from dq_nmpc.schema import (
    ClassicalState,
    ControlCommand,
    DualQuaternionState,
    SHMConfig,
    TrajectoryPoint,
)


class TestDualQuaternionState:
    def test_default(self):
        s = DualQuaternionState()
        assert s.qw == 1.0
        arr = s.to_array()
        assert arr.shape == (14,)
        assert arr[0] == 1.0

    def test_roundtrip(self):
        orig = DualQuaternionState(
            qw=0.5,
            qx=0.5,
            qy=0.5,
            qz=0.5,
            dw=0.1,
            dx=0.2,
            dy=0.3,
            dz=0.4,
            wx=1.0,
            wy=2.0,
            wz=3.0,
            vx=0.1,
            vy=0.2,
            vz=0.3,
        )
        arr = orig.to_array()
        restored = DualQuaternionState.from_array(arr)
        assert restored.qw == pytest.approx(0.5)
        assert restored.vz == pytest.approx(0.3)

    def test_split_dual(self):
        s = DualQuaternionState(qw=1.0, dx=1.0)
        dual, twist = s.split_dual()
        assert dual.shape == (8,)
        assert twist.shape == (6,)
        assert dual[0] == 1.0
        assert dual[5] == 1.0  # dx


class TestClassicalState:
    def test_default(self):
        s = ClassicalState()
        assert s.qw == 1.0
        assert s.x == 0.0

    def test_roundtrip(self):
        orig = ClassicalState(
            x=1.0,
            y=2.0,
            z=3.0,
            vx=0.1,
            vy=0.2,
            vz=0.3,
            qw=0.707,
            qx=0.707,
            qy=0.0,
            qz=0.0,
            wx=0.5,
            wy=0.6,
            wz=0.7,
        )
        arr = orig.to_array()
        assert arr.shape == (13,)
        restored = ClassicalState.from_array(arr)
        assert restored.x == pytest.approx(1.0)
        assert restored.qw == pytest.approx(0.707)
        assert restored.wz == pytest.approx(0.7)


class TestControlCommand:
    def test_roundtrip(self):
        c = ControlCommand(thrust=5.0, torque_x=0.1, torque_y=0.2, torque_z=0.3)
        arr = c.to_array()
        assert arr.shape == (4,)
        restored = ControlCommand.from_array(arr)
        assert restored.thrust == pytest.approx(5.0)
        assert restored.torque_z == pytest.approx(0.3)

    def test_nonnegative_thrust(self):
        with pytest.raises(ValidationError):
            ControlCommand(thrust=-1.0)

    def test_frozen_mutation_rejected(self):
        c = ControlCommand(thrust=5.0)
        with pytest.raises(Exception):
            c.thrust = 10.0


class TestTrajectoryPoint:
    def test_defaults(self):
        tp = TrajectoryPoint()
        arr = tp.to_array()
        assert arr.shape == (18,)
        assert arr[0] == 1.0  # qw=1 in identity DQ
        assert arr[8] == 0.0  # omega=0

    def test_roundtrip(self):
        import numpy as np

        orig = TrajectoryPoint(
            dq=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5]),
            omega=np.array([0.1, 0.2, 0.3]),
            vel_body=np.array([1.0, 2.0, 3.0]),
            u_nom=np.array([5.0, 0.1, 0.2, 0.3]),
        )
        arr = orig.to_array()
        assert arr.shape == (18,)
        restored = TrajectoryPoint.from_array(arr)
        assert np.allclose(restored.dq, orig.dq)
        assert np.allclose(restored.omega, orig.omega)
        assert np.allclose(restored.vel_body, orig.vel_body)
        assert np.allclose(restored.u_nom, orig.u_nom)

    def test_frozen_assignment_rejected(self):
        tp = TrajectoryPoint()
        with pytest.raises(Exception):
            tp.dq = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # attribute reassignment rejected


class TestSHMConfig:
    def test_defaults(self):
        cfg = SHMConfig()
        assert cfg.state_file == "/dev/shm/quadrotor_sim/state"
        assert cfg.ctrl_file == "/dev/shm/quadrotor_sim/ctrl"
        assert cfg.state_size == 192
        assert cfg.ctrl_size == 64

    def test_default_classmethod(self):
        cfg = SHMConfig.default()
        assert cfg.state_file == "/dev/shm/quadrotor_sim/state"
