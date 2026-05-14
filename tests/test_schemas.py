"""Tests for Pydantic schema models."""

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
    def test_arrays(self):
        tp = TrajectoryPoint(
            x=1.0,
            y=2.0,
            z=3.0,
            thrust=5.0,
            qw=1.0,
            torque_z=0.5,
        )
        state_arr = tp.state_as_array()
        ctrl_arr = tp.control_as_array()
        assert state_arr.shape == (13,)
        assert state_arr[0] == 1.0
        assert ctrl_arr.shape == (4,)
        assert ctrl_arr[0] == 5.0
        assert ctrl_arr[3] == 0.5


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
