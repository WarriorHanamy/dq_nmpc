"""Smoke test for schema/ — import, defaults, frozen validation.

Usage:
    uv run python src/dq_nmpc/schema/test_smoke.py           # full
    uv run python src/dq_nmpc/schema/test_smoke.py --quick   # sub-second
"""

import sys

from pydantic import ValidationError


def _test_import():
    from dq_nmpc.schema import (
        ClassicalState,
        ControlCommand,
        DockerConfig,
        DualQuaternionState,
        NMPCConfig,
        NMPCParams,
        ReferenceTrajectory,
        SHMConfig,
        TrajectoryPoint,
    )

    return (
        ClassicalState,
        ControlCommand,
        DockerConfig,
        DualQuaternionState,
        NMPCConfig,
        NMPCParams,
        ReferenceTrajectory,
        SHMConfig,
        TrajectoryPoint,
    )


def test_default_frozen():
    """All models must reject mutation after construction."""
    ClassicalState, ControlCommand, _, _, _, _, _, _, _ = _test_import()

    s = ClassicalState()
    try:
        s.x = 5.0
        assert False, "ClassicalState should be frozen"
    except Exception:
        pass

    c = ControlCommand(thrust=1.0)
    try:
        c.thrust = 10.0
        assert False, "ControlCommand should be frozen"
    except Exception:
        pass


def test_validation_rejects_invalid():
    """Field-level validation must raise on invalid input."""
    _, ControlCommand, _, _, _, _, _, _, _ = _test_import()

    try:
        ControlCommand(thrust=-1.0)
        assert False, "should reject negative thrust"
    except ValidationError:
        pass


def test_shm_config_defaults():
    _, _, _, _, _, _, _, SHMConfig, _ = _test_import()
    cfg = SHMConfig.default()
    assert cfg.state_file == "/dev/shm/quadrotor_sim/state"
    assert cfg.ctrl_file == "/dev/shm/quadrotor_sim/ctrl"
    assert cfg.state_size == 192


def main():
    quick = "--quick" in sys.argv

    print("schema/test_smoke.py ... ", end="")
    _test_import()
    if quick:
        print("quick OK")
        return 0

    test_default_frozen()
    test_validation_rejects_invalid()
    test_shm_config_defaults()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
