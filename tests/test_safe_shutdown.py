"""Tests for the arm safe-park sequence run at follower shutdown.

Covers the failure mode in Buzzy_Toolbox/docs/dexmate_arm_shutdown_joint_drift.md:
arms abandoned in position mode fall under gravity once the controller watchdog
times out, settling R_arm_j6 (array index 5) past its lower limit.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Loaded straight from its path rather than as omniteleop.follower.safe_shutdown:
# importing the package would pull in command_processor -> dexcomm and require
# the full hardware stack. The module under test only needs numpy and loguru.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "omniteleop"
    / "follower"
    / "safe_shutdown.py"
)
_spec = importlib.util.spec_from_file_location("safe_shutdown_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_safe_shutdown = importlib.util.module_from_spec(_spec)
# Registered before exec so @dataclass can resolve the module during creation.
sys.modules[_spec.name] = _safe_shutdown
_spec.loader.exec_module(_safe_shutdown)
park_arms_safely = _safe_shutdown.park_arms_safely

# Compressed timings so the tests stay fast; the production defaults are longer.
FAST = {
    "control_rate": 200.0,
    "hold_seconds": 0.1,
    "dwell_seconds": 0.05,
    "settle_timeout_seconds": 3.0,
}


class FakeArm:
    """Minimal stand-in for dexcontrol.core.arm.Arm."""

    def __init__(
        self,
        pos,
        *,
        brake_release_enabled: bool = False,
        drift_after_disable: float = 0.0,
        velocity: float = 0.0,
        settles: bool = True,
        fail_disable: bool = False,
        reject_disable: bool = False,
        brake_status_delay: float = 0.0,
    ) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.full(self.pos.shape, velocity, dtype=float)
        self.commands: list[np.ndarray] = []
        self.modes = ["position"] * 7
        self.brake_release_enabled = brake_release_enabled
        self.brake_calls: list[tuple[bool, object]] = []
        self.drift_after_disable = drift_after_disable
        self.settles = settles
        self.fail_disable = fail_disable
        self.reject_disable = reject_disable
        self.brake_status_delay = brake_status_delay
        self.disabled = False
        self.disabled_at: float | None = None

    def get_joint_pos(self):
        return self.pos.copy()

    def get_joint_vel(self):
        return self.vel.copy()

    def set_joint_pos(self, pos, wait_time: float = 0.0):
        assert not self.disabled, "published a position command after disable"
        self.commands.append(np.asarray(pos, dtype=float).copy())
        if self.settles:
            self.vel[:] = 0.0

    def set_modes(self, modes):
        if self.fail_disable:
            raise RuntimeError("mode service unavailable")
        if self.reject_disable:
            return {"success": False, "message": "mode request timed out"}
        self.modes = list(modes)
        self.disabled = modes[0] == "disable"
        if self.disabled:
            self.disabled_at = time.monotonic()
            self.pos = self.pos + self.drift_after_disable
        return {"success": True}

    def get_brake_status(self):
        time.sleep(self.brake_status_delay)
        return {"success": True, "enabled": self.brake_release_enabled, "joints": []}

    def release_brake(self, enable: bool, joints=None):
        self.brake_calls.append((enable, joints))
        self.brake_release_enabled = enable
        return {"success": True, "message": "Release brake disabled"}


class FakeEStop:
    def __init__(self, active: bool = False) -> None:
        self.active = active

    def is_software_estop_enabled(self) -> bool:
        return self.active


class FakeRobot:
    def __init__(self, left: FakeArm, right: FakeArm, estop_active: bool = False) -> None:
        self.left_arm = left
        self.right_arm = right
        self.estop = FakeEStop(estop_active)


def test_park_holds_measured_pose_then_engages_brakes() -> None:
    left = FakeArm(np.zeros(7))
    right = FakeArm([0.0, 0.0, 0.0, 0.0, 0.06, -1.30, 0.0])

    reports = park_arms_safely(FakeRobot(left, right), **FAST)

    # A single publish is not enough: wait_time=0 only holds under a fast loop.
    assert len(right.commands) > 5
    assert np.allclose(right.commands[0], [0.0, 0.0, 0.0, 0.0, 0.06, -1.30, 0.0])
    # disable mode keeps the mechanical brake engaged after the client exits.
    assert right.modes == ["disable"] * 7
    assert left.modes == ["disable"] * 7
    assert all(report.ok for report in reports.values())


def test_park_skips_the_slow_release_brake_call_when_already_disabled() -> None:
    """release_brake() can block for 45 s, which would overrun the stop grace."""
    left = FakeArm(np.zeros(7))
    right = FakeArm(np.zeros(7))

    park_arms_safely(FakeRobot(left, right), **FAST)

    assert right.brake_calls == []
    assert left.brake_calls == []


def test_park_disables_brake_release_when_it_was_left_enabled() -> None:
    left = FakeArm(np.zeros(7))
    right = FakeArm(np.zeros(7), brake_release_enabled=True)

    reports = park_arms_safely(FakeRobot(left, right), **FAST)

    assert right.brake_calls == [(False, None)]
    assert reports["right_arm"].brake_release_was_enabled is True
    assert reports["right_arm"].brake_release_disabled is True


def test_park_reports_drift_after_the_brake_transition() -> None:
    left = FakeArm(np.zeros(7))
    right = FakeArm(np.zeros(7), drift_after_disable=0.13)

    reports = park_arms_safely(FakeRobot(left, right), **FAST)

    assert reports["right_arm"].max_drift_rad == pytest.approx(0.13)
    assert not reports["right_arm"].ok
    assert any("drift" in warning for warning in reports["right_arm"].warnings)


def test_brake_transition_is_parallel_across_both_arms() -> None:
    """A slow service on one arm must not leave the other arm waiting unheld."""
    left = FakeArm(np.zeros(7), brake_status_delay=0.2)
    right = FakeArm(np.zeros(7), brake_status_delay=0.2)

    park_arms_safely(FakeRobot(left, right), **FAST)

    assert left.disabled_at is not None and right.disabled_at is not None
    assert abs(left.disabled_at - right.disabled_at) < 0.05
    assert len(left.commands) > 5 and len(right.commands) > 5


def test_park_never_raises_when_the_mode_service_fails() -> None:
    left = FakeArm(np.zeros(7))
    right = FakeArm(np.zeros(7), fail_disable=True)

    reports = park_arms_safely(FakeRobot(left, right), **FAST)

    assert reports["right_arm"].disabled is False
    assert any("set_modes" in w for w in reports["right_arm"].warnings)
    assert reports["left_arm"].disabled is True


def test_park_does_not_treat_a_failed_mode_response_as_disabled() -> None:
    left = FakeArm(np.zeros(7))
    right = FakeArm(np.zeros(7), reject_disable=True)

    reports = park_arms_safely(FakeRobot(left, right), **FAST)

    assert reports["right_arm"].disabled is False
    assert any("mode request timed out" in w for w in reports["right_arm"].warnings)


def test_park_skips_holding_but_still_brakes_under_software_estop() -> None:
    left = FakeArm(np.zeros(7))
    right = FakeArm(np.zeros(7))

    reports = park_arms_safely(FakeRobot(left, right, estop_active=True), **FAST)

    assert right.commands == []
    assert right.modes == ["disable"] * 7
    assert any("E-Stop" in w for w in reports["right_arm"].warnings)


def test_park_tolerates_a_missing_arm() -> None:
    class OneArmRobot:
        def __init__(self) -> None:
            self.right_arm = FakeArm(np.zeros(7))
            self.estop = FakeEStop()

    reports = park_arms_safely(OneArmRobot(), **FAST)

    assert reports["left_arm"].attempted is False
    assert reports["right_arm"].disabled is True


def test_park_bounds_the_settle_phase_without_starving_either_arm() -> None:
    """An arm that never settles must not starve the other arm's hold."""
    left = FakeArm(np.zeros(7), velocity=5.0, settles=False)
    right = FakeArm(np.zeros(7), velocity=5.0, settles=False)

    started = time.monotonic()
    reports = park_arms_safely(
        FakeRobot(left, right),
        control_rate=200.0,
        hold_seconds=0.1,
        dwell_seconds=0.05,
        settle_timeout_seconds=1.0,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.6
    assert len(left.commands) > 5 and len(right.commands) > 5
    assert any("settle" in w for w in reports["left_arm"].warnings)
    assert right.modes == ["disable"] * 7
