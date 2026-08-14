from __future__ import annotations

import numpy as np
import pytest

from omniteleop.follower.robot_controller import RobotController


class FakeEStop:
    def __init__(self) -> None:
        self.activations = 0

    def activate(self) -> None:
        self.activations += 1

    def deactivate(self) -> None:
        self.activations -= 1


class FakeArm:
    def __init__(
        self,
        position: list[float] | None = None,
        mode_response: dict | None = None,
        brake_status: dict | None = None,
    ) -> None:
        self.position = np.asarray(position or [0.0] * 7, dtype=float)
        self.mode_response = mode_response or {"success": True}
        self.brake_status = brake_status or {
            "success": True,
            "enabled": False,
            "joints": [],
        }
        self.joint_name = [f"joint_{index}" for index in range(7)]
        self.commands: list[np.ndarray] = []

    def set_modes(self, _modes):
        return self.mode_response

    def set_joint_pos(self, target, wait_time=0.0):
        del wait_time
        self.commands.append(np.asarray(target, dtype=float))

    def get_joint_pos(self):
        return self.position.copy()

    def get_brake_status(self):
        return self.brake_status


class FakeRobot:
    def __init__(self, left: FakeArm, right: FakeArm) -> None:
        self.left_arm = left
        self.right_arm = right
        self.estop = FakeEStop()


class FakeTrackingWatchdog:
    def __init__(self, failures: dict[str, float]) -> None:
        self.failures = failures

    def update(self, _targets, _actuals):
        return self.failures


def controller_for(robot: FakeRobot) -> RobotController:
    controller = RobotController.__new__(RobotController)
    controller.robot = robot
    controller._validate_live_arm_safety = lambda: None
    controller._establish_current_pose_hold = lambda: None
    return controller


def test_mode_rejection_aborts_before_hold_for_owner_to_park() -> None:
    robot = FakeRobot(
        FakeArm(),
        FakeArm(mode_response={"success": False, "message": "timeout"}),
    )
    controller = controller_for(robot)
    hold_called = False

    def hold() -> None:
        nonlocal hold_called
        hold_called = True

    controller._establish_current_pose_hold = hold

    with pytest.raises(RuntimeError, match="homing aborted"):
        controller._ensure_arms_in_position_mode()

    assert not hold_called
    assert robot.estop.activations == 0


def test_unknown_brake_status_aborts_before_position_control() -> None:
    robot = FakeRobot(
        FakeArm(),
        FakeArm(brake_status={"success": False, "message": "timeout"}),
    )
    controller = controller_for(robot)

    with pytest.raises(RuntimeError, match="brake status query failed"):
        controller._ensure_arms_in_position_mode()

    assert robot.right_arm.commands == []


def test_tracking_error_aborts_for_owner_to_park() -> None:
    robot = FakeRobot(FakeArm(position=[0.2] * 7), FakeArm())
    controller = controller_for(robot)
    controller._arm_tracking_watchdog = FakeTrackingWatchdog({"left_arm": 0.2})

    with pytest.raises(RuntimeError, match="tracking failed"):
        controller._require_tracking({"left_arm": np.zeros(7)})

    assert robot.estop.activations == 0


def test_tracking_within_tolerance_continues() -> None:
    robot = FakeRobot(FakeArm(position=[0.01] * 7), FakeArm())
    controller = controller_for(robot)
    controller._arm_tracking_watchdog = FakeTrackingWatchdog({})

    controller._require_tracking({"left_arm": np.zeros(7)})

    assert robot.estop.activations == 0


def test_planned_waypoint_pauses_until_arm_catches_up() -> None:
    left = FakeArm()
    robot = FakeRobot(left, FakeArm())
    controller = controller_for(robot)
    controller.exit_requested = False
    controller.WAYPOINT_CATCHUP_TIMEOUT_SECONDS = 1.0
    # The deployed tracking tolerance is 0.10 rad; the test's delayed arm
    # trails by exactly 0.1, so pin a stricter tolerance to exercise the
    # catch-up pause loop.
    controller.TRAJECTORY_TRACKING_TOL_RAD = 0.05

    target = np.full(7, 0.1)
    reads = 0

    def delayed_position():
        nonlocal reads
        reads += 1
        if reads < 3:
            return np.zeros(7)
        return target.copy()

    left.get_joint_pos = delayed_position

    class FakeRateLimiter:
        def __init__(self) -> None:
            self.sleeps = 0

        def sleep(self) -> None:
            self.sleeps += 1

    rate = FakeRateLimiter()
    controller._follow_planned_waypoint({"left_arm": target}, rate)

    assert len(left.commands) == 3
    assert rate.sleeps == 3


def test_realtime_target_is_not_lead_or_speed_limited() -> None:
    left = FakeArm()
    controller = controller_for(FakeRobot(left, FakeArm()))
    controller.control_rate = 100.0

    desired = np.ones(7)
    validated = controller._validate_realtime_arm_target(
        "left_arm",
        desired,
    )
    np.testing.assert_allclose(validated, desired)


@pytest.mark.parametrize(
    "target",
    [
        np.zeros(6),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan]),
    ],
)
def test_invalid_realtime_target_is_rejected(target: np.ndarray) -> None:
    controller = controller_for(FakeRobot(FakeArm(), FakeArm()))

    with pytest.raises(RuntimeError, match="invalid real-time target"):
        controller._validate_realtime_arm_target("left_arm", target)


def test_cleanup_holds_position_without_disable_or_estop() -> None:
    events: list[str] = []

    class CleanupEStop:
        def activate(self) -> None:
            events.append("estop")

    class CleanupRobot:
        estop = CleanupEStop()

        def shutdown(self) -> None:
            events.append("robot_shutdown")

    class CleanupNode:
        def shutdown(self) -> None:
            events.append("node_shutdown")

    controller = RobotController.__new__(RobotController)
    controller._cleanup_done = False
    controller.exit_requested = False
    controller.subscriber = None
    controller.robot = CleanupRobot()
    controller.has_torso = False
    controller.control_rate = 100.0
    controller.joint_publish_thread = None
    controller._debug_display = None
    controller.node = CleanupNode()
    controller._establish_current_pose_hold = lambda: events.append("hold")

    controller.cleanup()

    assert events == ["hold", "robot_shutdown", "node_shutdown"]


def test_skip_homing_keeps_robot_at_current_pose(monkeypatch) -> None:
    """Real-time teleop must not yank the arms to init_pos after alignment."""
    robot = FakeRobot(FakeArm(), FakeArm())
    controller = controller_for(robot)
    controller.skip_homing = True
    controller.interpolation_method = "none"

    homing_called = False
    monkeypatch.setattr(controller, "_move_to_home", lambda: (_ for _ in ()).throw(AssertionError("homing must be skipped")))
    mode_armed = []
    estop_events = []
    monkeypatch.setattr(controller, "_ensure_arms_in_position_mode", lambda: mode_armed.append(True))
    monkeypatch.setattr(robot.estop, "deactivate", lambda: estop_events.append("deactivate"))
    monkeypatch.setattr(robot.estop, "activate", lambda: estop_events.append("activate"))

    controller._perform_homing()

    assert homing_called is False
    assert mode_armed == [True]
    assert estop_events == ["deactivate", "activate"]


def test_homing_runs_when_not_skipped(monkeypatch) -> None:
    robot = FakeRobot(FakeArm(), FakeArm())
    controller = controller_for(robot)
    controller.skip_homing = False
    controller.interpolation_method = "none"

    homing_called = []
    monkeypatch.setattr(controller, "_move_to_home", lambda: homing_called.append(True))

    controller._perform_homing()

    assert homing_called == [True]


def test_homing_ruckig_path(monkeypatch) -> None:
    robot = FakeRobot(FakeArm(), FakeArm())
    controller = controller_for(robot)
    controller.skip_homing = False
    controller.interpolation_method = "ruckig"

    calls = []
    monkeypatch.setattr(controller, "_init_ruckig_generators", lambda: calls.append("init"))
    monkeypatch.setattr(controller, "_move_to_home_with_ruckig", lambda: calls.append("move"))

    controller._perform_homing()

    assert calls == ["init", "move"]
