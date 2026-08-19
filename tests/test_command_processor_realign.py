"""E-Stop release must re-verify exo/robot alignment before motion resumes.

Without the gate, the first tick after a release commands wherever the leader
drifted to while the robot sat frozen. That step lands outside the window the
arm firmware accepts around the present position, the command is rejected, the
arm does not move, and every later command is rejected for the same reason.
"""

from __future__ import annotations

import threading
import time

from omniteleop.follower.command_processor import CommandProcessor
from omniteleop.follower.input_handlers.base_handler import RobotCommand


def processor_for(aligned: bool) -> CommandProcessor:
    processor = CommandProcessor.__new__(CommandProcessor)
    processor.teleop_mode = "exo_joycon"
    processor.exit_after_publish = False
    processor._align_checked = True
    processor._post_align_estop_seen = True
    processor._motion_control_started = True
    processor._estop_was_active = False
    processor._pending_realign = False
    processor._last_align_log = {}
    processor.processors = {}
    processor.syncs = []
    processor._is_exo_aligned_with_robot = lambda _command: aligned
    processor._sync_motion_manager_to_robot_state = lambda: processor.syncs.append(True)
    return processor


def tick(processor: CommandProcessor, estop: bool) -> bool:
    """Run one processing tick and return the published estop state."""
    command = RobotCommand(timestamp_ns=time.time_ns())
    command.safety_flags.emergency_stop = estop
    processor._process_components(command)
    return command.safety_flags.emergency_stop


def test_release_after_estop_holds_until_realigned() -> None:
    processor = processor_for(aligned=False)

    assert tick(processor, estop=True) is True
    # Leader drifted while the robot was frozen: releasing must not resume.
    assert tick(processor, estop=False) is True
    assert tick(processor, estop=False) is True
    assert processor._pending_realign is True
    assert processor.syncs == []


def test_release_after_estop_resumes_once_realigned() -> None:
    processor = processor_for(aligned=True)

    assert tick(processor, estop=True) is True
    assert tick(processor, estop=False) is False
    assert processor._pending_realign is False
    # Motion manager is re-seeded from the robot's live pose on resume.
    assert processor.syncs == [True]


def test_realign_gate_clears_then_rearms_on_the_next_estop() -> None:
    processor = processor_for(aligned=True)

    tick(processor, estop=True)
    tick(processor, estop=False)
    assert processor._pending_realign is False

    # A second pause must re-arm the check rather than coast on the first one.
    tick(processor, estop=True)
    assert processor._pending_realign is True


def alignment_checker(age_s: float) -> CommandProcessor:
    processor = CommandProcessor.__new__(CommandProcessor)
    processor.teleop_mode = "exo_joycon"
    processor._robot_joints_lock = threading.RLock()
    processor._robot_joints_max_age_s = 1.0
    processor._align_threshold = 0.50
    processor._last_align_log = {}
    processor.latest_robot_joints = {
        "joints": {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7}
    }
    processor._robot_joints_at = time.monotonic() - age_s
    return processor


def command_with_arms_at(value: float) -> RobotCommand:
    command = RobotCommand(timestamp_ns=time.time_ns())
    command.input_components = {
        "left_arm": {"pos": [value] * 7},
        "right_arm": {"pos": [value] * 7},
    }
    return command


def test_fresh_feedback_allows_alignment() -> None:
    processor = alignment_checker(age_s=0.0)

    assert processor._is_exo_aligned_with_robot(command_with_arms_at(0.0)) is True


def test_stale_feedback_never_reads_as_aligned() -> None:
    """A dead robot_controller must not let motion resume on a frozen pose."""
    processor = alignment_checker(age_s=5.0)

    assert processor._is_exo_aligned_with_robot(command_with_arms_at(0.0)) is False


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.published.append(payload)


def test_published_flags_distinguish_a_realign_hold_from_a_held_joycon() -> None:
    """The UI cannot explain the hold without this flag on the wire."""
    processor = CommandProcessor.__new__(CommandProcessor)
    processor.safe_command_pub = FakePublisher()
    command = RobotCommand(timestamp_ns=time.time_ns())
    command.safety_flags.emergency_stop = True

    processor._pending_realign = True
    processor.publish_command(command)
    processor._pending_realign = False
    processor.publish_command(command)

    flags = [payload["safety_flags"] for payload in processor.safe_command_pub.published]
    assert flags[0]["emergency_stop"] is True
    assert flags[0]["pending_realign"] is True
    # Same E-Stop state, different reason — that is the whole point.
    assert flags[1]["emergency_stop"] is True
    assert flags[1]["pending_realign"] is False


def test_non_exo_modes_are_unaffected() -> None:
    """VR has no exo to align; the real check short-circuits to True."""
    processor = processor_for(aligned=False)
    processor.teleop_mode = "vr"
    processor._is_exo_aligned_with_robot = (
        lambda command: CommandProcessor._is_exo_aligned_with_robot(processor, command)
    )

    assert tick(processor, estop=True) is True
    assert tick(processor, estop=False) is False
