from __future__ import annotations

import numpy as np
import pytest

from omniteleop.app.backend.app_backend import (
    _control_arm_brake,
    _read_arm_brake_status,
    _require_arm_brakes_engaged,
)


class FakeArm:
    def __init__(self, side: str) -> None:
        self.position = np.zeros(7)
        self.joint_pos_limit = np.tile(np.array([[-2.0, 2.0]]), (7, 1))
        self.joint_name = [f"{side}_arm_j{index + 1}" for index in range(7)]
        self.calls: list[tuple[bool, list[int] | None]] = []
        self.enabled = False
        self.joints: list[int] = []

    def get_joint_pos(self):
        return self.position.copy()

    def release_brake(self, enable: bool, joints: list[int] | None = None):
        self.calls.append((enable, joints))
        self.enabled = enable
        self.joints = list(joints or []) if enable else []
        return {"success": True, "message": "ok"}

    def get_brake_status(self):
        return {
            "success": True,
            "enabled": self.enabled,
            "joints": self.joints,
            "message": "ok",
        }


class FakeRobot:
    def __init__(self) -> None:
        self.left_arm = FakeArm("L")
        self.right_arm = FakeArm("R")


def test_status_reports_all_fourteen_joints() -> None:
    bot = FakeRobot()
    bot.right_arm.position[5] = -1.423
    bot.right_arm.joint_pos_limit[5] = [-1.396, 1.396]
    bot.right_arm.enabled = True
    bot.right_arm.joints = [5]

    result = _read_arm_brake_status(bot)

    assert result["any_released"] is True
    assert len(result["arms"]["left_arm"]["joints"]) == 7
    assert len(result["arms"]["right_arm"]["joints"]) == 7
    right_j6 = result["arms"]["right_arm"]["joints"][5]
    assert right_j6["name"] == "R_arm_j6"
    assert right_j6["outside_limits"] is True
    assert right_j6["released"] is True


def test_enabled_status_without_joint_list_is_treated_as_all_released() -> None:
    bot = FakeRobot()
    bot.left_arm.enabled = True
    bot.left_arm.joints = []

    result = _read_arm_brake_status(bot)

    assert result["arms"]["left_arm"]["released_joints"] == list(range(7))
    assert all(
        joint["released"]
        for joint in result["arms"]["left_arm"]["joints"]
    )


def test_release_selected_joint_requires_explicit_confirmation() -> None:
    bot = FakeRobot()

    with pytest.raises(ValueError, match="confirmation"):
        _control_arm_brake(
            bot,
            "release",
            arm_name="left_arm",
            joint_index=2,
        )

    assert bot.left_arm.calls == []


def test_release_operates_on_only_the_selected_joint() -> None:
    bot = FakeRobot()

    result = _control_arm_brake(
        bot,
        "release",
        arm_name="left_arm",
        joint_index=2,
        confirmed=True,
    )

    assert bot.left_arm.calls == [(True, [2])]
    assert bot.right_arm.calls == []
    assert result["any_released"] is True
    assert result["selected_joint"]["name"] == "L_arm_j3"
    assert result["arms"]["left_arm"]["released_joints"] == [2]


def test_cannot_release_second_joint_until_existing_release_is_disabled() -> None:
    bot = FakeRobot()
    bot.right_arm.enabled = True
    bot.right_arm.joints = [5]

    with pytest.raises(ValueError, match="already released"):
        _control_arm_brake(
            bot,
            "release",
            arm_name="left_arm",
            joint_index=1,
            confirmed=True,
        )

    assert bot.left_arm.calls == []


def test_engage_disables_release_on_every_active_arm() -> None:
    bot = FakeRobot()
    bot.left_arm.enabled = True
    bot.left_arm.joints = [1]
    bot.right_arm.enabled = True
    bot.right_arm.joints = [5]

    result = _control_arm_brake(bot, "engage")

    assert bot.left_arm.calls == [(False, None)]
    assert bot.right_arm.calls == [(False, None)]
    assert result["any_released"] is False


def test_motion_guard_rejects_any_active_release() -> None:
    bot = FakeRobot()
    bot.right_arm.enabled = True
    bot.right_arm.joints = [5]

    with pytest.raises(RuntimeError, match="before movement"):
        _require_arm_brakes_engaged(_read_arm_brake_status(bot))
