from __future__ import annotations

import pytest

from omniteleop.follower.component_processors.hand_processor import HandProcessor
from omniteleop.follower.input_handlers.base_handler import CommandMode, RobotCommand
from omniteleop.follower.input_handlers.control.joycon import end_effectors
from omniteleop.follower.input_handlers.control.joycon.controller import (
    JoyConController,
)
from omniteleop.follower.input_handlers.control.joycon.end_effectors import (
    GripperController,
    JoyConEndEffectorInput,
)


def _effector_input(
    *,
    stick_y: float = 0.0,
    buttons: dict[str, bool] | None = None,
) -> JoyConEndEffectorInput:
    return JoyConEndEffectorInput(
        stick_x=0.0,
        stick_y=stick_y,
        buttons=buttons or {},
    )


def _gripper(side: str) -> GripperController:
    return GripperController(
        side,
        {
            "poses": {"open": [0.78], "close": [0.0]},
            "fine_max_speed": 0.2,
            "fine_stick_deadzone": 0.2,
            "fine_max_dt": 0.1,
        },
    )


def _joycon_state(
    *,
    left_buttons: dict[str, bool] | None = None,
    right_buttons: dict[str, bool] | None = None,
    left_stick_y: float = 0.0,
    right_stick_y: float = 0.0,
) -> dict:
    return {
        "left": {
            "buttons": left_buttons or {},
            "stick": {"x": 0.0, "y": left_stick_y},
        },
        "right": {
            "buttons": right_buttons or {},
            "stick": {"x": 0.0, "y": right_stick_y},
        },
    }


def test_stick_fine_control_is_rate_based_and_proportional(monkeypatch) -> None:
    clock = iter([10.0, 10.05, 10.10])
    monkeypatch.setattr(end_effectors.time, "monotonic", lambda: next(clock))
    gripper = _gripper("left")

    # The first neutral observation arms analog control.
    assert gripper.process_input(_effector_input()) == ([], "relative")

    positions, mode = gripper.process_input(_effector_input(stick_y=0.6))

    assert mode == "relative"
    # Deadzone-remapped magnitude: (0.6 - 0.2) / 0.8 = 0.5.
    # Delta: 0.5 * 0.2 rad/s * 0.05 s = 0.005 rad.
    assert positions == pytest.approx([0.005])


@pytest.mark.parametrize(
    ("side", "button", "expected"),
    [
        ("left", "up", [0.78]),
        ("left", "down", [0.0]),
        ("right", "x", [0.78]),
        ("right", "b", [0.0]),
    ],
)
def test_full_travel_buttons(side: str, button: str, expected: list[float]) -> None:
    positions, mode = _gripper(side).process_input(
        _effector_input(buttons={button: True})
    )

    assert mode == "absolute"
    assert positions == expected


def test_analog_control_requires_neutral_after_mode_or_chord(monkeypatch) -> None:
    times = iter([20.0, 20.05, 20.10, 20.15, 20.20, 20.25, 20.30])
    monkeypatch.setattr(end_effectors.time, "monotonic", lambda: next(times))
    gripper = _gripper("right")

    assert gripper.process_input(_effector_input()) == ([], "relative")
    assert gripper.process_input(_effector_input(stick_y=1.0))[0]

    gripper.require_neutral()
    assert gripper.process_input(_effector_input(stick_y=1.0)) == ([], "relative")
    assert gripper.process_input(_effector_input()) == ([], "relative")
    assert gripper.process_input(_effector_input(stick_y=1.0))[0]


def test_single_shoulder_buttons_do_not_toggle_a_mode_or_recording(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_CONFIG", "vega_1u_gripper")
    controller = JoyConController(
        {
            "hands": {
                "left": {"poses": {"open": [0.78], "close": [0.0]}},
                "right": {"poses": {"open": [0.78], "close": [0.0]}},
            },
            "button_timings": {
                "default_debounce": 0.0,
                "recording_hold_duration": 0.0,
            },
        }
    )
    controller.estop_active = False

    for buttons in ({"l": True}, {"r": True}):
        state = _joycon_state(
            left_buttons=buttons if "l" in buttons else {},
            right_buttons=buttons if "r" in buttons else {},
        )
        controller.process(state)
        controller.process(state)
        controller.process(_joycon_state())

    assert controller.get_recording_command() is None
    assert controller.get_activation_states()["fine_adjustment"] is False


def test_recording_chord_toggles_once_only_after_both_buttons_release(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_CONFIG", "vega_1u_gripper")
    controller = JoyConController(
        {
            "hands": {
                "left": {"poses": {"open": [0.78], "close": [0.0]}},
                "right": {"poses": {"open": [0.78], "close": [0.0]}},
            },
            "button_timings": {
                "default_debounce": 0.0,
                "recording_hold_duration": 0.0,
            },
        }
    )
    controller.estop_active = False
    chord = _joycon_state(
        left_buttons={"l": True},
        right_buttons={"r": True},
        left_stick_y=1.0,
        right_stick_y=-1.0,
    )

    controller.process(chord)
    held_command = controller.process(chord)

    assert held_command.hands.active is False
    assert controller.get_recording_command() is None

    controller.process(_joycon_state())

    assert controller.get_recording_command() == "toggle"
    assert controller.get_recording_command() is None


class _RobotInfo:
    def get_component_joints(self, component: str) -> list[str]:
        assert component == "left_hand"
        return ["L_gripper"]

    def get_joint_pos_limits(self, joints: list[str]) -> list[tuple[float, float]]:
        assert joints == ["L_gripper"]
        return [(0.0, 0.78)]


def test_relative_gripper_command_never_falls_back_to_closed_anchor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_CONFIG", "vega_1u_gripper")
    processor = HandProcessor(
        side="left",
        config={},
        motion_manager=object(),
        robot_info=_RobotInfo(),
        teleop_mode="exo_joycon",
        lock_collision_avoidance=True,
    )
    command = RobotCommand(timestamp_ns=1)

    assert (
        processor.process(
            {"mode": CommandMode.RELATIVE, "pos": [-0.01]},
            command,
        )
        is False
    )
    assert command.output_components == {}

    processor.sync_to_robot_state({"left_hand": [0.4]})
    assert processor.process(
        {"mode": CommandMode.RELATIVE, "pos": [-0.01]},
        command,
    )
    assert command.output_components["left_hand"]["pos"] == pytest.approx([0.39])
