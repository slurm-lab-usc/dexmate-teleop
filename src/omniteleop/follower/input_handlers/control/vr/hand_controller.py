"""VR hand controller for end-effectors (hands/grippers)."""

from typing import Dict, Any

from loguru import logger

from omniteleop.follower.input_handlers.control.commands import HandCommand
from omniteleop.follower.input_handlers.control.vr.end_effectors import (
    create_end_effector,
    VREndEffectorInput,
)
from omniteleop.follower.input_handlers.control.hand_controller import (
    AbstractHandController,
)
from omniteleop.follower.input_handlers.control.end_effector import (
    AbstractEndEffectorController,
)


class VRHandController(AbstractHandController):
    """Interprets VR (Quest) inputs for hand/gripper control.

    Control schemes (when no gripper trigger is pressed):
    - Basic gestures via thumbstick/index trigger mapping in end effectors
    - Fine adjustment through stick movement; stick click available as a boolean

    Modifiers:
    - Gripper triggers act as modifiers and DISABLE hand control
    """

    def __init__(
        self, left_config: Dict[str, Any] = None, right_config: Dict[str, Any] = None
    ):
        """Initialize VR hand controller.

        Args:
            left_config: Configuration for left end-effector
            right_config: Configuration for right end-effector
        """
        super().__init__(left_config, right_config)

        # Create end-effector controllers
        self.left_effector = create_end_effector("left", self.left_config)
        self.right_effector = create_end_effector("right", self.right_config)

        logger.info(
            f"Hand controller initialized with {left_config.get('type', 'none')} (left) and {right_config.get('type', 'none')} (right)"
        )

    def process(self, vr_data: Dict[str, Any]) -> HandCommand:
        """Process VR data for hand control.

        Args:
            vr_data: Raw VR data with 'left' and 'right' controller info

        Returns:
            HandCommand with joint positions or deltas
        """
        command = HandCommand()

        # Disable hand control if modifiers (gripper triggers) are pressed
        if self._has_modifiers(vr_data):
            return command

        # Process left hand
        left_input = self._parse_vr_input("left", vr_data)
        left_result = self.left_effector.process_input(left_input)

        # Process right hand
        right_input = self._parse_vr_input("right", vr_data)
        right_result = self.right_effector.process_input(right_input)

        # Check if result is a tuple (positions, mode) or just positions
        if isinstance(left_result, tuple):
            left_positions, left_mode = left_result
        else:
            left_positions = left_result
            left_mode = "absolute"

        if isinstance(right_result, tuple):
            right_positions, right_mode = right_result
        else:
            right_positions = right_result
            right_mode = "absolute"

        # Update command if we have positions
        if left_positions:
            command.left_positions = left_positions
            command.left_mode = left_mode
            command.active = True

        if right_positions:
            command.right_positions = right_positions
            command.right_mode = right_mode
            command.active = True

        return command

    def _has_modifiers(self, vr_data: Dict[str, Any]) -> bool:
        """Check if any modifier inputs are pressed (gripper triggers).

        Args:
            vr_data: Raw VR data

        Returns:
            True if modifiers are pressed
        """
        left = vr_data.get("left", {})
        right = vr_data.get("right", {})

        left_grip = float(left.get("grip_trigger", 0.0))
        right_grip = float(right.get("grip_trigger", 0.0))
        return (left_grip > 0.5) or (right_grip > 0.5)

    def _parse_vr_input(self, side: str, vr_data: Dict[str, Any]) -> VREndEffectorInput:
        """Parse VR data for a specific side.

        Args:
            side: "left" or "right"
            vr_data: Raw VR data

        Returns:
            Parsed VREndEffectorInput
        """
        controller = vr_data.get(side, {})

        # Strict PoseData: 'thumbstick' is [x, y]
        ts = controller.get("thumbstick", [0.0, 0.0])
        if isinstance(ts, dict):
            stick_x = float(ts.get("x", 0.0))
            stick_y = float(ts.get("y", 0.0))
        else:
            stick_x = float(ts[0]) if len(ts) > 0 else 0.0
            stick_y = float(ts[1]) if len(ts) > 1 else 0.0

        # Buttons: synthesize with stick click only (strict key)
        stick_click = bool(controller.get("thumbstick_click", False))
        buttons = {"stick": stick_click}

        # Index trigger (0..1) for four-finger control
        index_trigger = float(controller.get("index_trigger", 0.0))

        return VREndEffectorInput(
            stick_x=stick_x,
            stick_y=stick_y,
            buttons=buttons,
            index_trigger=index_trigger,
        )

    def is_active(self, vr_data: Dict[str, Any]) -> bool:
        """Check if hand control is active without processing.

        Args:
            vr_data: Raw VR data

        Returns:
            True if hand control should be active
        """
        # Hand control is active when no modifiers are pressed
        return not self._has_modifiers(vr_data)

    def get_left_effector(self) -> AbstractEndEffectorController:
        """Get the left end-effector controller."""
        return self.left_effector

    def get_right_effector(self) -> AbstractEndEffectorController:
        """Get the right end-effector controller."""
        return self.right_effector
