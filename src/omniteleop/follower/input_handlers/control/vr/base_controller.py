"""VR base controller for mobile robot platform."""

from typing import Dict, Any

from omniteleop.follower.input_handlers.control.base_controller import (
    AbstractBaseController,
)
from omniteleop.follower.input_handlers.control.commands import BaseCommand


class VRBaseController(AbstractBaseController):
    """VR-specific controller for mobile base movement.

    Uses VR controller thumbsticks for base control:
    - Left stick: Forward/backward (Y) and left/right (X) movement
    - Right stick X: Rotation (yaw)
    - Now activated via discrete toggle (left grip trigger) instead of continuous check
    """

    def __init__(
        self,
        linear_velocity: float = 0.5,
        angular_velocity: float = 1.0,
        deadzone: float = 0.1,
    ):
        """Initialize base controller.

        Args:
            linear_velocity: Max linear velocity in m/s
            angular_velocity: Max angular velocity in rad/s
            deadzone: Input deadzone threshold
        """
        super().__init__(linear_velocity, angular_velocity, deadzone)

    def is_active(self, vr_data: Dict[str, Any]) -> bool:
        """Check if base control is active.

        Note: With the new toggle-based system, this method is called
        only when base control is already activated in the main controller.

        Args:
            vr_data: Raw VR data

        Returns:
            True if base control is activated (always True when called)
        """
        # In the new toggle-based system, this method is called only when
        # base control is already activated, so we always return True
        return True

    def process(self, vr_data: Dict[str, Any]) -> BaseCommand:
        """Process VR data for base control.

        Args:
            vr_data: VR controller data

        Returns:
            BaseCommand with velocity commands
        """
        left = vr_data.get("left", {})
        right = vr_data.get("right", {})

        left_stick = left.get("thumbstick", {})
        right_stick = right.get("thumbstick", {})

        left_stick_x = float(left_stick.get("x", 0.0))
        left_stick_y = float(left_stick.get("y", 0.0))
        right_stick_x = float(right_stick.get("x", 0.0))

        command = BaseCommand()

        # Left stick controls planar velocity
        if abs(left_stick_y) > self.deadzone:
            command.vx = left_stick_y * self.linear_velocity  # Forward/backward
        if abs(left_stick_x) > self.deadzone:
            command.vy = left_stick_x * self.linear_velocity  # Left/right

        # Right stick X controls rotation
        if abs(right_stick_x) > self.deadzone:
            command.wz = right_stick_x * self.angular_velocity  # Rotation

        # Base is active if any movement is commanded
        command.active = True

        return command
