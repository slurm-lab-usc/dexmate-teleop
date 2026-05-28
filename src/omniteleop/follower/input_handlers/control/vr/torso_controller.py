"""VR torso controller for robot waist/lift mechanism."""

from typing import Dict, Any

from omniteleop.follower.input_handlers.control.torso_controller import (
    AbstractTorsoController,
)
from omniteleop.follower.input_handlers.control.commands import TorsoCommand


class VRTorsoController(AbstractTorsoController):
    """VR-specific controller for torso movement.

    Uses VR controller thumbsticks for torso control:
    - Right stick: Forward/backward (X) and up/down (Y) movement
    - Now activated via discrete toggle (right grip trigger) instead of continuous check
    """

    def __init__(self, sensitivity: float = 0.01, deadzone: float = 0.1):
        """Initialize VR torso controller.

        Args:
            sensitivity: Movement sensitivity in meters per frame
            deadzone: Input deadzone threshold
        """
        super().__init__(sensitivity, deadzone)

    def is_active(self, vr_data: Dict[str, Any]) -> bool:
        """Check if torso control is active.

        Note: With the new toggle-based system, this method is called
        only when torso control is already activated in the main controller.

        Args:
            vr_data: VR controller data

        Returns:
            True if torso control is activated (always True when called)
        """
        # In the new toggle-based system, this method is called only when
        # torso control is already activated, so we always return True
        return True

    def process(self, vr_data: Dict[str, Any]) -> TorsoCommand:
        """Process VR data for torso control.

        Args:
            vr_data: VR controller data

        Returns:
            TorsoCommand with movement deltas
        """
        right = vr_data.get("right", {})
        thumbstick = right.get("thumbstick", {})

        command = TorsoCommand()

        delta_x = 0.0
        delta_z = 0.0

        rx = float(thumbstick.get("x", 0.0))
        ry = float(thumbstick.get("y", 0.0))

        if abs(rx) > self.deadzone:
            delta_x = rx * self.sensitivity

        if abs(ry) > self.deadzone:
            delta_z = ry * self.sensitivity

        command.active = True
        command.delta_x = delta_x
        command.delta_z = delta_z

        return command
