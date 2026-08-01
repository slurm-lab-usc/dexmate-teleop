"""VR controller implementation."""

from typing import Dict, Any, Optional

from loguru import logger

from omniteleop.follower.input_handlers.control.controller import AbstractController
from omniteleop.follower.input_handlers.control.commands import RobotCommands
from omniteleop.follower.input_handlers.utils.button_manager import ButtonManager
from .base_controller import VRBaseController
from .torso_controller import VRTorsoController
from .hand_controller import VRHandController


class VRController(AbstractController):
    """VR-specific controller implementation.

    Manages VR input processing with Quest-specific emergency controls
    and routing to appropriate component controllers.

    Direct trigger-based activation system:
    - Base control: hold left grip trigger (highest priority)
    - Torso control: hold right grip trigger (medium priority)
    - Hand control: when no grip triggers pressed (lowest priority)

    Priority order: exit > estop > base > torso > hands
    """

    # Trigger thresholds for mode activation
    GRIP_TRIGGER_THRESHOLD = 0.5

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VR controller.

        Args:
            config: Configuration dictionary with VR-specific settings
        """
        super().__init__(config)

        # Initialize component controllers
        self._init_component_controllers()

        # Initialize button manager for safety combos only (estop/exit)
        self._init_button_manager()

        # Update stats to include component-specific counters
        self.stats.update(
            {
                "base_activations": 0,
                "torso_activations": 0,
                "hand_activations": 0,
                "total_updates": 0,
                "estop_toggles": 0,
            }
        )

        logger.info("VR controller initialized with direct trigger-based controls")

    def _init_component_controllers(self) -> None:
        """Initialize component controllers with configuration."""
        # Extract component configurations
        base_config = self.config.get("base", {})
        torso_config = self.config.get("torso", {})
        hands_config = self.config.get("hands", {})

        # Determine hand type from ROBOT_CONFIG env var (same as JoyCon controller)
        import os

        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1u_gripper")
        if "gripper" in robot_config:
            hand_type = "gripper"
        elif "f5d6" in robot_config:
            hand_type = "hand_f5d6"
        else:
            hand_type = "none"

        # Get left and right configurations directly from hands config (flat structure)
        left_config = hands_config.get("left", {})
        right_config = hands_config.get("right", {})

        # Add type and sensitivity to each config for hand controller
        left_config["type"] = hand_type
        right_config["type"] = hand_type
        left_config["sensitivity"] = hands_config.get("sensitivity", 0.05)
        right_config["sensitivity"] = hands_config.get("sensitivity", 0.05)

        # Initialize component controllers
        self.base_controller = VRBaseController(
            linear_velocity=base_config.get("linear_velocity", 0.5),
            angular_velocity=base_config.get("angular_velocity", 1.0),
            deadzone=base_config.get("deadzone", 0.1),
        )

        self.torso_controller = VRTorsoController(
            sensitivity=torso_config.get("sensitivity", 0.01),
            deadzone=torso_config.get("deadzone", 0.1),
        )

        self.hand_controller = VRHandController(
            left_config=left_config,
            right_config=right_config,
        )

    def _init_button_manager(self) -> None:
        """Initialize button manager for safety combos only (estop/exit)."""
        self.button_manager = ButtonManager()

        # Get button timing configs
        button_config = self.config.get("button_timings", {})
        default_debounce = button_config.get("default_debounce", 0.05)

        # Add thumbstick click buttons for safety features only
        self.button_manager.add_button(
            "left_thumbstick_click", debounce_time=default_debounce
        )
        self.button_manager.add_button(
            "right_thumbstick_click", debounce_time=default_debounce
        )
        # Add grip triggers for exit combo
        self.button_manager.add_button(
            "left_grip_trigger", debounce_time=default_debounce
        )
        self.button_manager.add_button(
            "right_grip_trigger", debounce_time=default_debounce
        )

        # Add button combos for safety features
        estop_duration = button_config.get("estop_hold_duration", 1.0)
        exit_duration = button_config.get("exit_hold_duration", 2.0)
        combo_grace_period = button_config.get("combo_grace_period", 0.1)

        # Emergency stop combo: both thumbstick clicks for 1 second
        self.button_manager.add_combo(
            name="estop_toggle",
            buttons=["left_thumbstick_click", "right_thumbstick_click"],
            hold_duration=estop_duration,
            require_simultaneous=True,
            grace_period=combo_grace_period,
            on_triggered=self._on_estop_triggered,
        )

        # Emergency exit combo: both thumbstick clicks + both grip triggers for 2 seconds
        self.button_manager.add_combo(
            name="emergency_exit",
            buttons=[
                "left_thumbstick_click",
                "right_thumbstick_click",
                "left_grip_trigger",
                "right_grip_trigger",
            ],
            hold_duration=exit_duration,
            require_simultaneous=True,
            grace_period=combo_grace_period,
            on_triggered=self._on_exit_triggered,
        )

    def _on_estop_triggered(self) -> None:
        """Handle emergency stop combo trigger."""
        self.estop_active = not self.estop_active
        self.stats["estop_toggles"] += 1
        logger.warning(
            f"Emergency stop {'ACTIVATED' if self.estop_active else 'DEACTIVATED'}"
        )

    def _on_exit_triggered(self) -> None:
        """Handle emergency exit combo trigger."""
        self.exit_requested = True
        self.estop_active = True
        logger.critical("EMERGENCY EXIT ACTIVATED - Shutting down all systems")

    def process(self, vr_data: Dict[str, Any]) -> RobotCommands:
        """Process VR data and return robot commands.

        Uses direct trigger values for mode activation:
        - Left grip held -> base mode
        - Right grip held -> torso mode
        - Neither grip pressed -> hands mode

        Args:
            vr_data: Raw VR controller data

        Returns:
            RobotCommands with appropriate component commands
        """
        self.stats["total_updates"] += 1
        commands = RobotCommands()

        # Extract controller inputs
        left = vr_data.get("left", {})
        right = vr_data.get("right", {})

        # Get trigger values directly
        left_grip = float(left.get("grip_trigger", 0.0))
        right_grip = float(right.get("grip_trigger", 0.0))

        # Prepare button states for button manager (safety combos only)
        raw_button_states = {
            "left_grip_trigger": left_grip > self.GRIP_TRIGGER_THRESHOLD,
            "right_grip_trigger": right_grip > self.GRIP_TRIGGER_THRESHOLD,
            "left_thumbstick_click": bool(left.get("thumbstick_click", False)),
            "right_thumbstick_click": bool(right.get("thumbstick_click", False)),
        }

        # Update button manager for safety combos (estop/exit)
        self.button_manager.update(raw_button_states)

        # Handle emergency exit first (highest priority)
        if self.exit_requested:
            commands.estop = True
            commands.exit_requested = True
            commands.priority = "exit"
            return commands

        # Handle emergency stop
        if self.estop_active:
            commands.estop = True
            commands.priority = "estop"

            # Check if we're currently holding estop combo for progress feedback
            estop_combo = self.button_manager.get_combo("estop_toggle")
            if (
                estop_combo
                and estop_combo.progress is not None
                and estop_combo.progress < 1.0
            ):
                remaining = estop_combo.time_remaining
                if remaining and int((2.0 - remaining) * 10) % 5 == 0:  # Log every 0.5s
                    logger.info(
                        f"Hold both sticks for {remaining:.1f}s more to toggle estop"
                    )

            return commands

        # Direct trigger-based mode activation (no toggle, hold to activate)
        # Priority: base > torso > hands

        # Base control: hold left grip trigger
        if left_grip > self.GRIP_TRIGGER_THRESHOLD:
            commands.base = self.base_controller.process(vr_data)
            commands.priority = "base"
            self.stats["base_activations"] += 1
            return commands

        # Torso control: hold right grip trigger
        if right_grip > self.GRIP_TRIGGER_THRESHOLD:
            commands.torso = self.torso_controller.process(vr_data)
            commands.priority = "torso"
            self.stats["torso_activations"] += 1
            return commands

        # Hand control: when no grip triggers pressed
        commands.hands = self.hand_controller.process(vr_data)
        commands.priority = "hands"
        self.stats["hand_activations"] += 1
        return commands

    def get_control_mode(self, vr_data: Dict[str, Any]) -> str:
        """Get current control mode based on trigger values.

        Args:
            vr_data: Raw VR controller data

        Returns:
            String indicating active control mode
        """
        left = vr_data.get("left", {})
        right = vr_data.get("right", {})
        left_grip = float(left.get("grip_trigger", 0.0))
        right_grip = float(right.get("grip_trigger", 0.0))

        if left_grip > self.GRIP_TRIGGER_THRESHOLD:
            return "base"
        elif right_grip > self.GRIP_TRIGGER_THRESHOLD:
            return "torso"
        else:
            return "hands"

    def get_base_controller(self) -> VRBaseController:
        """Get the base controller for direct access."""
        return self.base_controller

    def get_torso_controller(self) -> VRTorsoController:
        """Get the torso controller for direct access."""
        return self.torso_controller

    def get_hand_controller(self) -> VRHandController:
        """Get the hand controller for direct access."""
        return self.hand_controller
