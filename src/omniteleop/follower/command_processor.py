#!/usr/bin/env python3
"""Command processor with integrated safety validation.

This module processes teleoperation inputs from various sources,
applies safety checks, and publishes validated commands for the robot controller.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional

import numpy as np

from dexcomm import Node
from omniteleop import LIB_PATH
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.log_utils import suppress_loguru_module
from omniteleop.follower.input_handlers.base_handler import CommandMode
from dexcomm.codecs import DictDataCodec
from loguru import logger
from dexmotion.motion_manager import MotionManager
from dexcomm import RateLimiter

# Import component processors
from omniteleop.follower.component_processors import (
    ArmProcessor,
    HandProcessor,
    TorsoProcessor,
    HeadProcessor,
    ChassisProcessor,
    SafetyValidator,
)

# Import input handlers
from omniteleop.follower.input_handlers.base_handler import (
    BaseInputHandler,
    RobotCommand,
    ArmCommandType,
)
from omniteleop.follower.input_handlers.exo_joycon_handler import ExoJoyconHandler
from dexbot_utils import RobotInfo


class CommandProcessor:
    """Processes teleoperation commands with integrated safety validation.

    This class receives commands from various teleoperation input sources,
    applies safety checks including collision detection and joint limits,
    and publishes validated commands for robot control.

    Uses modular component processors for handling different robot parts.

    Attributes:
        config: Configuration dictionary loaded from YAML.
        teleop_mode: Current teleoperation mode (e.g., 'exo_joycon', 'vr').
        motion_manager: MotionManager instance for kinematics and collision checking.
        input_handler: Handler for processing teleoperation inputs.
        processors: Dictionary of component processors.
        safety_validator: Safety validation instance.
    """

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        visualize: bool = False,
        initial_joint_positions: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None,
        lock_hand_collision_avoidance: bool = False,
    ):
        """Initialize the command processor with configuration and handlers.

        Args:
            namespace: Namespace for Zenoh topics.
            debug: Whether to enable debug display output.
            visualize: Whether to enable visualization in simulator.
            initial_joint_positions: Initial joint positions for motion manager.
            config_name: Name of config file (without .yaml extension).
            lock_hand_collision_avoidance: If True, freeze LEFT_HAND/RIGHT_HAND
                joints in the MotionManager at the configured "open" pose so the
                collision checker treats them as a fixed open hand. Joycon hand
                commands still flow through to the robot, but bypass MotionManager
                updates for hand joints.

        Raises:
            RuntimeError: If input handler initialization fails.
        """
        self.lock_hand_collision_avoidance = lock_hand_collision_avoidance
        # Initialize Node with namespace support
        self.node = Node(name="command_processor", namespace=namespace)
        self.namespace = namespace

        self.robot_info = RobotInfo()
        self.has_torso = self.robot_info.has_torso
        self.has_base = self.robot_info.has_chassis

        # Load configuration
        config_path = None
        if config_name is not None:
            config_path = LIB_PATH / "configs" / f"{config_name}.yaml"
        self.config = get_config(config_path)
        self.running = False
        self.teleop_mode = self.config.get("teleop_mode", "exo_joycon")

        # Initialize safety parameters
        self.safety_config = self.config.get("safety")
        self.bypass_real_robot = self.safety_config.get("bypass_real_robot", False)

        # Initialize motion manager
        joints_to_lock = ["BASE"]
        if self.lock_hand_collision_avoidance:
            joints_to_lock += ["LEFT_HAND", "RIGHT_HAND"]
            initial_joint_positions = self._seed_open_hand_positions(
                initial_joint_positions
            )
        self.motion_manager = MotionManager(
            visualizer_type="sapien",
            init_visualizer=visualize,
            joint_regions_to_lock=joints_to_lock,
            initial_joint_configuration_dict=initial_joint_positions,
        )

        # Get init positions based on robot type
        init_pos_config = self.config.get("init_pos", {})

        # Store init positions for alignment check
        self._init_pos_left_arm = init_pos_config.get("left_arm", [0.0] * 7)
        self._init_pos_right_arm = init_pos_config.get("right_arm", [0.0] * 7)

        # Max allowed difference (rad) between exo and init_pos before motion
        # starts. Kept in sync with StateChecker.INIT_POS_ALIGN_THRESHOLD (0.50).
        self._align_threshold = 0.50

        # Initialize arms with robot-type-specific positions
        self.motion_manager.left_arm.set_joint_pos(
            init_pos_config.get("left_arm", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.motion_manager.right_arm.set_joint_pos(
            init_pos_config.get("right_arm", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        # Only initialize torso if robot has torso
        if self.has_torso:
            self.motion_manager.torso.set_joint_pos(
                init_pos_config.get("torso", [0.0, 0.0, 0.0])
            )

        # Initialize component processors
        self._initialize_processors()

        # Initialize safety validator
        self.safety_validator = SafetyValidator(self.config, self.motion_manager)

        # Create input handler
        self.input_handler: Optional[BaseInputHandler] = self._create_input_handler()

        # Thread-safe storage for robot joint feedback
        self.latest_robot_joints: Optional[Dict[str, Any]] = None
        self._robot_joints_lock = threading.RLock()

        # Control flags
        self.exit_after_publish = False
        self._robot_joint_data_initialized = self.bypass_real_robot
        self._motion_control_started = self.bypass_real_robot
        self._align_checked = self.bypass_real_robot
        self._last_safe_command: Optional[RobotCommand] = None

        # Fresh-release gate: after alignment passes, require the operator
        # to engage joycon estop at least once, then release, before motion
        # actually starts. Prevents surprise motion when the user had
        # already released joycon while the exo was still out of range.
        # Bypass mode auto-satisfies this (True from start).
        self._post_align_estop_seen = self.bypass_real_robot

        # Realignment tracking: when estop is released, re-check proximity
        # to current robot joint positions before resuming motion.
        self._estop_was_active = False
        self._pending_realign = False
        # Robot joint feedback older than this is treated as no feedback at
        # all by the alignment checks, so a dead robot_controller cannot let
        # motion resume against a frozen snapshot of the robot's pose.
        self._robot_joints_max_age_s = 1.0
        self._robot_joints_at: Optional[float] = None
        self._last_align_log: Dict[str, float] = {}  # suppress repeated align warnings

        # Set up subscribers and publishers through Dexcomm Node
        self._setup_communication()

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if self.debug:
            rate = self.config.get_rate("command_rate", 40)
            self._debug_display = get_debug_display(
                "CmdSafety", rate, refresh_rate=10, precision=3
            )

        logger.info(f"CommandProcessor initialized with {self.teleop_mode} mode")

    def _seed_open_hand_positions(
        self, initial_joint_positions: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Seed initial_joint_positions with the configured open-hand pose.

        Used when hand collision-avoidance is locked: hand joints are frozen
        in the reduced pinocchio model at this pose, so the collision checker
        sees a consistent open hand regardless of joycon hand commands.
        """
        seeded = dict(initial_joint_positions) if initial_joint_positions else {}
        handler_cfg = self.config.get("input_handlers", {}).get(self.teleop_mode, {})
        hands_cfg = handler_cfg.get("hands", {})
        for side in ("left", "right"):
            component = f"{side}_hand"
            if not self.robot_info.has_component(component):
                continue
            open_pose = (
                hands_cfg.get(side, {}).get("poses", {}).get("open")
            )
            if open_pose is None:
                logger.warning(
                    f"lock_hand_collision_avoidance: no '{side}.poses.open' under "
                    f"input_handlers.{self.teleop_mode}.hands; hand will lock at "
                    "whatever default the model uses."
                )
                continue
            joint_names = self.robot_info.get_component_joints(component)
            if len(joint_names) != len(open_pose):
                logger.warning(
                    f"lock_hand_collision_avoidance: {component} has "
                    f"{len(joint_names)} joints but 'open' pose has "
                    f"{len(open_pose)} values; skipping."
                )
                continue
            for name, value in zip(joint_names, open_pose):
                seeded[name] = value
        return seeded

    def _initialize_processors(self) -> None:
        """Initialize component processors based on robot capabilities."""
        self.processors = {}

        # Arms - always present
        self.processors["left_arm"] = ArmProcessor(
            "left", self.config, self.motion_manager, self.robot_info, self.teleop_mode
        )
        self.processors["right_arm"] = ArmProcessor(
            "right", self.config, self.motion_manager, self.robot_info, self.teleop_mode
        )

        # Hands - check if component exists
        if self.robot_info.has_component("left_hand"):
            self.processors["left_hand"] = HandProcessor(
                "left",
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
                lock_collision_avoidance=self.lock_hand_collision_avoidance,
            )
        if self.robot_info.has_component("right_hand"):
            self.processors["right_hand"] = HandProcessor(
                "right",
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
                lock_collision_avoidance=self.lock_hand_collision_avoidance,
            )

        # Single-instance components - check if they exist on robot
        if self.robot_info.has_torso:
            self.processors["torso"] = TorsoProcessor(
                None,
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )
        if self.robot_info.has_head:
            self.processors["head"] = HeadProcessor(
                None,
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )
        if self.robot_info.has_chassis:
            self.processors["chassis"] = ChassisProcessor(
                None,
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )

        logger.debug(f"Initialized processors: {list(self.processors.keys())}")

    def _create_input_handler(self) -> Optional[BaseInputHandler]:
        """Create input handler based on configured teleoperation mode.

        Factory method that creates the appropriate input handler
        (currently only ExoJoyconHandler) based on the teleop_mode setting.

        Returns:
            BaseInputHandler instance or None if mode is unsupported.

        Note:
            Only 'exo_joycon' is supported here. VR paddle teleop is now
            owned by the `omni-paddle` leader process
            (`src/omniteleop/leader/paddle_leader.py`), which subscribes to
            dexstream's `vr/controllers` Zenoh topic, runs IK in-process,
            and publishes `robot_commands` directly. `command_processor`
            is no longer in the VR path.
        """
        # Get handler configuration from unified structure
        handler_config = self.config.get("input_handlers", {}).get(self.teleop_mode, {})

        if self.teleop_mode == "exo_joycon":
            handler_config["topics"] = {
                "exo_joints": self.config.get_topic("exo_joints"),
                "exo_joycon": self.config.get_topic("exo_joycon"),
            }
            handler = ExoJoyconHandler(handler_config, self.namespace)
            logger.info("Created ExoJoyconHandler for exoskeleton + JoyCon input")
            return handler
        elif self.teleop_mode == "vr":
            raise ValueError(
                "teleop_mode='vr' is no longer supported by command_processor. "
                "VR paddle control is now owned by the omni-paddle leader "
                "(src/omniteleop/leader/paddle_leader.py), which publishes "
                "robot_commands directly. Remove 'vr' from this config or "
                "launch omni-paddle instead."
            )

        else:
            logger.error(f"Unsupported teleop mode: {self.teleop_mode}")
            return None

    def _setup_communication(self):
        """Configure Zenoh publishers and subscribers for robot communication.

        Sets up:
        - Subscriber for robot joint feedback
        - Publisher for validated robot commands
        - Input handler initialization

        Raises:
            RuntimeError: If input handler initialization fails.
        """
        # Initialize input handler
        if self.input_handler:
            if not self.input_handler.initialize():
                logger.error("Failed to initialize input handler")
                raise RuntimeError(
                    f"Input handler initialization failed for {self.teleop_mode}"
                )

        # Subscribe to robot joint feedback
        robot_joints_topic = self.config.get_topic("robot_joints")
        self.robot_joints_sub = self.node.create_subscriber(
            robot_joints_topic,
            self._on_robot_joints_received,
            decoder=DictDataCodec.decode,
        )
        commands_topic = self.config.get_topic("robot_commands")
        self.safe_command_pub = self.node.create_publisher(
            commands_topic,
            encoder=DictDataCodec.encode,
        )

    def _on_robot_joints_received(self, robot_joints_data: Dict[str, Any]) -> None:
        """Store latest robot joint positions and velocities from feedback.

        Thread-safe callback that stores robot joint data for synchronization
        and safety checking.

        Args:
            robot_joints_data: Dictionary containing timestamp, joint positions, and velocities.
        """
        try:
            with self._robot_joints_lock:
                self.latest_robot_joints = robot_joints_data
                self._robot_joints_at = time.monotonic()
            if self.latest_robot_joints:
                self._robot_joint_data_initialized = True
        except Exception as e:
            logger.error(f"Error storing robot joints feedback: {e}")

    def process_commands(self) -> Optional[RobotCommand]:
        """Process and validate teleoperation commands.

        Main processing pipeline that:
        1. Gets commands from input handler
        2. Processes components via modular processors
        3. Applies safety validation
        4. Returns validated command or last safe command

        Returns:
            Validated RobotCommand or None if no input available.

        Note:
            Falls back to last safe command if current command fails validation.
        """
        # Get command from input handler
        if not self.input_handler:
            logger.error("No input handler available")
            return None

        command = self.input_handler.process_inputs()

        if not command:
            return None

        # Process components using modular processors
        self._process_components(command)

        if not self._motion_control_started:
            command = RobotCommand(timestamp_ns=time.time_ns())
            command.safety_flags.emergency_stop = True
            return command

        # Apply safety validation
        self.safety_validator.validate(command)

        # Return validated command or fall back to last safe command
        if command.valid:
            self._last_safe_command = command
            return command
        else:
            # Use last known safe command to maintain stability
            return self._last_safe_command

    def _process_components(self, command: RobotCommand) -> None:
        """Process all components using modular processors.

        Routes input components to appropriate processors. Handles:
        - Emergency stops and exit requests
        - Motion control startup
        - Component-specific processing via processors
        - Special case: dual-arm EE pose IK

        Args:
            command: RobotCommand to process (modified in-place)

        Side Effects:
            - Sets exit_after_publish flag if exit requested
            - Starts motion control when emergency stop is released
            - Syncs motion manager to robot state on startup
            - Modifies command.output_components with processed data
        """
        # Check for exit request flag
        if command.safety_flags.exit_requested:
            logger.critical("Exit requested - will shutdown after publishing")
            self.exit_after_publish = True
            # Ensure emergency stop is also active when exiting
            command.safety_flags.emergency_stop = True

        # ── Alignment check (runs every tick until it latches) ──────────
        # Run the alignment check unconditionally — do NOT gate it on
        # joycon estop state. The user must be able to observe the
        # ALIGN → PAUSED transition as soon as the exo enters range,
        # regardless of whether they have joycon engaged or released.
        # Joycon releases during ALIGN are suppressed further down by
        # force-stopping the output command until alignment passes.
        if not self._align_checked and self._is_exo_aligned_with_robot(command):
            self._align_checked = True
            self._sync_motion_manager_to_robot_state()
            logger.success("Exo joints aligned — motion gate lifted.")

        # ── Force estop while not aligned ───────────────────────────────
        # Overrides joycon — any release attempts during ALIGN have no
        # effect on the published command.
        if not self._align_checked:
            command.safety_flags.emergency_stop = True

        # ── Post-align fresh-engagement gate ────────────────────────────
        # Once aligned, we require the user to have joycon engaged
        # (emergency_stop=True) at least once before we'll honor a
        # release. This prevents motion from starting the instant
        # alignment passes if the user had already released joycon
        # during ALIGN. If they held joycon engaged throughout, this
        # flag flips True on the same tick alignment passes, so a
        # single subsequent release starts motion — matching the
        # flow "ALIGN → (exo in range) → PAUSED → (release joycon)
        # → ACTIVE".
        if (
            self._align_checked
            and not self._post_align_estop_seen
            and command.safety_flags.emergency_stop
        ):
            self._post_align_estop_seen = True

        # If aligned but user has never engaged estop since alignment,
        # force estop on this tick (they're coasting on a pre-align
        # release).
        if self._align_checked and not self._post_align_estop_seen:
            command.safety_flags.emergency_stop = True

        # ── Re-alignment after an E-Stop pause ──────────────────────────
        # _align_checked latches for the whole session, so on its own the
        # first tick after an E-Stop release commands wherever the leader
        # drifted to while the robot sat frozen. That step lands outside the
        # window the arm firmware accepts around the present position, the
        # command is rejected, and because the arm never moves every later
        # command is rejected too — a deadlock that clearing the error does
        # not fix. Re-run the same proximity check the initial gate uses and
        # hold E-Stop until the operator brings the leader back to the
        # robot's live pose.
        estop_now = command.safety_flags.emergency_stop
        if estop_now and not self._estop_was_active:
            self._pending_realign = True
        self._estop_was_active = estop_now

        if self._pending_realign and not estop_now:
            if self._is_exo_aligned_with_robot(command):
                self._pending_realign = False
                self._sync_motion_manager_to_robot_state()
                logger.success("Exo re-aligned after E-Stop — resuming motion.")
            else:
                command.safety_flags.emergency_stop = True
                self._estop_was_active = True

        # ── Motion control start (standard path) ────────────────────────
        if command.safety_flags.emergency_stop:
            # Emergency stop is active, don't start motion control
            pass
        elif not self._motion_control_started:
            # First tick with estop=False reaching here necessarily has
            # _align_checked=True AND _post_align_estop_seen=True (the
            # earlier blocks re-force estop=True otherwise). Start motion.
            self._motion_control_started = True
            logger.success("Motion control started.")

        # Special case: Dual-arm EE pose requires both arms together
        if self._should_process_dual_arm_ee_pose(command):
            self._process_dual_arm_ee_pose(command)
            return

        # Process each component independently via processors
        for component_name in list(command.input_components.keys()):
            input_data = command.input_components.get(component_name)
            processor = self.processors.get(component_name)

            if processor is None:
                # Unknown component - remove it
                command.input_components.pop(component_name)
                continue

            if not self.robot_info.has_component(component_name):
                # Component disabled for this robot type
                command.input_components.pop(component_name)
                continue

            # Process component and remove from input_components
            command.input_components.pop(component_name)
            processor.process(input_data, command)

    def _should_process_dual_arm_ee_pose(self, command: RobotCommand) -> bool:
        """Check if command requires dual-arm EE pose processing.

        Args:
            command: RobotCommand to check

        Returns:
            True if both arms have EE_POSE command type
        """
        if (
            "left_arm" not in command.input_components
            or "right_arm" not in command.input_components
        ):
            return False

        left_data = command.input_components["left_arm"]
        right_data = command.input_components["right_arm"]

        left_type = left_data.get("command_type", ArmCommandType.JOINT)
        right_type = right_data.get("command_type", ArmCommandType.JOINT)

        return (
            left_type == ArmCommandType.EE_POSE and right_type == ArmCommandType.EE_POSE
        )

    def _process_dual_arm_ee_pose(self, command: RobotCommand) -> None:
        """Process dual-arm end-effector pose commands with IK.

        Solves IK for both arms simultaneously and applies safety limiting.

        Args:
            command: RobotCommand with EE pose targets for both arms
        """
        left_arm_data = command.input_components.pop("left_arm")
        right_arm_data = command.input_components.pop("right_arm")

        if "pose" not in left_arm_data or "pose" not in right_arm_data:
            return

        # Extract poses
        left_pose = np.array(left_arm_data.get("pose", np.eye(4)))
        right_pose = np.array(right_arm_data.get("pose", np.eye(4)))
        target_poses = {"L_ee": left_pose, "R_ee": right_pose}

        # Solve IK
        with suppress_loguru_module("dexmotion", enabled=True):
            solution, is_collision, within_limits = self.motion_manager.ik(
                target_pose=target_poses, type="pink"
            )

        if not solution or not within_limits or is_collision:
            return

        # Get arm processors for safety limiting
        left_processor = self.processors["left_arm"]
        right_processor = self.processors["right_arm"]

        # Extract and limit joint positions
        left_positions = [solution[f"L_arm_j{i + 1}"] for i in range(7)]
        right_positions = [solution[f"R_arm_j{i + 1}"] for i in range(7)]

        safe_left = left_processor.limit_joint_step(left_positions)
        safe_right = right_processor.limit_joint_step(right_positions)

        # Update motion manager
        left_processor.apply_positions(safe_left)
        right_processor.apply_positions(safe_right)

        # Update command
        command.output_components["left_arm"] = {
            "pos": safe_left.tolist(),
            "mode": CommandMode.ABSOLUTE.value,
        }
        command.output_components["right_arm"] = {
            "pos": safe_right.tolist(),
            "mode": CommandMode.ABSOLUTE.value,
        }

    def publish_command(self, command: RobotCommand) -> None:
        """Publish validated command to robot controller.

        Converts RobotCommand to dictionary format and publishes
        via Zenoh. Only output components are sent.

        Args:
            command: Validated RobotCommand to publish.

        Message Format:
            {
                'timestamp_ns': int,
                'components': {component_name: component_data},
                'safety_flags': {flag_name: bool}
            }
        """
        # Convert to publishable format
        command_dict = {
            "timestamp_ns": command.timestamp_ns,
            "components": {},
            "safety_flags": {
                "emergency_stop": command.safety_flags.emergency_stop,
                "exit_requested": command.safety_flags.exit_requested,
                # Distinguishes "the operator is holding E-Stop" from "the
                # re-alignment gate is holding it". Without this the UI cannot
                # explain why releasing the joycon does not resume motion, and
                # the E-Stop clear button just reports a generic refusal.
                "pending_realign": self._pending_realign,
            },
        }

        # Convert numpy arrays to lists (only output components are published)
        for name, data in command.output_components.items():
            component_data = {}

            # Handle position data
            if "pos" in data:
                component_data["pos"] = (
                    data["pos"].tolist()
                    if isinstance(data["pos"], np.ndarray)
                    else data["pos"]
                )

            # Handle velocity data
            if "vel" in data:
                component_data["vel"] = (
                    data["vel"].tolist()
                    if isinstance(data["vel"], np.ndarray)
                    else data["vel"]
                )

            # Add mode if present
            if "mode" in data:
                component_data["mode"] = data["mode"]

            # Add other fields (for chassis: vx, vy, wz)
            for key in data:
                if key not in ["pos", "vel", "mode"]:
                    component_data[key] = data[key]

            command_dict["components"][name] = component_data

        # Convert numpy types to native Python types (DictDataCodec uses orjson
        # without OPT_SERIALIZE_NUMPY, so numpy types aren't auto-handled)
        command_dict = self._to_json_serializable(command_dict)

        # Publish through Dexcomm
        self.safe_command_pub.publish(command_dict)

    @staticmethod
    def _to_json_serializable(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types.

        DictDataCodec uses orjson without OPT_SERIALIZE_NUMPY flag,
        so numpy scalars (float64, int64, etc.) cause TypeError.
        """
        if isinstance(obj, dict):
            return {
                k: CommandProcessor._to_json_serializable(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [CommandProcessor._to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _is_exo_aligned_with_robot(self, command: RobotCommand) -> bool:
        """Checks whether exo arm joints are within threshold of the robot's
        current joint positions.

        Only runs when teleop_mode is 'exo_joycon'. For other modes (e.g. VR)
        alignment is not required and this returns True immediately.

        Compares exo against the live robot joints (not config init_pos), so
        the check is valid regardless of where the robot physically is when
        teleop starts.

        Args:
            command: Current RobotCommand containing exo input_components.

        Returns:
            True if aligned (or mode doesn't require check), False otherwise.
        """
        if self.teleop_mode != "exo_joycon":
            return True

        with self._robot_joints_lock:
            robot_joints_snapshot = (
                self.latest_robot_joints.copy() if self.latest_robot_joints else None
            )
            received_at = self._robot_joints_at

        if robot_joints_snapshot is None or received_at is None:
            return False

        # A frozen snapshot from a dead robot_controller must not read as
        # "aligned" — that would resume motion against an unknown robot pose.
        age = time.monotonic() - received_at
        if age > self._robot_joints_max_age_s:
            key = "stale_feedback"
            if abs(age - self._last_align_log.get(key, -999)) > 0.5:
                self._last_align_log[key] = age
                logger.warning(
                    f"[ALIGN] robot joint feedback is {age:.1f}s old "
                    f"(max {self._robot_joints_max_age_s:.1f}s) — cannot verify "
                    "alignment"
                )
            return False

        joint_data = robot_joints_snapshot.get("joints", {})

        checks = [
            ("left_arm", joint_data.get("left_arm", [])),
            ("right_arm", joint_data.get("right_arm", [])),
        ]
        for arm_name, robot_pos in checks:
            if not robot_pos or len(robot_pos) < 7:
                return False
            arm_data = command.input_components.get(arm_name, {})
            exo_pos = arm_data.get("pos", [])
            if not exo_pos or len(exo_pos) < 7:
                return False
            for i, (exo_val, robot_val) in enumerate(zip(exo_pos, robot_pos)):
                diff = abs(exo_val - robot_val)
                if diff > self._align_threshold:
                    key = f"{arm_name}_{i}"
                    prev = self._last_align_log.get(key, -999)
                    if abs(diff - prev) > 0.01:
                        logger.warning(
                            f"[ALIGN] {arm_name} joint {i + 1} not aligned: "
                            f"exo={exo_val:.3f} robot={robot_val:.3f} diff={diff:.3f} "
                            f"(threshold {self._align_threshold})"
                        )
                        self._last_align_log[key] = diff
                    return False
        self._last_align_log.clear()
        return True

    def _sync_motion_manager_to_robot_state(self) -> None:
        """Synchronize motion manager with actual robot joint positions.

        Updates motion manager's internal state to match the real robot's
        current configuration. Called when motion control starts.

        Uses component processors to sync each component.
        """
        with self._robot_joints_lock:
            latest_robot_joints = (
                self.latest_robot_joints.copy() if self.latest_robot_joints else None
            )

        if latest_robot_joints is None:
            return

        joint_data = latest_robot_joints.get("joints", {})
        if not joint_data:
            return

        # Sync each component via its processor
        for component_name, processor in self.processors.items():
            if processor.is_enabled():
                processor.sync_to_robot_state(joint_data)

    def run(self) -> None:
        """Execute main command processing loop.

        Runs at configured rate (default 40Hz), processing commands
        and publishing validated outputs until shutdown.

        Processing Steps:
            1. Wait for robot joint data initialization
            2. Process teleoperation commands
            3. Apply safety validation
            4. Publish validated commands
            5. Handle exit requests
            6. Update debug display (if enabled)
        """
        self.running = True
        rate_hz = self.config.get_rate("command_rate", 40)
        rate_limiter = RateLimiter(rate_hz)

        logger.info(f"Starting command & safety processing at {rate_hz}Hz")

        # Start the live display if debug mode
        if self._debug_display:
            self._debug_display.start()

        while not self._robot_joint_data_initialized:
            time.sleep(0.1)

        while self.running:
            command = self.process_commands()

            if command:
                if not self.bypass_real_robot:
                    self.publish_command(command)

                # Check if we should exit after publishing
                if self.exit_after_publish or command.safety_flags.exit_requested:
                    logger.critical("EXIT SIGNAL SENT - Shutting down all systems")
                    self.running = False
                    time.sleep(0.2)  # Small delay to ensure message is sent
                    break

                # Update debug display
                if self._debug_display:
                    # Convert safety flags to dict
                    safety_flags_dict = {
                        "collision_detected": command.safety_flags.collision_detected,
                        "limits_enforced": command.safety_flags.limits_enforced,
                        "emergency_stop_active": command.safety_flags.emergency_stop,
                    }
                    self._debug_display.print_robot_command(
                        command.output_components, safety_flags_dict
                    )

            # Maintain rate
            rate_limiter.sleep()

        # Cleanup on exit
        logger.info("Command processor loop exited, cleaning up...")
        self.stop()

    def stop(self) -> None:
        """Shutdown command processor cleanly.

        Stops processing loop, cleans up resources, and shuts down
        communication channels.
        """
        self.running = False

        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        # Clean up input handler
        if self.input_handler:
            self.input_handler.cleanup()

        self.node.shutdown()  # Dexcomm Node cleanup
        logger.info("CommandProcessor stopped")


def main(
    namespace: str = "",
    debug: bool = False,
    visualize: bool = False,
    config_name: Optional[str] = None,
    lock_hand_collision_avoidance: bool = False,
) -> None:
    """Main entry point for command processor.

    Creates and runs the command processor with specified configuration.

    Args:
        namespace: Zenoh namespace for topic isolation.
        debug: Whether to enable debug display output.
        visualize: Whether to visualize the robot.
        config_name: Name of config file (without .yaml extension).
        lock_hand_collision_avoidance: Freeze hand joints at the configured
            "open" pose in the MotionManager so the collision checker ignores
            finger motion. Joycon hand commands still reach the robot.

    Example:
        $ omni-cmd --namespace /robot1 --debug
    """
    # Setup logging
    logger = setup_logging(debug)
    # Create and run processor (config loaded automatically)
    processor = CommandProcessor(
        namespace=namespace,
        debug=debug,
        visualize=visualize,
        config_name=config_name,
        lock_hand_collision_avoidance=lock_hand_collision_avoidance,
    )

    exit_requested = False
    try:
        processor.run()
        exit_requested = processor.exit_after_publish
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt...")
        processor.stop()

    # Force exit to ensure all threads terminate
    if exit_requested:
        import os

        logger.info("Forcing process exit due to exit request")
        os._exit(0)  # Force immediate exit without cleanup (all threads killed)


if __name__ == "__main__":
    import sys
    import tyro

    sys.exit(tyro.cli(main))
