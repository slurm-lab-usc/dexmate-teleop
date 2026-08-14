#!/usr/bin/env python3
"""Robot controller with interpolation and hardware control."""

import os
import signal
import sys
import time
import numpy as np
import ruckig
from typing import Optional, Dict, List
import tyro
from enum import Enum

from dexcomm import Node
from dexcomm.utils import RateLimiter
from dexcontrol.robot import Robot
from omniteleop import LIB_PATH
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.filters import MultiChannelFilter
from omniteleop.common.trajectory_interpolator import TrajectoryInterpolator
from omniteleop.common.ruckig_trajectory import (
    RuckigArmTrajectoryGenerator,
    RuckigTorsoTrajectoryGenerator,
)
from omniteleop.common.joint_state_safety import has_active_joint_error
from omniteleop.common.tracking_watchdog import DelayedTrackingWatchdog
from omniteleop.common.trajectory_safety import limit_sampled_joint_speed
from loguru import logger
import threading
from dexbot_utils import RobotInfo
from dexcomm.codecs import DictDataCodec


class RobotMode(Enum):
    RUNNING = "running"
    STOP = "stop"
    EXIT = "exit"


class RobotController:
    """Robot controller with integrated motion interpolation and hardware control."""

    ARM_DOF = 7
    MODE_HOLD_SECONDS = 0.5
    MODE_HOLD_DRIFT_TOL_RAD = 0.02
    TRAJECTORY_TRACKING_TOL_RAD = 0.10
    # Real-time Teleop commands intentionally use a much wider, sustained
    # tracking threshold than planned homing. A moving arm normally trails the
    # leader by a few hundredths of a radian; treating that servo lag as a hard
    # fault makes normal Teleop stop. Gross failures (for example, a stuck
    # joint) are still caught, while live joint errors and position limits are
    # checked independently on every command.
    REALTIME_TRACKING_TOL_RAD = 0.30
    TRACKING_REFERENCE_DELAY_SECONDS = 0.25
    TRACKING_VIOLATION_DURATION_SECONDS = 0.75
    TRACKING_MINIMUM_PROGRESS_RAD = 0.01
    FINAL_REACH_TIMEOUT_SECONDS = 2.0
    HOMING_MAX_JOINT_SPEED_RAD_S = 0.35
    WAYPOINT_CATCHUP_TIMEOUT_SECONDS = 2.0

    def __init__(
        self,
        namespace: str = "",
        interpolation_method: str = "none",
        history_size: int = 4,
        use_velocity_control: bool = False,
        debug: bool = False,
        publish_telemetry: bool = True,
        config_name: Optional[str] = None,
        no_arm_filter: bool = False,
    ) -> None:
        """Initialize robot controller.

        Args:
            namespace: Optional namespace prefix for Zenoh topics.
            interpolation_method: Method for interpolation ('none', 'linear', 'cubic').
            history_size: Number of past commands to keep for interpolation.
            use_velocity_control: Enable velocity control for smoother motion.
            debug: Enable debug output.
            publish_telemetry: Enable telemetry publishing for visualization.
            config_name: Name of the configuration file (without .yaml extension).
            no_arm_filter: If True, skip Butterworth filtering for left_arm and right_arm.

        Note:
            Input rate, control rate, and input topic are loaded from config file.
        """
        # Initialize Node (namespace is handled automatically by Node)
        self.node = Node(name="robot_controller", namespace=namespace)

        # Detect robot type (with or without torso/base)
        self.robot_info = RobotInfo()
        self.has_torso = self.robot_info.has_torso
        self.has_base = self.robot_info.has_chassis

        # Load configuration
        config_path = None
        if config_name is not None:
            config_path = LIB_PATH / "configs" / f"{config_name}.yaml"
        self.config = get_config(config_path)
        logger.info(f"Robot controller configured: {self.config}")

        self.input_rate = self.config.get_rate("input_rate", 40)
        self.control_rate = self.config.get_rate("control_rate", 100)
        self.input_topic = self.config.get_topic("robot_commands")
        self.interpolation_method = interpolation_method
        self.history_size = history_size
        self.use_velocity_control = use_velocity_control
        self.no_arm_filter = no_arm_filter

        # Communication
        self.subscriber = None
        self.joint_publisher = None

        # Robot hardware interface
        self.robot = None

        # Joint feedback publishing
        self.feedback_rate = self.config.get_rate("feedback_rate", 50)
        self.joint_publish_thread = None
        self.joint_publish_running = False

        # Rate limiter for precise timing
        self.rate_limiter = RateLimiter(self.control_rate)

        # Trajectory interpolator for smooth motion
        self.interpolator = TrajectoryInterpolator(
            method=interpolation_method,
            history_size=history_size,
        )

        # Latest command storage for safety checks and telemetry
        self.latest_command = None

        # Current state - dict based per component
        self.current_state = {}  # {component: {'pos': [...], 'vel': [...]}}
        self._arm_tracking_watchdog = DelayedTrackingWatchdog(
            tolerance_rad=self.REALTIME_TRACKING_TOL_RAD,
            reference_delay_s=self.TRACKING_REFERENCE_DELAY_SECONDS,
            violation_duration_s=self.TRACKING_VIOLATION_DURATION_SECONDS,
            minimum_progress_rad=self.TRACKING_MINIMUM_PROGRESS_RAD,
        )

        # Components that need interpolation
        self.interpolated_components = {"left_arm", "right_arm"}
        # Add torso only if robot has torso
        if self.has_torso:
            self.interpolated_components.add("torso")

        # Components that are passed through directly
        self.direct_components = {"head"}
        # Add chassis only if robot has base
        if self.has_base:
            self.direct_components.add("chassis")
        # Initialize multi-channel filter from config
        filter_config = self.config.get("filters", None)

        # Disable arm filtering when no_arm_filter is set.
        # Setting type="none" explicitly prevents fallback to the default
        # butterworth filter, which would still filter arms at 10 Hz.
        if self.no_arm_filter and filter_config is not None:
            components = filter_config.setdefault("components", {})
            for arm in ("left_arm", "right_arm"):
                components[arm] = {"type": "none"}

        # Create the multi-channel filter
        self.filter = MultiChannelFilter(
            filter_config=filter_config, control_rate=self.control_rate
        )

        # Determine hand type from ROBOT_CONFIG env var
        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1u_gripper")
        if "gripper" in robot_config:
            self.hand_type = "gripper"
        elif "f5d6" in robot_config:
            self.hand_type = "f5d6"
        else:
            self.hand_type = None  # No hands (e.g., vega_1, vega_1p, vega_1u)

        # Add hands to interpolated components only if robot has hands
        if self.hand_type is not None:
            self.direct_components.add("left_hand")
            self.direct_components.add("right_hand")

        # Log filter configuration
        if filter_config:
            default_type = filter_config.get("default", {}).get("type", "none")
            logger.info(f"Filter configuration loaded - default: {default_type}")
            if "components" in filter_config and filter_config["components"]:
                comp_filters = {}
                for comp, cfg in filter_config["components"].items():
                    comp_filters[comp] = cfg.get("type", "unknown")
                logger.info(f"Component-specific filters: {comp_filters}")

        # Robot state tracking

        # Store home positions for reuse
        self.home_positions = {}

        # Ruckig trajectory generators (initialized after home positions are known)
        self.ruckig_generators: Dict[
            str, RuckigArmTrajectoryGenerator | RuckigTorsoTrajectoryGenerator
        ] = {}

        self._robot_mode = None
        self._is_first_command = True
        self._stale_error_warned_at = 0.0
        # Rate-limited warnings for tracking faults / bad frames.
        self._tracking_fault_warned_at = 0.0
        self._bad_target_warned_at = 0.0
        self.exit_requested = False
        # Set by the SIGTERM/SIGINT handler so the shutdown path can tell an
        # operator-requested exit from a supervisor-requested one.
        self._termination_signal: Optional[int] = None
        self._cleanup_done = False

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "Robot", self.control_rate, refresh_rate=10
            )

        # Telemetry publishing for visualization
        self.publish_telemetry = publish_telemetry
        self.telemetry_publisher = None

        logger.info(
            f"Robot controller: {self.input_rate}Hz input -> {self.control_rate}Hz control"
        )

    def install_signal_handlers(self) -> None:
        """Route SIGTERM/SIGINT into the normal exit path.

        Python's default SIGTERM disposition kills the interpreter outright, so
        neither ``finally`` blocks nor ``atexit`` hooks run. The app backend
        stops teleop with ``killpg(SIGTERM)``, which means without this handler
        the arms are abandoned in position mode with no publisher and fall
        under gravity once the controller watchdog times out.
        """

        def _handle(signum, _frame):
            if self._termination_signal is None:
                self._termination_signal = signum
                logger.warning(
                    f"Received signal {signal.Signals(signum).name} — "
                    f"exiting via the safe-park shutdown path"
                )
            self.exit_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError) as exc:
                # Not the main thread, or the platform disallows it.
                logger.warning(f"Could not install {sig!r} handler: {exc}")

    def initialize(self) -> None:
        """Initialize communication and robot hardware.

        Sets up Zenoh publishers/subscribers and initializes the robot
        hardware interface. Moves robot to home position.
        """
        # Initialize subscriber through Node (namespace handled automatically)
        self.subscriber = self.node.create_subscriber(
            self.input_topic,
            callback=self._on_safe_command,
            decoder=DictDataCodec.decode,
        )

        # Initialize joint feedback publisher
        joint_topic = self.config.get_topic("robot_joints")
        self.joint_publisher = self.node.create_publisher(
            joint_topic,
            encoder=DictDataCodec.encode,
        )
        logger.info(
            f"Publishing robot joints to {self.node.resolve_topic(joint_topic)} "
            f"at {self.feedback_rate}Hz"
        )

        # Initialize telemetry publisher for visualization
        if self.publish_telemetry:
            telemetry_topic = self.config.get_topic("telemetry")
            self.telemetry_publisher = self.node.create_publisher(
                telemetry_topic,
                encoder=DictDataCodec.encode,
            )
            logger.info(
                f"Publishing telemetry to {self.node.resolve_topic(telemetry_topic)}"
            )

        # Keep this process's handlers installed. Robot's default handlers call
        # sys.exit() and would overwrite the exit-request path below.
        logger.info("Initializing robot hardware...")
        self.robot = Robot(
            auto_shutdown=False,
            configure_default_state=False,
        )

        # Parse home positions from config (no hardware movement)
        self._parse_home_positions()
        
        logger.info("Disable torso auto-idle mode")
        if self.has_torso:
            self.robot.torso.set_idle_mode(False)
        else:
            logger.warning("Robot has no torso, cannot disable auto-idle mode")

        if self.exit_requested:
            logger.warning("Exit requested during initialization — skipping homing")
            return

        if self.interpolation_method == "ruckig":
            self._init_ruckig_generators()
            self._move_to_home_with_ruckig()
        else:
            self._move_to_home()

        self._robot_mode = RobotMode.RUNNING

        logger.success("Robot controller initialized")

    def _ensure_arms_in_position_mode(self) -> None:
        """Re-arm both arms for position control before any homing motion.

        ``Robot._set_default_state()`` sets position mode during ``Robot()``,
        but it returns early when the software E-Stop is active — and a previous
        session's park leaves the arms in ``disable`` mode. Without this, homing
        would publish position commands that the arms ignore. Must be called
        after ``estop.deactivate()``.
        """
        failures = []
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(self.robot, arm_name, None)
            if arm is None:
                continue
            try:
                brake_status = arm.get_brake_status()
                if (
                    not isinstance(brake_status, dict)
                    or brake_status.get("success") is not True
                ):
                    raise RuntimeError(
                        f"brake status query failed: {brake_status}"
                    )
                if bool(brake_status.get("enabled", False)):
                    raise RuntimeError(
                        "brake release is active for joints "
                        f"{brake_status.get('joints', [])}"
                    )
                response = arm.set_modes(["position"] * self.ARM_DOF)
                if not isinstance(response, dict) or response.get("success") is not True:
                    raise RuntimeError(f"mode service rejected request: {response}")
            except Exception as exc:  # pylint: disable=broad-except
                failures.append(f"{arm_name}: {exc}")
        if failures:
            raise RuntimeError(
                "Cannot enter position control; homing aborted (" + "; ".join(failures) + ")"
            )
        self._validate_live_arm_safety()
        self._establish_current_pose_hold()
        self._arm_tracking_watchdog.reset()

    def _validate_live_arm_safety(self) -> None:
        """Reject stale errors, non-finite positions, and URDF-limit violations."""
        violations = []
        now = time.monotonic()
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(self.robot, arm_name, None)
            if arm is None:
                continue
            state = arm._get_state()  # noqa: SLF001 - public getters omit errors
            errors = state.get("error")
            active = has_active_joint_error(errors)
            if errors and not active:
                # Historical note (error_code==0/severity==0) — e.g. the motor
                # rejected an out-of-limit command earlier and already
                # recovered. Warn (rate-limited) instead of killing the
                # controller over a self-resolved fault.
                if now - self._stale_error_warned_at > 10.0:
                    self._stale_error_warned_at = now
                    logger.warning(
                        f"{arm_name} reports stale error notes "
                        f"(no active fault): {errors}"
                    )
            elif active:
                violations.append(f"{arm_name} reports errors: {errors}")
            position = np.asarray(state.get("pos", []), dtype=float)
            if position.shape != (self.ARM_DOF,) or not bool(np.isfinite(position).all()):
                violations.append(f"{arm_name} position state is invalid")
                continue
            limits = arm.joint_pos_limit
            if limits is None or limits.shape != (self.ARM_DOF, 2):
                violations.append(f"{arm_name} joint limits are unavailable")
                continue
            outside = np.flatnonzero(
                (position < limits[:, 0] - 1e-3)
                | (position > limits[:, 1] + 1e-3)
            )
            for index in outside:
                violations.append(
                    f"{arm_name}[{int(index)}]={position[index]:.6f} outside "
                    f"[{limits[index, 0]:.6f}, {limits[index, 1]:.6f}]"
                )
        if violations:
            raise RuntimeError("Unsafe live arm state: " + "; ".join(violations))

    def _establish_current_pose_hold(self) -> None:
        """Prove both arms accept position commands before any planned motion."""
        arms = {
            name: getattr(self.robot, name)
            for name in ("left_arm", "right_arm")
            if getattr(self.robot, name, None) is not None
        }
        targets = {
            name: np.asarray(arm.get_joint_pos(), dtype=float).copy()
            for name, arm in arms.items()
        }
        rate = RateLimiter(self.control_rate)
        deadline = time.monotonic() + self.MODE_HOLD_SECONDS
        while time.monotonic() < deadline:
            for name, arm in arms.items():
                arm.set_joint_pos(targets[name], wait_time=0.0)
            rate.sleep()
        self._validate_live_arm_safety()
        drift = {
            name: float(
                np.max(
                    np.abs(
                        np.asarray(arm.get_joint_pos(), dtype=float)
                        - targets[name]
                    )
                )
            )
            for name, arm in arms.items()
        }
        failed = {
            name: value
            for name, value in drift.items()
            if value > self.MODE_HOLD_DRIFT_TOL_RAD
        }
        if failed:
            raise RuntimeError(f"Arm hold verification failed: {failed}")

    def _init_ruckig_generators(self) -> None:
        """Initialize per-component ruckig trajectory generators from home positions."""
        control_cycle = 1.0 / self.control_rate

        for component, home_pos in self.home_positions.items():
            qpos = np.array(home_pos)
            if component in {"left_arm", "right_arm"}:
                self.ruckig_generators[component] = RuckigArmTrajectoryGenerator(
                    init_qpos=qpos,
                    control_cycle=control_cycle,
                    safety_factor=1.0,
                )
            elif component == "torso":
                self.ruckig_generators[component] = RuckigTorsoTrajectoryGenerator(
                    init_qpos=qpos,
                    control_cycle=control_cycle,
                )
            # head, hands, chassis are passed through directly

        logger.info(
            f"Ruckig generators initialized for: {list(self.ruckig_generators.keys())}"
        )

    def _move_to_home_with_ruckig(self) -> None:
        """Drive arms and torso to home position using ruckig.

        Seeds each generator from actual hardware position, sets home as target,
        then steps the ruckig loop until ruckig reports Finished for all components.
        """
        rate_limiter = RateLimiter(self.control_rate)

        # Seed generators from actual robot positions and set home as target
        actual_positions = self._get_robot_joint_pos()
        for component, gen in self.ruckig_generators.items():
            current = np.array(
                actual_positions.get(component, self.home_positions[component])
            )
            target = np.array(self.home_positions[component])
            gen.reset(current)
            gen.inp.target_position = target.tolist()
            gen.inp.target_velocity = [0.0] * gen.dof
            gen.inp.target_acceleration = [0.0] * gen.dof

        logger.info("Moving to home position with ruckig...")
        self.robot.estop.deactivate()
        time.sleep(2.0)
        self._ensure_arms_in_position_mode()

        # Track per-component completion
        finished = {comp: False for comp in self.ruckig_generators}
        self._arm_tracking_watchdog.reset()

        final_arm_targets: dict[str, np.ndarray] = {}
        while not all(finished.values()):
            if self.exit_requested:
                logger.warning("Exit requested mid-homing — stopping ruckig homing")
                return
            for component, gen in self.ruckig_generators.items():
                if finished[component]:
                    continue

                result = gen.otg.update(gen.inp, gen.out)
                gen.out.pass_to_input(gen.inp)
                cmd_pos = np.array(gen.out.new_position)

                if component == "torso":
                    self.robot.torso.move_to_joint_pos(cmd_pos)
                elif component in {"left_arm", "right_arm"}:
                    getattr(self.robot, component).set_joint_pos(cmd_pos, wait_time=0.0)
                    final_arm_targets[component] = cmd_pos

                if result != ruckig.Result.Working:
                    finished[component] = True

            self._validate_live_arm_safety()
            self._require_tracking(final_arm_targets)
            rate_limiter.sleep()

        self._wait_for_final_arm_targets(final_arm_targets)

        # Open hands only if robot has hands
        if self.hand_type is not None:
            self.robot.left_hand.open_hand()
            self.robot.right_hand.open_hand()
        self.robot.estop.activate()
        logger.info("Home position reached via ruckig")

    def _get_robot_joint_pos(self) -> Dict[str, List[float]]:
        """Get robot joint positions.

        Returns:
            Dictionary mapping component names to joint position lists.
        """
        positions = {
            "left_arm": self.robot.left_arm.get_joint_pos().tolist(),
            "right_arm": self.robot.right_arm.get_joint_pos().tolist(),
            "head": self.robot.head.get_joint_pos().tolist(),
        }

        # Add torso only if robot has torso
        if self.has_torso:
            positions["torso"] = self.robot.torso.get_joint_pos().tolist()

        # Add hands only if robot has hands
        if self.hand_type is not None:
            positions["left_hand"] = self.robot.left_hand.get_joint_pos().tolist()
            positions["right_hand"] = self.robot.right_hand.get_joint_pos().tolist()

        return positions

    def _get_robot_joint_vel(self) -> Dict[str, List[float]]:
        """Get robot joint velocities.

        Returns:
            Dictionary mapping component names to joint velocity lists.
        """
        velocities = {
            "left_arm": self.robot.left_arm.get_joint_vel().tolist(),
            "right_arm": self.robot.right_arm.get_joint_vel().tolist(),
            "head": self.robot.head.get_joint_vel().tolist(),
        }

        # Add torso only if robot has torso
        if self.has_torso:
            velocities["torso"] = self.robot.torso.get_joint_vel().tolist()

        return velocities

    def _parse_init_pos(self, component, init_pos: str | list[float]) -> list[float]:
        """Parse initial position from string or list.

        Args:
            component: Robot component with get_predefined_pose method.
            init_pos: Either a predefined pose name or list of joint positions.

        Returns:
            List of joint positions.
        """
        if isinstance(init_pos, str):
            return component.get_predefined_pose(init_pos)
        else:
            return init_pos

    def _parse_home_positions(self) -> None:
        """Parse home positions from config and store for reuse. No hardware movement."""
        init_pos_config = self.config.get("init_pos", {})

        for arm in ["left_arm", "right_arm"]:
            if hasattr(self.robot, arm):
                init_pos = init_pos_config.get(arm)
                if init_pos is not None:
                    robot_component = getattr(self.robot, arm)
                    self.home_positions[arm] = self._parse_init_pos(
                        robot_component, init_pos
                    )

        other_components = ["head", "left_hand", "right_hand"]
        if self.has_torso:
            other_components.insert(0, "torso")

        for component in other_components:
            if hasattr(self.robot, component):
                init_pos = self.config.get_init_pos(component)
                if init_pos is not None:
                    robot_component = getattr(self.robot, component)
                    self.home_positions[component] = self._parse_init_pos(
                        robot_component, init_pos
                    )

        logger.debug(f"Home positions parsed: {list(self.home_positions.keys())}")

    def _move_to_home(self) -> None:
        """Move robot to home using collision-aware OMPL planning for the arms.

        Flow: move torso directly, plan arms with OMPL, set head directly,
        open hands. Raises if planning fails rather than falling back to an
        interpolation that could sweep through a self-collision.
        """
        self.robot.estop.deactivate()
        time.sleep(0.1)
        self._ensure_arms_in_position_mode()

        self._move_torso_to_home_direct()
        self._plan_and_execute_arms_to_home()
        self._move_head_to_home_direct()

        if self.hand_type is not None:
            self.robot.left_hand.open_hand()
            self.robot.right_hand.open_hand()
        self.robot.estop.activate()

    def _move_torso_to_home_direct(self) -> None:
        """Send torso directly to its home position (no planning)."""
        if not (self.has_torso and "torso" in self.home_positions):
            return
        self.robot.move.torso.move_to_joint_pos(
            self.home_positions["torso"],
        )

    def _move_head_to_home_direct(self) -> None:
        """Send head directly to its home position (no planning)."""
        if "head" not in self.home_positions:
            return
        self.robot.head.set_joint_pos(
            self.home_positions["head"], wait_time=2.0, exit_on_reach=True
        )

    def _plan_and_execute_arms_to_home(self) -> None:
        """Plan a collision-free trajectory for both arms to their home pose.

        Uses a temporary MotionManager + OMPL. The planner's pinocchio model
        only knows about arm joints, so start/goal dicts passed to
        MoveToConfigurationTask.run() must be arms-only — even though the
        MotionManager itself is initialized with torso in its config so
        collision checks account for the current torso pose.
        """
        has_left = "left_arm" in self.home_positions
        has_right = "right_arm" in self.home_positions
        if not (has_left or has_right):
            return

        from dexmotion.motion_manager import MotionManager
        from dexmotion.tasks.move_out_of_self_collision_task import (
            MoveOutOfSelfCollisionTask,
        )
        from dexmotion.tasks.move_to_configuration_task import (
            MoveToConfigurationTask,
        )
        from dexmotion.utils import robot_utils

        # Initial config for MotionManager includes torso so collision checks
        # see the correct base pose; arms+torso is the same set of components
        # arm_safe_initializer uses.
        mm_init_components = ["left_arm", "right_arm"]
        if self.has_torso:
            mm_init_components.append("torso")
        mm_init_config = self.robot.get_joint_pos_dict(mm_init_components)

        motion_manager = MotionManager(
            init_visualizer=False,
            initial_joint_configuration_dict=mm_init_config,
            init_local_ik=False,
        )

        # Arms-only dicts for the planner. Its pinocchio model's joint_names
        # is the authoritative set — start/goal keys must exactly equal it.
        assert motion_manager.pin_robot is not None
        planner_joint_names = robot_utils.get_joint_names(motion_manager.pin_robot)

        # Resolve any existing self-collision before planning the main motion.
        # MoveOutOfSelfCollisionTask only *computes* a collision-free config; it
        # does not command the robot. We must physically move the arms there,
        # otherwise the OMPL start state below is still in collision and
        # RRTConnect rejects it ("invalid start state").
        present, escape_success, escape_cfg = MoveOutOfSelfCollisionTask(
            initial_joint_configuration=mm_init_config,
            motion_manager=motion_manager,
            range_size=0.3,
            visualize=False,
        ).run()
        if present:
            if not escape_success:
                raise RuntimeError(
                    "Robot is in self-collision and no collision-free "
                    "configuration could be found"
                )
            # escape_cfg is keyed by the pinocchio joint names, i.e. the same
            # planner_joint_names used everywhere here.
            escape_arms = {
                name: float(escape_cfg[name])
                for name in planner_joint_names
                if name in escape_cfg
            }
            # Interpolate from the in-collision start (the config the task was
            # initialized with, arms-only) to the collision-free config and move
            # the real arms there — same approach as dexcontrol's
            # ArmSafeInitializer._handle_self_collisions.
            in_collision_arms = {
                name: float(mm_init_config[name])
                for name in planner_joint_names
                if name in mm_init_config
            }
            logger.info("Moving arms out of self-collision before planning...")
            self._move_arms_out_of_collision(
                in_collision_arms, escape_arms, planner_joint_names
            )
            logger.info("Waiting for robot to stabilize...")
            time.sleep(3)

        # Seed the OMPL start from the live robot position. When we escaped a
        # collision above we waited for the arms to settle first, so this
        # readback reflects the real (now collision-free) pose. Sync the
        # MotionManager's collision model to the same live pose.
        start_arms = self._current_arms_qpos_dict(planner_joint_names)
        motion_manager.set_joint_pos(start_arms)
        goal_arms = self._home_arms_qpos_dict(planner_joint_names)

        planner_task = MoveToConfigurationTask(
            initial_joint_configuration=start_arms,
            motion_manager=motion_manager,
            planner_type="ompl",
            visualize=False,
        )

        logger.info("Planning collision-free path to home (OMPL)...")
        try:
            _ts, qs_sample, _qds, _qdds, _dur = planner_task.run(
                start_configuration_dict=start_arms,
                goal_configuration_dict=goal_arms,
                control_frequency=self.control_rate,
                generate_trajectory=True,
            )
        except Exception as exc:
            raise RuntimeError(f"OMPL planning to home failed: {exc}") from exc

        if qs_sample is None or len(qs_sample) == 0:
            raise RuntimeError("OMPL planning returned an empty trajectory")

        qs_sample, scaling = limit_sampled_joint_speed(
            qs_sample,
            control_frequency=self.control_rate,
            max_joint_speed_rad_s=self.HOMING_MAX_JOINT_SPEED_RAD_S,
        )
        logger.info(
            "Safety time-scaling home trajectory: "
            f"{scaling.original_waypoints} -> {scaling.scaled_waypoints} waypoints, "
            f"planned peak={scaling.original_peak_speed_rad_s:.3f} rad/s, "
            f"limit={self.HOMING_MAX_JOINT_SPEED_RAD_S:.3f} rad/s, "
            f"scale={scaling.scale_factor:.2f}x"
        )

        self._execute_arm_trajectory(qs_sample, planner_joint_names)
        logger.info("Robot moved to home position via OMPL planner")

    def _current_arms_qpos_dict(self, planner_joint_names: List[str]) -> Dict[str, float]:
        """Return current arm joint positions keyed by planner joint names."""
        current: Dict[str, float] = {}
        live = self.robot.get_joint_pos_dict(["left_arm", "right_arm"])
        # Copy only entries the planner knows about — drops anything not in
        # the pinocchio model (e.g. torso joints that slip through).
        for name in planner_joint_names:
            if name in live:
                current[name] = float(live[name])
        return current

    def _home_arms_qpos_dict(self, planner_joint_names: List[str]) -> Dict[str, float]:
        """Return home arm joint positions keyed by planner joint names."""
        goal: Dict[str, float] = {}
        for arm_key in ("left_arm", "right_arm"):
            if arm_key not in self.home_positions:
                continue
            arm = getattr(self.robot, arm_key)
            for joint_name, pos in zip(arm.joint_name, self.home_positions[arm_key]):
                if joint_name in planner_joint_names:
                    goal[joint_name] = float(pos)
        # If an arm lacks a configured home, pin it at its current position
        # so start_qpos.keys() == goal_qpos.keys() still holds.
        live = self.robot.get_joint_pos_dict(["left_arm", "right_arm"])
        for name in planner_joint_names:
            if name not in goal and name in live:
                goal[name] = float(live[name])
        return goal

    def _execute_arm_trajectory(
        self, qs_sample, planner_joint_names: List[str]
    ) -> None:
        """Stream planned waypoints to both arms at self.control_rate."""
        rate_limiter = RateLimiter(self.control_rate)
        logger.info(
            f"Executing planned home trajectory ({len(qs_sample)} waypoints)..."
        )
        left_names = self.robot.left_arm.joint_name
        right_names = self.robot.right_arm.joint_name
        has_left = "left_arm" in self.home_positions
        has_right = "right_arm" in self.home_positions
        final_targets: dict[str, np.ndarray] = {}
        for qpos_array in qs_sample:
            if self.exit_requested:
                logger.warning(
                    "Exit requested mid-trajectory — stopping planned home motion"
                )
                return
            waypoint = dict(zip(planner_joint_names, qpos_array))
            if has_left:
                left_target = np.array([waypoint[j] for j in left_names])
                final_targets["left_arm"] = left_target
            if has_right:
                right_target = np.array([waypoint[j] for j in right_names])
                final_targets["right_arm"] = right_target
            self._follow_planned_waypoint(final_targets, rate_limiter)
        self._wait_for_final_arm_targets(final_targets)

    def _follow_planned_waypoint(
        self,
        targets: Dict[str, np.ndarray],
        rate_limiter: RateLimiter,
    ) -> None:
        """Pause the collision-planned path while an arm catches up."""
        catchup_started: float | None = None
        while True:
            if self.exit_requested:
                return
            for name, target in targets.items():
                getattr(self.robot, name).set_joint_pos(target, wait_time=0.0)
            self._validate_live_arm_safety()

            actuals: dict[str, np.ndarray] = {}
            for name, target in targets.items():
                actual = np.asarray(
                    getattr(self.robot, name).get_joint_pos(),
                    dtype=float,
                )
                if actual.shape != target.shape or not bool(np.isfinite(actual).all()):
                    raise RuntimeError(f"{name} returned invalid position state")
                actuals[name] = actual
            errors = {
                name: float(np.max(np.abs(actuals[name] - target)))
                for name, target in targets.items()
            }
            if all(
                error <= self.TRAJECTORY_TRACKING_TOL_RAD
                for error in errors.values()
            ):
                rate_limiter.sleep()
                return

            now = time.monotonic()
            if catchup_started is None:
                catchup_started = now
            elif now - catchup_started >= self.WAYPOINT_CATCHUP_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "Arm waypoint catch-up timed out: "
                    + self._format_position_target_errors(targets, actuals)
                )
            rate_limiter.sleep()

    def _format_position_target_errors(
        self,
        targets: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
    ) -> str:
        formatted: dict[str, object] = {}
        for name, target in targets.items():
            differences = np.abs(actuals[name] - target)
            joint_index = int(np.argmax(differences))
            arm = getattr(self.robot, name)
            formatted[name] = {
                "joint": arm.joint_name[joint_index],
                "error_rad": round(float(differences[joint_index]), 6),
                "actual_rad": round(float(actuals[name][joint_index]), 6),
                "target_rad": round(float(target[joint_index]), 6),
            }
        return str(formatted)

    def _require_tracking(self, targets: Dict[str, np.ndarray]) -> None:
        actuals = {
            name: np.asarray(
                getattr(self.robot, name).get_joint_pos(),
                dtype=float,
            )
            for name in targets
        }
        watchdog = getattr(self, "_arm_tracking_watchdog", None)
        if watchdog is None:
            watchdog = DelayedTrackingWatchdog(
                tolerance_rad=self.REALTIME_TRACKING_TOL_RAD,
                reference_delay_s=self.TRACKING_REFERENCE_DELAY_SECONDS,
                violation_duration_s=self.TRACKING_VIOLATION_DURATION_SECONDS,
                minimum_progress_rad=self.TRACKING_MINIMUM_PROGRESS_RAD,
            )
            self._arm_tracking_watchdog = watchdog
        errors = watchdog.update(targets, actuals)
        if errors:
            raise RuntimeError(
                "Arm trajectory tracking failed: "
                + self._format_tracking_failures(watchdog, errors)
            )

    def _format_tracking_failures(
        self,
        watchdog: DelayedTrackingWatchdog,
        errors: Dict[str, float],
    ) -> str:
        """Format arm tracking errors with the responsible physical joint."""
        details = getattr(watchdog, "last_failure_details", {})
        formatted: dict[str, object] = {}
        for arm_name, error in errors.items():
            detail = details.get(arm_name)
            arm = getattr(self.robot, arm_name, None)
            joint_names = getattr(arm, "joint_name", ())
            if detail is None or detail.joint_index >= len(joint_names):
                formatted[arm_name] = error
                continue
            formatted[arm_name] = {
                "joint": joint_names[detail.joint_index],
                "error_rad": round(detail.error_rad, 6),
                "actual_rad": round(detail.actual_rad, 6),
                "reference_rad": round(detail.reference_rad, 6),
            }
        return str(formatted)

    def _wait_for_final_arm_targets(
        self, targets: Dict[str, np.ndarray]
    ) -> None:
        rate = RateLimiter(self.control_rate)
        deadline = time.monotonic() + self.FINAL_REACH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            for name, target in targets.items():
                getattr(self.robot, name).set_joint_pos(target, wait_time=0.0)
            self._validate_live_arm_safety()
            errors = {
                name: float(
                    np.max(
                        np.abs(
                            np.asarray(
                                getattr(self.robot, name).get_joint_pos(),
                                dtype=float,
                            )
                            - target
                        )
                    )
                )
                for name, target in targets.items()
            }
            if all(
                error <= self.TRAJECTORY_TRACKING_TOL_RAD
                for error in errors.values()
            ):
                return
            rate.sleep()
        raise RuntimeError(f"Arms failed to reach final target: {errors}")
    def _move_arms_out_of_collision(
        self,
        start_arms: Dict[str, float],
        collision_free_arms: Dict[str, float],
        planner_joint_names: List[str],
        collision_escape_time: float = 2.0,
    ) -> None:
        """Move the real arms from their in-collision pose to a safe config.

        Mirrors ArmSafeInitializer._handle_self_collisions in dexcontrol
        (examples/advanced_examples/init_arm_safe.py): linearly interpolate
        between the start (in-collision) and collision-free configs over
        control_hz * collision_escape_time steps, stream to the real arms, then
        let the caller wait for the robot to stabilize before planning.

        Args:
            start_arms: In-collision arm positions keyed by planner joint names.
            collision_free_arms: Target safe positions, same keying.
            planner_joint_names: Authoritative joint-name ordering.
            collision_escape_time: Seconds over which to make the escape move.
        """
        joints = [
            n
            for n in planner_joint_names
            if n in start_arms and n in collision_free_arms
        ]
        if not joints:
            return

        num_steps = int(self.control_rate * collision_escape_time)
        trajectory: List[Dict[str, float]] = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            trajectory.append(
                {
                    n: (1 - alpha) * start_arms[n] + alpha * collision_free_arms[n]
                    for n in joints
                }
            )

        left_names = self.robot.left_arm.joint_name
        right_names = self.robot.right_arm.joint_name
        has_left = "left_arm" in self.home_positions
        has_right = "right_arm" in self.home_positions

        rate_limiter = RateLimiter(self.control_rate)
        for waypoint in trajectory:
            if has_left and all(j in waypoint for j in left_names):
                self.robot.left_arm.set_joint_pos(
                    np.array([waypoint[j] for j in left_names]), wait_time=0.0
                )
            if has_right and all(j in waypoint for j in right_names):
                self.robot.right_arm.set_joint_pos(
                    np.array([waypoint[j] for j in right_names]), wait_time=0.0
                )
            rate_limiter.sleep()

    def _on_safe_command(self, data: Dict) -> None:
        """Handle incoming safe command.

        Args:
            data: Command dictionary from command processor with components
                  and safety flags.
        """
        try:
            # Store full command for safety checks and telemetry
            self.latest_command = data

            # While emergency stop is active, drop the position payload: the
            # safety validator skips limit enforcement during estop, so a
            # paused leader position may exceed motor limits. Keeping it in
            # latest_command (direct-pass mode) or in the interpolator would
            # make the resume jump straight to that unclipped target and
            # trip a hardware "Position command limit exceeded" error.
            if data.get("safety_flags", {}).get("emergency_stop", False):
                data["components"] = {}
                return

            # Extract positions for interpolation
            timestamp = time.perf_counter()
            positions = {}

            cmd_components = data.get("components", {})
            for component, comp_data in cmd_components.items():
                # Only add interpolated components to the interpolator
                if component in self.interpolated_components and "pos" in comp_data:
                    positions[component] = np.array(comp_data["pos"])

            # Add to interpolator
            if positions:
                self.interpolator.add_point(timestamp, positions)

        except Exception as e:
            logger.error(f"Error processing safe command: {e}")

    def _compute_interpolated_command(self) -> Optional[Dict]:
        """Compute interpolated command for current time using TrajectoryInterpolator.

        Returns:
            Optional[Dict]: Command dictionary with interpolated components.
        """
        if not self.latest_command:
            return None

        current_time = time.perf_counter()

        # Check for exit signal first (highest priority)
        safety_flags = self.latest_command.get("safety_flags", {})
        if safety_flags.get("exit_requested", False):
            logger.critical("Exit signal received - shutting down immediately")
            self._robot_mode = RobotMode.EXIT
            self.exit_requested = True
            return None

        # Check for emergency stop
        emergency_stop = safety_flags.get("emergency_stop", False)

        if emergency_stop:
            if self._robot_mode != RobotMode.STOP:
                logger.warning("Emergency stop activated")
                self.robot.estop.activate()
                self._robot_mode = RobotMode.STOP
            return None  # Don't process any commands during estop
        elif self._robot_mode == RobotMode.STOP:
            # Emergency stop released
            logger.info("Emergency stop released")
            self.robot.estop.deactivate()
            self._ensure_arms_in_position_mode()
            self.robot.head.set_mode("enable")
            self.robot.head.set_joint_pos(
                self.home_positions["head"], wait_time=2.0, exit_on_reach=True
            )
            # Drop trajectory points accumulated before the pause so the
            # resume follows the fresh command flow instead of a stale,
            # possibly unclipped target.
            self.interpolator.clear()
            # Seed the interpolator with the robot's CURRENT actual joint
            # positions before the next exo update arrives. Without this, the
            # first point the interpolator sees is the exo position — which
            # may be far from where the robot stopped. The resulting velocity
            # jump can trip a firmware "Position command limit exceeded" error.
            now = time.perf_counter()
            for arm_name in ("left_arm", "right_arm"):
                if arm_name not in self.interpolated_components:
                    continue
                arm = getattr(self.robot, arm_name, None)
                if arm is None:
                    continue
                try:
                    pos = np.asarray(arm.get_joint_pos(), dtype=float)
                except Exception:
                    continue
                if pos.shape == (self.ARM_DOF,) and bool(np.isfinite(pos).all()):
                    self.interpolator.add_point(now, {arm_name: pos})
            self._robot_mode = RobotMode.RUNNING

        # Get command components
        cmd_components = self.latest_command.get("components", {})

        if not cmd_components:
            return None

        # Final output components
        output_components = {}

        # Handle interpolation based on method
        if self.interpolation_method == "none" or (
            self._is_first_command and self.interpolation_method != "ruckig"
        ):
            # No interpolation: pass through command directly
            for component, data in cmd_components.items():
                formatted_data = {}
                if "pos" in data:
                    formatted_data["pos"] = np.array(data["pos"])
                if "vel" in data:
                    formatted_data["vel"] = np.array(data["vel"])
                # Pass through other fields (like vx, vy, wz for base)
                for key in data:
                    if key not in ["pos", "vel"]:
                        formatted_data[key] = data[key]
                output_components[component] = formatted_data
        elif self.interpolation_method == "ruckig":
            # On first command, reset generators to incoming positions to avoid jumps
            if self._is_first_command:
                for component, gen in self.ruckig_generators.items():
                    if (
                        component in cmd_components
                        and "pos" in cmd_components[component]
                    ):
                        gen.reset(np.array(cmd_components[component]["pos"]))

            # Ruckig jerk-limited interpolation for arms and torso
            for component, gen in self.ruckig_generators.items():
                target = None
                if component in cmd_components and "pos" in cmd_components[component]:
                    target = np.array(cmd_components[component]["pos"])
                pos, vel = gen.update(target)
                output_components[component] = {"pos": pos, "vel": vel}

            # Pass through all other components directly (head, hands, chassis)
            ruckig_components = set(self.ruckig_generators.keys())
            for component, data in cmd_components.items():
                if component not in ruckig_components:
                    formatted_data = {}
                    if "pos" in data:
                        formatted_data["pos"] = np.array(data["pos"])
                    if "vel" in data:
                        formatted_data["vel"] = np.array(data["vel"])
                    for key in data:
                        if key not in ["pos", "vel"]:
                            formatted_data[key] = data[key]
                    output_components[component] = formatted_data
        else:
            # Use interpolator for smooth trajectories
            positions, velocities = self.interpolator.interpolate(
                current_time, compute_velocity=True
            )

            if positions and velocities:
                # Add interpolated components with positions and velocities
                for component in positions:
                    output_components[component] = {
                        "pos": positions[component],
                        "vel": velocities[component],
                    }

            # Add direct-pass components (e.g., chassis) from latest command
            for component, data in cmd_components.items():
                if component in self.direct_components:
                    formatted_data = {}
                    for key, value in data.items():
                        if isinstance(value, list):
                            formatted_data[key] = np.array(value)
                        else:
                            formatted_data[key] = value
                    output_components[component] = formatted_data

        # Apply smoothing filter only for linear/cubic interpolation
        if output_components and self.interpolation_method in ("linear", "cubic"):
            output_components = self.filter.apply(output_components)

        # Update current state
        self.current_state = output_components

        return {
            "components": output_components,
            "timestamp_ns": time.time_ns(),  # Absolute timestamp for logging/diagnostics
        }

    def _send_robot_command(self, command: Dict):
        """Send command to robot hardware.

        Args:
            command: Command dictionary with components.
        """
        try:
            components = command.get("components", {})

            if not components:
                logger.debug("No components to send")
                return

            # Process base command first (only if robot has base)
            if self.has_base and "chassis" in components:
                self._send_base_command(components["chassis"])

            # Process torso command (only if robot has torso)
            if self.has_torso and "torso" in components:
                self._send_torso_command(components["torso"])

            # Process head command
            if "head" in components:
                self._send_head_command(components["head"])

            # Match the original real-time Teleop behavior: validated leader
            # targets are sent directly. Clamping every target around measured
            # feedback creates an implicit speed limit and makes the follower
            # feel sluggish. CommandProcessor still enforces collision and
            # joint limits; hardware state and gross sustained tracking faults
            # remain checked below.
            arm_targets: dict[str, np.ndarray] = {}
            for name in ("left_arm", "right_arm"):
                if name not in components or "pos" not in components[name]:
                    continue
                try:
                    target = self._validate_realtime_arm_target(
                        name,
                        np.asarray(components[name]["pos"], dtype=float),
                    )
                except RuntimeError:
                    # One bad frame (NaN / wrong shape from a sensor glitch)
                    # must not kill the controller — skip this frame; the
                    # leader stream recovers on its own.
                    if time.monotonic() - self._bad_target_warned_at > 10.0:
                        self._bad_target_warned_at = time.monotonic()
                        logger.warning(
                            f"{name} invalid real-time target skipped"
                        )
                    continue
                # Send the clipped target, not the raw component value.
                components[name]["pos"] = target.tolist()
                arm_targets[name] = target
                self._send_arm_command(name, components[name])
            if arm_targets:
                self._validate_live_arm_safety()
                try:
                    self._require_tracking(arm_targets)
                except RuntimeError as exc:
                    # Tracking fault during live teleop: the arm is lagging
                    # behind the leader (motor at limit, mechanical stall,
                    # etc.). Keep sending clipped commands at full rate — the
                    # arm recovers on its own once the leader returns within
                    # range. Pausing commands would only starve the motor and
                    # make recovery impossible.
                    if time.monotonic() - self._tracking_fault_warned_at > 10.0:
                        self._tracking_fault_warned_at = time.monotonic()
                        logger.error(
                            "Arm tracking fault (still sending commands): " + str(exc)
                        )

            if "left_hand" in components:
                self._send_hand_command("left_hand", components["left_hand"])
            if "right_hand" in components:
                self._send_hand_command("right_hand", components["right_hand"])

        except Exception as e:
            logger.error(f"Failed to send robot command: {e}")
            self._robot_mode = RobotMode.STOP
            raise

    def _validate_realtime_arm_target(
        self,
        arm_name: str,
        desired: np.ndarray,
    ) -> np.ndarray:
        """Validate a Teleop target without changing its motion profile."""
        if desired.shape != (self.ARM_DOF,) or not bool(np.isfinite(desired).all()):
            raise RuntimeError(f"{arm_name} received an invalid real-time target")
        # Clip to the limits the motor itself enforces (queried from firmware).
        # The CommandProcessor clips to URDF model limits, which can be wider
        # than the motor profile — a command inside URDF but past the motor
        # limit makes the Dynamixel reject it and log "Position command limit
        # exceeded", which used to crash the whole controller.
        arm = getattr(self.robot, arm_name, None)
        if arm is not None:
            limits = getattr(arm, "joint_pos_limit", None)
            if limits is not None and limits.shape == (self.ARM_DOF, 2):
                desired = np.clip(desired, limits[:, 0], limits[:, 1])
        return desired

    def _publish_telemetry(self, command: Dict):
        """Publish telemetry data for visualization.

        Args:
            command: Command dictionary with components.
        """
        try:
            telemetry_data = {
                "timestamp_ns": time.time_ns(),
                "timestamp": time.time(),  # Keep both for compatibility
                "components": {},
                "robot_state": {},
            }

            # Get command data for each component (after filtering - filtered command)
            components = command.get("components", {})
            for comp_name, comp_data in components.items():
                # Convert numpy arrays to lists for proper serialization
                component_copy = {}
                for key, value in comp_data.items():
                    if isinstance(value, np.ndarray):
                        component_copy[key] = value.tolist()
                    else:
                        component_copy[key] = value
                telemetry_data["components"][comp_name] = component_copy

            # Get current robot state (actual positions and velocities from hardware)
            if self.robot:
                robot_state_pos = self._get_robot_joint_pos()
                robot_state_vel = self._get_robot_joint_vel()

                # Combine positions and velocities into robot_state
                telemetry_data["robot_state"] = {
                    "positions": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in robot_state_pos.items()
                    },
                    "velocities": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in robot_state_vel.items()
                    },
                }

            # Get raw command (before filtering) from latest_command
            if self.latest_command:
                raw_components = self.latest_command.get("components", {})
                raw_command = {}
                for comp_name, comp_data in raw_components.items():
                    if isinstance(comp_data, dict):
                        component_copy = {}
                        for key, value in comp_data.items():
                            if isinstance(value, np.ndarray):
                                component_copy[key] = value.tolist()
                            elif isinstance(value, list):
                                component_copy[key] = value
                            else:
                                component_copy[key] = value
                        raw_command[comp_name] = component_copy
                    else:
                        raw_command[comp_name] = comp_data
                telemetry_data["raw_command"] = raw_command

            # Publish telemetry
            self.telemetry_publisher.publish(telemetry_data)

        except Exception as e:
            logger.error(f"Failed to publish telemetry: {e}", exc_info=True)

    def _sync_robot_arms_with_leader(self, command: Dict):
        """Sync robot arms with leader arms."""
        components = command.get("components", {})
        arm_commands = {}
        if "left_arm" in components:
            arm_commands["left_arm"] = components["left_arm"]["pos"]
        if "right_arm" in components:
            arm_commands["right_arm"] = components["right_arm"]["pos"]
        if arm_commands:
            self.robot.move_to_joint_pos(arm_commands)
            self._arm_tracking_watchdog.reset()

    def _send_base_command(self, base_data: Dict):
        """Send command to mobile base.

        Args:
            base_data: Dictionary with vx, vy, wz velocities.
        """
        self.robot.chassis.set_velocity(
            vx=base_data["vx"],
            vy=base_data["vy"],
            wz=base_data["wz"],
        )

    def _send_torso_command(self, torso_data: Dict):
        """Send command to torso joints.

        Args:
            torso_data: Dictionary with position and velocity.
        """
        # logger.info(f'Debugging interploration_method: {self.interpolation_method} and torso_data: {torso_data}')
        self.robot.torso.move_joint_pos(
            torso_data["pos"],
        )

    def _send_head_command(self, head_data: Dict):
        """Send command to head joints.

        Args:
            head_data: Dictionary with position and optional velocity.
        """
        head = getattr(self.robot, "head", None)
        if head is not None:
            if "pos" in head_data:
                positions = head_data["pos"]
                head.set_joint_pos(positions)

    def _send_arm_command(self, arm_name: str, arm_data: Dict):
        """Send command to arm.

        Args:
            arm_name: 'left_arm' or 'right_arm'.
            arm_data: Dictionary with position and optional velocity.
        """
        arm = getattr(self.robot, arm_name, None)
        if arm is not None:
            if "pos" in arm_data:
                positions = arm_data["pos"]
                arm.move_joint_pos(positions, velocity_scale=1.0)

    def _send_hand_command(self, hand_name: str, hand_data: Dict):
        """Send command to hand.

        Args:
            hand_name: 'left_hand' or 'right_hand'
            hand_data: Dictionary with position.
        """
        hand = getattr(self.robot, hand_name, None)
        if hand is not None:
            if "pos" in hand_data:
                positions = hand_data["pos"]
                hand.set_joint_pos(positions)

    def _publish_joint_feedback(self):
        """Publish robot joint positions and velocities at specified rate."""
        rate_limiter = RateLimiter(self.feedback_rate)
        logger.info(f"Starting joint feedback publishing at {self.feedback_rate}Hz")

        while self.joint_publish_running:
            # Get current joint positions and velocities
            joint_positions = self._get_robot_joint_pos()
            joint_velocities = self._get_robot_joint_vel()

            # Create message with timestamp, positions and velocities
            feedback_msg = {
                "timestamp_ns": time.time_ns(),
                "joints": joint_positions,
                "velocities": joint_velocities,
            }

            # Publish via Zenoh
            self.joint_publisher.publish(feedback_msg)

            # Maintain rate
            rate_limiter.sleep()

    def run(self) -> None:
        """Main control loop.

        Runs at the configured control rate, processes commands, sends to robot,
        and publishes telemetry. Exits when exit signal is received.
        """
        logger.info(f"Starting robot control at {self.control_rate}Hz")
        logger.info(
            "Real-time arm control: direct targets, no application lead/speed "
            f"clamp; tracking stop threshold={self.REALTIME_TRACKING_TOL_RAD:.2f} "
            f"rad with <{self.TRACKING_MINIMUM_PROGRESS_RAD:.2f} rad progress "
            f"for {self.TRACKING_VIOLATION_DURATION_SECONDS:.2f}s"
        )

        # Start the joint feedback publishing thread
        self.joint_publish_running = True
        self.joint_publish_thread = threading.Thread(
            target=self._publish_joint_feedback,
            daemon=True,
            name="JointFeedbackPublisher",
        )
        self.joint_publish_thread.start()

        # Start the live display if debug mode
        if self._debug_display:
            self._debug_display.start()

        while not self.exit_requested:
            # Compute interpolated command
            command = self._compute_interpolated_command()

            if command:
                # Send command to robot hardware
                if self._is_first_command:
                    self._sync_robot_arms_with_leader(command)
                    self._is_first_command = False
                else:
                    self._send_robot_command(command)

                # Efficient debug output using rich
                if self._debug_display:
                    self._debug_display.print_robot_command(
                        command.get("components", {})
                    )

                # Publish telemetry for visualization
                if self.telemetry_publisher and command:
                    self._publish_telemetry(command)

            # Use RateLimiter for precise timing
            self.rate_limiter.sleep()

        if self.exit_requested:
            logger.critical("Robot controller loop exited due to exit request")

    def cleanup(self) -> None:
        """Clean up resources.

        Latches the measured pose in position mode, stops joint feedback and
        the debug display, then shuts down robot hardware and communication.

        Idempotent: safe to call from both the normal path and a ``finally``.
        """
        if self._cleanup_done:
            return
        self._cleanup_done = True

        # Stop accepting new trajectories before touching the hardware.
        self.exit_requested = True
        if self.subscriber is not None:
            try:
                self.subscriber.shutdown()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(f"Command subscriber shutdown: {exc}")
            self.subscriber = None

        # Hardware tests showed that set_modes(disable) makes both arms fall.
        # Restore the original position-mode shutdown behavior: briefly latch
        # the measured pose, then disconnect without forcing disable/E-Stop.
        if self.robot:
            try:
                self._establish_current_pose_hold()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"Final position hold before shutdown failed: {exc}")

        # Stop the joint feedback publishing thread
        if self.joint_publish_thread:
            self.joint_publish_running = False
            self.joint_publish_thread.join(timeout=1.0)
            logger.info("Joint feedback publishing stopped")

        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        # Telemetry publisher cleanup handled by Node

        if self.robot:
            logger.info("Re-enabling torso auto-idle mode")
            if self.has_torso:
                self.robot.torso.set_idle_mode(True)
            else:
                logger.warning("Robot has no torso, cannot re-enable auto-idle mode")
            self.robot.shutdown()
            

        # Node handles cleanup
        self.node.shutdown()

        logger.info("Robot controller cleaned up")


def main(
    namespace: str = "",
    interpolation_method: str = "none",
    use_velocity_control: bool = False,
    debug: bool = False,
    publish_telemetry: bool = False,
    config_name: Optional[str] = None,
    no_arm_filter: bool = False,
):
    """Main entry point for robot controller.

    Filter configuration is loaded from the YAML config file.
    Uses ROBOT_CONFIG env var to select config if config_name is not provided.

    Args:
        namespace: Optional namespace prefix.
        interpolation_method: Method for interpolation ('none', 'linear', 'cubic', 'ruckig').
        use_velocity_control: Enable velocity control for smoother motion.
        debug: Enable debug output.
        publish_telemetry: Enable telemetry publishing for visualization.
        config_name: Config file name (without .yaml). Uses ROBOT_CONFIG env var if None.
        no_arm_filter: If True, skip Butterworth filtering for left_arm and right_arm.
    """
    # Setup logging
    logger = setup_logging(debug)
    logger.info(
        f"Starting Dexexo Robot Controller{f' (namespace: {namespace})' if namespace else ''}"
    )

    controller = RobotController(
        namespace=namespace,
        interpolation_method=interpolation_method,
        use_velocity_control=use_velocity_control,
        debug=debug,
        publish_telemetry=publish_telemetry,
        config_name=config_name,
        no_arm_filter=no_arm_filter,
    )

    # Installed before any hardware exists so a stop during Robot() init or
    # homing still unwinds through cleanup() instead of killing the process.
    controller.install_signal_handlers()

    exit_code = 0
    try:
        controller.initialize()
        controller.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted — parking arms before exit")
        exit_code = 130
    except Exception:  # noqa: BLE001 - top-level hardware owner must unwind cleanly
        logger.exception("Robot controller failed")
        exit_code = 1
    finally:
        # The arms are in position mode from Robot() onward, so every exit path
        # (clean, exception, SIGINT, SIGTERM) must reach the park sequence.
        controller.cleanup()

    return exit_code


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
