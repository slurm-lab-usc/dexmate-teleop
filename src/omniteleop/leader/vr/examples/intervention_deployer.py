#!/usr/bin/env python3
"""Example intervention deployer using VR controller tracking.

This script demonstrates how to use RealRobotInterventionController to control
robot arms based on VR controller input. The controller computes joint
positions through IK, and this script publishes them to robot_controller
via dexcomm.
"""

import threading
import time
from typing import Optional, Dict, Any

import numpy as np
import tyro
from dexcomm import Node
from dexcomm.utils import RateLimiter
from dexcomm.codecs import DictDataCodec
from loguru import logger

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging

from omniteleop.leader.vr.controllers import get_hand_poses
from omniteleop.leader.vr.controllers.intervention import RealRobotInterventionController
from omniteleop.common.config import RobotConfig


class FakePolicy:
    """Fake policy that applies small periodic deltas to current arm positions.

    Reads current arm joint positions from MotionManager and adds small
    oscillating deltas to create gentle periodic motion.
    """

    # Joint indices to apply delta to (0-indexed, out of 7 DOF)
    DELTA_JOINT_INDICES = [3, 5]

    def __init__(self, cycle_duration: float = 4.0, delta_amplitude: float = 0.01):
        """Initialize fake policy.

        Args:
            cycle_duration: Duration of one full oscillation cycle in seconds
            delta_amplitude: Maximum amplitude of the periodic delta (radians)
        """
        self.cycle_duration = cycle_duration
        self.delta_amplitude = delta_amplitude
        self.start_time = time.time()

    def reset(self):
        """Reset the policy timer."""
        self.start_time = time.time()

    def get_action(
        self, current_left_arm_qpos: np.ndarray, current_right_arm_qpos: np.ndarray
    ) -> dict:
        """Get the current action based on time.

        Reads current arm positions from MotionManager and adds a small
        periodic delta using a sinusoidal pattern to selected joints.

        Returns:
            Dictionary with "left_arm" and "right_arm" joint positions (7 DOF each)
        """
        elapsed = time.time() - self.start_time

        # Sinusoidal delta factor (-1 to 1)
        t = (elapsed % self.cycle_duration) / self.cycle_duration
        delta = self.delta_amplitude * np.sin(2 * np.pi * t)

        # Get current arm positions from MotionManager (returns numpy arrays)
        left_arm = current_left_arm_qpos.copy()
        right_arm = current_right_arm_qpos.copy()

        # Apply small delta only to selected joints
        for idx in self.DELTA_JOINT_INDICES:
            left_arm[idx] += delta
            right_arm[idx] += delta

        return {
            "left_arm": left_arm,
            "right_arm": right_arm,
        }


class InterventionDeployer:
    """Deployer for VR-based intervention control.

    This class uses RealRobotInterventionController to get joint positions from VR
    controller tracking and publishes them to robot_controller via dexcomm.
    """

    def __init__(
        self,
        topic: str = "vr/controllers",
        control_hz: int = 40,
        visualize: bool = False,
        debug: bool = False,
        namespace: str = "",
    ):
        """Initialize the intervention deployer.

        Args:
            topic: Zenoh topic for VR controller data.
            control_hz: Control loop frequency in Hz.
            visualize: Whether to enable visualization in MotionManager.
            debug: Enable debug logging.
            namespace: Namespace for Zenoh topics.
        """
        # Setup logging
        setup_logging(debug)

        # Initialize Node for communication
        self.node = Node(name="intervention_deployer", namespace=namespace)
        self.config = get_config()

        # Load hand open/close poses from robot config
        robot_config = RobotConfig()
        hand_poses = get_hand_poses(robot_config)
        hand_open_poses = hand_poses[0] if hand_poses else None
        hand_close_poses = hand_poses[1] if hand_poses else None

        # Initialize intervention controller
        logger.info("Initializing intervention controller...")
        self.controller = RealRobotInterventionController(
            topic=topic,
            rate=control_hz,
            visualize=visualize,
            hand_open_poses=hand_open_poses,
            hand_close_poses=hand_close_poses,
        )

        # Initialize fake policy for when tracking is not active
        self.fake_policy = FakePolicy(
            cycle_duration=4.0,
            delta_amplitude=0.04,
        )

        # Setup communication
        self._setup_communication()

        # State tracking for robot joints
        self.latest_robot_joints: Optional[Dict[str, Any]] = None
        self._robot_joints_lock = threading.RLock()
        self._robot_joint_data_initialized = False

        self.control_hz = control_hz

        # Periodic sync interval: correct IK drift during tracking without
        # injecting round-trip latency every iteration.
        vr_config = self.config.get("input_handlers", {}).get("vr", {})
        self.sync_interval = vr_config.get("sync_interval", 2.0)

        logger.info("Intervention deployer initialized")
        logger.info(f"Sync interval: {self.sync_interval}s")
        logger.info("Press index trigger on VR controllers to start tracking")

    def _setup_communication(self) -> None:
        """Setup Zenoh publishers and subscribers."""
        robot_joints_topic = self.config.get_topic("robot_joints")
        commands_topic = self.config.get_topic("robot_commands")

        # Subscribe to robot joints for getting current state
        self.robot_joints_sub = self.node.create_subscriber(
            robot_joints_topic,
            self._on_robot_joints_received,
            decoder=DictDataCodec.decode,
        )

        # Publisher for robot commands
        self.command_pub = self.node.create_publisher(
            commands_topic, encoder=DictDataCodec.encode
        )

        logger.info(f"Subscribed to: {self.node.resolve_topic(robot_joints_topic)}")
        logger.info(f"Publishing to: {self.node.resolve_topic(commands_topic)}")

    def _on_robot_joints_received(self, robot_joints_data: Dict[str, Any]) -> None:
        """Store latest robot joint positions and velocities."""
        try:
            with self._robot_joints_lock:
                self.latest_robot_joints = robot_joints_data
            if self.latest_robot_joints:
                self._robot_joint_data_initialized = True
        except Exception as e:
            logger.error(f"Error storing robot joints feedback: {e}")

    def _get_home_pose(self) -> Dict[str, Dict[str, Any]]:
        """Get home pose from robot config.

        Returns:
            Dictionary mapping component names to their home positions.
            Format: {"left_arm": {"pos": [...]}, "right_arm": {"pos": [...]}, ...}
        """
        home_pose = {}
        for component in [
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "torso",
            "head",
        ]:
            pos = self.config.get_init_pos(component)
            if pos is not None:
                home_pose[component] = {"pos": pos}
        return home_pose

    def _send_estop_and_home(self, home_pose: Dict[str, Dict[str, Any]]) -> None:
        """Send estop activation followed by home pose command.

        Args:
            home_pose: Dictionary mapping component names to their home positions.
        """
        logger.info("Activating emergency stop...")
        self._publish_command(
            home_pose,
            safety_flags={"emergency_stop": True, "exit_requested": False},
        )
        time.sleep(1.0)

        logger.info("Sending home pose command...")
        self._publish_command(
            home_pose,
            safety_flags={"emergency_stop": False, "exit_requested": False},
        )

    def _get_current_arm_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get current arm positions from subscribed robot joints data.

        Returns:
            Tuple of (left_arm_qpos, right_arm_qpos) or (None, None) if not available.
        """
        with self._robot_joints_lock:
            if self.latest_robot_joints is None:
                return None, None

            joints_data = self.latest_robot_joints.get("joints", {})
            left_arm_qpos = joints_data.get("left_arm")
            right_arm_qpos = joints_data.get("right_arm")

            if left_arm_qpos is not None:
                left_arm_qpos = np.array(left_arm_qpos)
            if right_arm_qpos is not None:
                right_arm_qpos = np.array(right_arm_qpos)

            return left_arm_qpos, right_arm_qpos

    def _publish_command(
        self,
        components: Dict[str, Dict[str, Any]],
        safety_flags: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Publish command to robot_controller.

        Args:
            components: Dictionary mapping component names to their data.
                       Format: {"left_arm": {"pos": [...]}, "right_arm": {"pos": [...]}}
                       Positions should be lists, not numpy arrays.
            safety_flags: Optional dictionary with safety flags. If None, uses defaults.
        """
        # Convert numpy arrays to lists for serialization
        serialized_components = {}
        for comp_name, comp_data in components.items():
            serialized_comp = {}
            for key, value in comp_data.items():
                if isinstance(value, np.ndarray):
                    serialized_comp[key] = value.tolist()
                else:
                    serialized_comp[key] = value
            serialized_components[comp_name] = serialized_comp

        # Default safety flags
        if safety_flags is None:
            safety_flags = {"emergency_stop": False, "exit_requested": False}

        command_dict = {
            "timestamp_ns": time.time_ns(),
            "components": serialized_components,
            "safety_flags": safety_flags,
        }
        self.command_pub.publish(command_dict)

    def run(self) -> None:
        """Main control loop.

        Continuously gets intervention joint positions and publishes them
        to robot_controller when tracking is active.
        """
        logger.info(f"Starting intervention deployer at {self.control_hz}Hz")

        # Wait for robot joint data
        logger.info("Waiting for robot joint data...")
        while not self._robot_joint_data_initialized:
            time.sleep(0.1)
        logger.info("Robot joint data received!")

        # Initialize robot: estop -> home pose
        logger.info("Initializing robot to home position...")
        home_pose = self._get_home_pose()
        self._send_estop_and_home(home_pose)
        logger.info("Robot initialized to home position")

        input("Press Enter to start intervention control...")

        rate_limiter = RateLimiter(self.control_hz)
        last_sync_time = time.monotonic()

        try:
            while True:
                # Get intervention joint positions
                tracking, left_arm_qpos, right_arm_qpos, left_gripper, right_gripper = (
                    self.controller._get_intervention_qpos()
                )

                # Sync MotionManager qpos with real robot feedback.
                # The IK target (reference_pose + VR delta) is absolute and
                # independent of MotionManager state — syncing only refreshes
                # the IK seed and non-tracked arm FK target.
                # When NOT tracking: sync every iteration so MotionManager is
                # fresh when tracking starts (reference capture uses FK).
                # When tracking: sync every sync_interval to correct IK seed
                # drift without per-frame round-trip latency.
                now = time.monotonic()
                if not tracking:
                    current_left_arm, current_right_arm = (
                        self._get_current_arm_positions()
                    )
                    if current_left_arm is not None and current_right_arm is not None:
                        self.controller.set_joint_state(
                            current_left_arm, current_right_arm
                        )
                    last_sync_time = now
                elif now - last_sync_time >= self.sync_interval:
                    current_left_arm, current_right_arm = (
                        self._get_current_arm_positions()
                    )
                    if current_left_arm is not None and current_right_arm is not None:
                        self.controller.set_joint_state(
                            current_left_arm, current_right_arm
                        )
                    last_sync_time = now

                # Prepare components dictionary for command
                components = {}

                # Apply joint positions to real robot if tracking is active
                if tracking:
                    if left_arm_qpos is not None:
                        components["left_arm"] = {"pos": left_arm_qpos.tolist()}

                    if right_arm_qpos is not None:
                        components["right_arm"] = {"pos": right_arm_qpos.tolist()}

                    if left_gripper is not None:
                        components["left_hand"] = {"pos": left_gripper}

                    if right_gripper is not None:
                        components["right_hand"] = {"pos": right_gripper}

                    # Publish intervention command
                    if components:
                        self._publish_command(components)
                else:
                    # Apply fake policy when tracking is not active
                    try:
                        # Get current arm positions from subscribed data
                        current_left_arm_qpos, current_right_arm_qpos = (
                            self._get_current_arm_positions()
                        )

                        if (
                            current_left_arm_qpos is not None
                            and current_right_arm_qpos is not None
                        ):
                            # Get action from fake policy
                            action = self.fake_policy.get_action(
                                current_left_arm_qpos,
                                current_right_arm_qpos,
                            )

                            # Publish fake policy command
                            fake_components = {
                                "left_arm": {"pos": action["left_arm"].tolist()},
                                "right_arm": {"pos": action["right_arm"].tolist()},
                            }
                            self._publish_command(fake_components)
                    except Exception as e:
                        logger.warning(f"Error applying fake policy: {e}")

                rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("Intervention deployer stopped by user")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.controller.cleanup()
        self.node.shutdown()
        logger.info("Intervention deployer cleaned up")


def main(
    topic: str = "vr/controllers",
    control_hz: int = 40,
    visualize: bool = False,
    debug: bool = False,
    namespace: str = "",
) -> None:
    """Main entry point for the intervention deployer.

    Args:
        topic: Zenoh topic for VR controller data.
        control_hz: Control loop frequency in Hz.
        visualize: Whether to enable visualization.
        debug: Enable debug logging.
        namespace: Namespace for Zenoh topics.
    """
    deployer = InterventionDeployer(
        topic=topic,
        control_hz=control_hz,
        visualize=visualize,
        debug=debug,
        namespace=namespace,
    )
    deployer.run()


if __name__ == "__main__":
    tyro.cli(main)
