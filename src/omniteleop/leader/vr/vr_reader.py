#!/usr/bin/env python3
"""VR reader: publishes robot commands only while the operator is actively tracking.

Identical to intervention_deployer but with no fallback policy — when VR tracking
is inactive the loop simply skips publishing, leaving the robot at rest.
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


class VRReader:
    """Runs a VR intervention loop and publishes commands only while tracking.

    When the operator is not holding the trigger the control loop is silent —
    no commands are sent and the robot stays wherever it is.
    """

    def __init__(
        self,
        topic: str = "vr/controllers",
        control_hz: int = 40,
        visualize: bool = False,
        debug: bool = False,
        namespace: str = "",
    ):
        setup_logging(debug)

        self.node = Node(name="vr_reader", namespace=namespace)
        self.config = get_config()

        robot_config = RobotConfig()
        hand_poses = get_hand_poses(robot_config)
        hand_open_poses = hand_poses[0] if hand_poses else None
        hand_close_poses = hand_poses[1] if hand_poses else None

        logger.info("Initializing intervention controller...")
        self.controller = RealRobotInterventionController(
            topic=topic,
            rate=control_hz,
            visualize=visualize,
            hand_open_poses=hand_open_poses,
            hand_close_poses=hand_close_poses,
        )

        self._setup_communication()

        self.latest_robot_joints: Optional[Dict[str, Any]] = None
        self._robot_joints_lock = threading.RLock()
        self._robot_joint_data_initialized = False

        self.control_hz = control_hz

        vr_config = self.config.get("input_handlers", {}).get("vr", {})
        self.sync_interval = vr_config.get("sync_interval", 2.0)

        logger.info("VR reader initialized")
        logger.info(f"Sync interval: {self.sync_interval}s")
        logger.info("Press index trigger on VR controllers to start tracking")

    def _setup_communication(self) -> None:
        robot_joints_topic = self.config.get_topic("robot_joints")
        commands_topic = self.config.get_topic("robot_commands")

        self.robot_joints_sub = self.node.create_subscriber(
            robot_joints_topic,
            self._on_robot_joints_received,
            decoder=DictDataCodec.decode,
        )
        self.command_pub = self.node.create_publisher(
            commands_topic, encoder=DictDataCodec.encode
        )

        logger.info(f"Subscribed to: {self.node.resolve_topic(robot_joints_topic)}")
        logger.info(f"Publishing to: {self.node.resolve_topic(commands_topic)}")

    def _on_robot_joints_received(self, robot_joints_data: Dict[str, Any]) -> None:
        with self._robot_joints_lock:
            self.latest_robot_joints = robot_joints_data
        if self.latest_robot_joints:
            self._robot_joint_data_initialized = True

    def _get_home_pose(self) -> Dict[str, Dict[str, Any]]:
        home_pose = {}
        for component in ["left_arm", "right_arm", "left_hand", "right_hand", "torso", "head"]:
            pos = self.config.get_init_pos(component)
            if pos is not None:
                home_pose[component] = {"pos": pos}
        return home_pose

    def _send_estop_and_home(self, home_pose: Dict[str, Dict[str, Any]]) -> None:
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
        serialized_components = {}
        for comp_name, comp_data in components.items():
            serialized_comp = {}
            for key, value in comp_data.items():
                if isinstance(value, np.ndarray):
                    serialized_comp[key] = value.tolist()
                else:
                    serialized_comp[key] = value
            serialized_components[comp_name] = serialized_comp

        if safety_flags is None:
            safety_flags = {"emergency_stop": False, "exit_requested": False}

        self.command_pub.publish({
            "timestamp_ns": time.time_ns(),
            "components": serialized_components,
            "safety_flags": safety_flags,
        })

    def run(self) -> None:
        logger.info(f"Starting VR reader at {self.control_hz}Hz")

        logger.info("Waiting for robot joint data...")
        while not self._robot_joint_data_initialized:
            time.sleep(0.1)
        logger.info("Robot joint data received!")

        logger.info("Initializing robot to home position...")
        home_pose = self._get_home_pose()
        self._send_estop_and_home(home_pose)
        logger.info("Robot initialized to home position")

        input("Press Enter to start VR reader...")

        rate_limiter = RateLimiter(self.control_hz)
        last_sync_time = time.monotonic()

        try:
            while True:
                tracking, left_arm_qpos, right_arm_qpos, left_gripper, right_gripper = (
                    self.controller._get_intervention_qpos()
                )

                now = time.monotonic()
                if not tracking:
                    # Sync every iteration so MotionManager is fresh when tracking starts.
                    current_left_arm, current_right_arm = self._get_current_arm_positions()
                    if current_left_arm is not None and current_right_arm is not None:
                        self.controller.set_joint_state(current_left_arm, current_right_arm)
                    last_sync_time = now
                elif now - last_sync_time >= self.sync_interval:
                    current_left_arm, current_right_arm = self._get_current_arm_positions()
                    if current_left_arm is not None and current_right_arm is not None:
                        self.controller.set_joint_state(current_left_arm, current_right_arm)
                    last_sync_time = now

                if not tracking:
                    # No active intervention — publish nothing.
                    rate_limiter.sleep()
                    continue

                components = {}
                if left_arm_qpos is not None:
                    components["left_arm"] = {"pos": left_arm_qpos.tolist()}
                if right_arm_qpos is not None:
                    components["right_arm"] = {"pos": right_arm_qpos.tolist()}
                if left_gripper is not None:
                    components["left_hand"] = {"pos": left_gripper}
                if right_gripper is not None:
                    components["right_hand"] = {"pos": right_gripper}

                if components:
                    self._publish_command(components)

                rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("VR reader stopped by user")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        logger.info("Cleaning up...")
        self.controller.cleanup()
        self.node.shutdown()
        logger.info("VR reader cleaned up")


def main(
    topic: str = "vr/controllers",
    control_hz: int = 40,
    visualize: bool = False,
    debug: bool = False,
    namespace: str = "",
) -> None:
    """VR reader: publishes robot commands only while the operator is actively tracking.

    Args:
        topic: Zenoh topic for VR controller data.
        control_hz: Control loop frequency in Hz.
        visualize: Whether to enable visualization.
        debug: Enable debug logging.
        namespace: Namespace for Zenoh topics.
    """
    VRReader(
        topic=topic,
        control_hz=control_hz,
        visualize=visualize,
        debug=debug,
        namespace=namespace,
    ).run()


if __name__ == "__main__":
    tyro.cli(main)
