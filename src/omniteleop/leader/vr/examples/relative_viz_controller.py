#!/usr/bin/env python3
"""Visualization-only relative cartesian controller.

Connects VR controller/hand tracking to IK solving with MotionManager visualization.
No real robot required — the MotionManager visualizer shows the IK result
for each VR input step.

Supports:
- Any robot config (vega_1_gripper, vega_1_f5d6, etc.) via ROBOT_CONFIG env var
- Paddle mode (VR controllers) or hand tracking mode
"""

import os
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from dexcomm import RateLimiter

from omniteleop.leader.vr.solvers.cartesian import BaseIKController
from omniteleop.leader.vr.trackers.pose import InterventionTracker, install_reverse_parity
from omniteleop.leader.vr.controllers import get_hand_poses
from omniteleop.leader.vr.controllers.intervention import InterventionController

from omniteleop.common.config import RobotConfig
from omniteleop.common.logging import setup_logging

# Components with sequential joint naming (component → URDF prefix).
# Hands are excluded — their joint naming varies by robot config
# (L_gripper_j for gripper, L_th_j/L_ff_j/... for f5d6) and are
# controlled via MotionManager's component interface instead.
_SEQUENTIAL_JOINT_COMPONENTS = {
    "left_arm": "L_arm_j",
    "right_arm": "R_arm_j",
    "head": "head_j",
    "torso": "torso_j",
}

# Hardware-specific mode constraints: the end-effector variant baked into a
# robot config dictates which VR input modality is valid. Grippers are paddle-
# only; f5d6 dexterous hands are hand-tracking only.
_CONFIG_SUFFIX_TO_REQUIRED_MODE = {
    "gripper": "paddle",
    "f5d6": "hand",
}


def _required_mode_for_config(config_name: str) -> str | None:
    """Return the required input mode for a robot config, or None if unconstrained."""
    for suffix, mode in _CONFIG_SUFFIX_TO_REQUIRED_MODE.items():
        if config_name.endswith(f"_{suffix}"):
            return mode
    return None


def _config_init_pos_to_waypoint(config: RobotConfig) -> dict:
    """Convert RobotConfig init_pos to named-joint dict for MotionManager.

    Only includes components with sequential joint naming (arms, head, torso).
    Hand init positions are applied separately via set_hand_positions.
    """
    init_pos = config.get("init_pos", {})

    waypoint = {}
    for component, values in init_pos.items():
        if values is None:
            continue
        prefix = _SEQUENTIAL_JOINT_COMPONENTS.get(component)
        if prefix is None:
            continue
        for i, val in enumerate(values):
            waypoint[f"{prefix}{i + 1}"] = val

    return waypoint


def main(
    mode: Literal["paddle", "hand"],
    topic: str = "vr/controllers",
    rate: int = 40,
    debug: bool = False,
    reverse_parity: bool = False,
    urdf_path: str | None = None,
) -> None:
    """Visualization-only relative cartesian controller.

    Tracks VR input, solves IK, and visualizes the result in MotionManager.
    No real robot connection needed.

    Args:
        mode: Input mode - "paddle" for VR controllers, "hand" for hand
            tracking. Required. Must match the robot config's end-effector:
            `*_gripper` configs require "paddle"; `*_f5d6` configs require
            "hand".
        topic: Zenoh topic for VR controller data.
        rate: Control loop frequency in Hz.
        debug: Enable debug logging.
        reverse_parity: If True, install the reverse-parity tracker wrapper
            (swap L/R controllers and conjugate deltas by 180° about Z) so a
            backward-facing robot can be teleop'd as if front-facing. Default
            False.
        urdf_path: Path to a custom URDF forwarded to MotionManager as
            ``custom_urdf_path``. If None, MotionManager uses the URDF baked
            into the robot config.
    """
    setup_logging(debug)

    robot_name = os.environ.get("ROBOT_CONFIG", "vega_1_gripper")
    logger.info(f"Robot config: {robot_name}")

    required_mode = _required_mode_for_config(robot_name)
    if required_mode is not None and required_mode != mode:
        raise ValueError(
            f"Robot config '{robot_name}' requires --mode {required_mode}, "
            f"got --mode {mode}. Grippers are paddle-only; f5d6 is hand-only."
        )

    config = RobotConfig()

    tracker = InterventionTracker(
        topic=topic,
        rate=rate,
        visualize=False,
        input_mode=mode,
    )

    if reverse_parity:
        install_reverse_parity(tracker)

    # Build initial joint configuration from config
    init_waypoint = _config_init_pos_to_waypoint(config)

    mm_kwargs = {"custom_urdf_path": Path(urdf_path)} if urdf_path else None
    ik_controller = BaseIKController(
        initial_joint_configuration_dict=init_waypoint,
        visualize=True,
        joint_regions_to_lock=["HEAD", "BASE", "TORSO"],
        mm_kwargs=mm_kwargs,
    )

    # Set camera pose on the SAPIEN viewer
    import sapien

    viewer = ik_controller.motion_manager.visualizer.viewer
    viewer.set_camera_pose(
        sapien.Pose(
            [-0.268641, -0.0310805, 1.20076],
            [0.980484, 0.00244194, 0.196203, -0.0122443],
        )
    )

    # Set initial hand positions from config (uses component interface, not joint names)
    init_pos = config.get("init_pos", {})
    left_hand_init = init_pos.get("left_hand")
    right_hand_init = init_pos.get("right_hand")
    if left_hand_init is not None or right_hand_init is not None:
        ik_controller.set_hand_positions(left_hand_init, right_hand_init)

    # Load hand open/close poses for interpolation
    hand_poses = get_hand_poses(config)
    hand_open = hand_poses[0] if hand_poses else None
    hand_close = hand_poses[1] if hand_poses else None

    controller = InterventionController(
        ik_controller=ik_controller,
        tracker=tracker,
        hand_open_poses=hand_open,
        hand_close_poses=hand_close,
    )

    rate_limiter = RateLimiter(rate)

    logger.info(f"Starting viz-only relative controller at {rate}Hz ({mode} mode)")
    if mode == "paddle":
        logger.info("Press index trigger on VR controllers to start tracking")
    else:
        logger.info("Pinch thumb+index to start tracking")

    try:
        while True:
            tracking, left_arm_qpos, right_arm_qpos, left_hand, right_hand = (
                controller._get_intervention_qpos()
            )

            if tracking and debug:
                parts = []
                if left_arm_qpos is not None:
                    parts.append(f"L: {left_arm_qpos[:3]}")
                if right_arm_qpos is not None:
                    parts.append(f"R: {right_arm_qpos[:3]}")
                if left_hand is not None:
                    parts.append(f"LH: {left_hand[:2]}")
                if right_hand is not None:
                    parts.append(f"RH: {right_hand[:2]}")
                logger.debug(" | ".join(parts))

            rate_limiter.sleep()

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    tyro.cli(main)
