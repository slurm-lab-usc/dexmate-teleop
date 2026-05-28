#!/usr/bin/env python3
"""Visualization-only absolute joint controller.

Connects VR body tracking to direct joint retargeting with MotionManager
visualization. Paddle or hand input provides activation gating and hand
closure. No real robot required.

Supports:
- Any robot config (vega_1_gripper, vega_1_f5d6, etc.) via ROBOT_CONFIG env var
- Paddle mode (VR controllers) or hand tracking mode for activation + hand closure
- Body tracking for arm joint retargeting
"""

import os
from typing import Literal

import tyro
from loguru import logger

from dexcomm import RateLimiter

from omniteleop.leader.vr.solvers.cartesian import BaseIKController
from omniteleop.leader.vr.solvers.joint import BodyRetargetingController
from omniteleop.leader.vr.trackers.activation import ActivationTracker
from omniteleop.leader.vr.controllers import get_hand_poses
from omniteleop.leader.vr.controllers.absolute_joint import AbsoluteJointController

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
    topic: str = "vr/controllers",
    rate: int = 40,
    robot: str = "",
    mode: Literal["paddle", "hand"] = "paddle",
    debug: bool = False,
) -> None:
    """Visualization-only absolute joint controller.

    Uses VR body tracking for arm joint retargeting and paddle/hand input
    for activation gating and hand closure. No real robot connection needed.

    Args:
        topic: Zenoh topic for VR controller data.
        rate: Control loop frequency in Hz.
        robot: Robot config name (e.g. "vega_1_gripper", "vega_1_f5d6").
            If empty, uses ROBOT_CONFIG env var (default: vega_1_f5d6).
        mode: Input mode - "paddle" for VR controllers, "hand" for hand tracking.
            Controls activation gating and hand closure source.
        debug: Enable debug logging.
    """
    setup_logging(debug)

    # Set robot config from arg or env var
    if robot:
        os.environ["ROBOT_CONFIG"] = robot
    robot_name = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
    logger.info(f"Robot config: {robot_name}")

    config = RobotConfig()

    activation_tracker = ActivationTracker(
        topic=topic,
        input_mode=mode,
    )

    # Build initial joint configuration from config
    init_waypoint = _config_init_pos_to_waypoint(config)

    # Unlock torso for body tracking retargeting
    ik_controller = BaseIKController(
        initial_joint_configuration_dict=init_waypoint,
        visualize=True,
        joint_regions_to_lock=["HEAD", "BASE"],
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

    body_controller = BodyRetargetingController(ik_controller.motion_manager)

    controller = AbsoluteJointController(
        ik_controller=ik_controller,
        tracker=activation_tracker,
        body_controller=body_controller,
        hand_open_poses=hand_open,
        hand_close_poses=hand_close,
    )

    rate_limiter = RateLimiter(rate)

    logger.info(
        f"Starting viz-only absolute joint controller at {rate}Hz ({mode} mode)"
    )
    if mode == "paddle":
        logger.info("Press index trigger on VR controllers to start tracking")
    else:
        logger.info("Pinch thumb+index to start tracking")
    logger.info("Body tracking data required for arm retargeting")

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
