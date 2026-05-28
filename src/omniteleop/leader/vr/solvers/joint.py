"""Absolute position control via direct joint retargeting.

Maps VR body tracking joint transforms to robot joint angles via affine
transformation (scale + offset per joint). No IK solving — joints are set
directly on MotionManager.
"""

import numpy as np
from dexmotion.motion_manager import MotionManager
from loguru import logger
from scipy.spatial.transform import Rotation


class BodyRetargetingController:
    """Absolute position control from VR body tracking to robot joints.

    Extracts relative joint angles from VR body tracking transforms and maps
    them to robot joints via an affine transformation (scale + offset per joint).
    Shares a MotionManager instance with the end-effector relative controller
    for visualization.

    At tracking transitions, a per-joint delta threshold prevents jumps: if any
    joint delta exceeds DELTA_THRESHOLD, the update is rejected entirely. The
    operator naturally aligns their body to the robot's current pose before
    control engages.

    The mapping is defined by BODY_JOINT_MAP: each entry maps a robot joint
    name to (parent_body_joint, child_body_joint, euler_axis, scale, offset).

    All coefficients are placeholder values — tune them for your specific robot.
    """

    DELTA_THRESHOLD = 0.17  # rad — reject update if any joint delta exceeds this

    # Each entry: (parent_vr_joint, child_vr_joint, euler_order, axis_index, scale, offset)
    BODY_JOINT_MAP: dict[str, tuple[str, str, str, int, float, float]] = {
        # Left arm
        "L_arm_j1": ("left_shoulder", "left_arm_upper", "xyz", 0, 1.0, 0.0),
        "L_arm_j2": ("left_shoulder", "left_arm_upper", "xyz", 1, 1.0, 0.0),
        "L_arm_j3": ("left_shoulder", "left_arm_upper", "xyz", 2, 1.0, 0.0),
        "L_arm_j4": ("left_arm_upper", "left_arm_lower", "xyz", 1, 1.0, 0.0),
        "L_arm_j5": ("left_arm_lower", "left_hand_wrist_twist", "xyz", 0, 1.0, 0.0),
        "L_arm_j6": ("left_arm_lower", "left_hand_wrist_twist", "xyz", 1, 1.0, 0.0),
        "L_arm_j7": ("left_arm_lower", "left_hand_wrist_twist", "xyz", 2, 1.0, 0.0),
        # Right arm
        "R_arm_j1": ("right_shoulder", "right_arm_upper", "xyz", 0, 1.0, 0.0),
        "R_arm_j2": ("right_shoulder", "right_arm_upper", "xyz", 1, 1.0, 0.0),
        "R_arm_j3": ("right_shoulder", "right_arm_upper", "xyz", 2, 1.0, 0.0),
        "R_arm_j4": ("right_arm_upper", "right_arm_lower", "xyz", 1, 1.0, 0.0),
        "R_arm_j5": ("right_arm_lower", "right_hand_wrist_twist", "xyz", 0, 1.0, 0.0),
        "R_arm_j6": ("right_arm_lower", "right_hand_wrist_twist", "xyz", 1, 1.0, 0.0),
        "R_arm_j7": ("right_arm_lower", "right_hand_wrist_twist", "xyz", 2, 1.0, 0.0),
    }

    def __init__(self, motion_manager: MotionManager):
        """Initialize the body tracking controller.

        Args:
            motion_manager: Shared MotionManager instance for setting joint
                positions and visualization.
        """
        self.motion_manager = motion_manager
        logger.info("BodyRetargetingController initialized (placeholder mapping)")

    def _extract_relative_angle(
        self,
        body_state,
        parent_name: str,
        child_name: str,
        euler_order: str,
        axis_index: int,
    ) -> float:
        """Extract a single relative joint angle between two VR body joints.

        Args:
            body_state: BodyState from dexstream VRSubscriber (dict-style access).
            parent_name: Name of the parent body joint.
            child_name: Name of the child body joint.
            euler_order: Euler angle convention (e.g., "xyz").
            axis_index: Which euler angle to extract (0, 1, or 2).

        Returns:
            Relative angle in radians.
        """
        parent_rot = body_state[parent_name].rotation
        child_rot = body_state[child_name].rotation
        relative_rot = parent_rot.T @ child_rot
        euler_angles = Rotation.from_matrix(relative_rot).as_euler(
            euler_order, degrees=False
        )
        return float(euler_angles[axis_index])

    def update(self, body_state) -> tuple[np.ndarray, np.ndarray] | None:
        """Retarget VR body tracking to robot joints and apply to MotionManager.

        Computes target joint positions from body tracking, then checks that
        every joint delta is within DELTA_THRESHOLD. If any joint exceeds the
        threshold, the entire update is rejected (robot stays fixed).

        Args:
            body_state: BodyState from dexstream VRSubscriber with body joint transforms.

        Returns:
            Tuple of (left_arm_qpos(7), right_arm_qpos(7)) if applied,
            None if rejected due to delta threshold.
        """
        # Compute target joint positions from body tracking
        target = {}
        for joint_name, (
            parent,
            child,
            euler_order,
            axis_idx,
            scale,
            offset,
        ) in self.BODY_JOINT_MAP.items():
            angle = self._extract_relative_angle(
                body_state, parent, child, euler_order, axis_idx
            )
            target[joint_name] = angle * scale + offset

        # Check per-joint deltas against threshold
        current = self.motion_manager.get_joint_pos_dict()
        max_delta_joint = None
        max_delta = 0.0
        for joint_name, target_val in target.items():
            delta = abs(target_val - current[joint_name])
            if delta > max_delta:
                max_delta = delta
                max_delta_joint = joint_name
            if delta > self.DELTA_THRESHOLD:
                logger.debug(
                    f"Body tracking rejected: {joint_name} delta={delta:.3f} rad "
                    f"(threshold={self.DELTA_THRESHOLD}), "
                    f"worst={max_delta_joint} at {max_delta:.3f} rad"
                )
                return None

        self.motion_manager.set_joint_pos(target)

        left_arm_qpos = np.array([target[f"L_arm_j{i + 1}"] for i in range(7)])
        right_arm_qpos = np.array([target[f"R_arm_j{i + 1}"] for i in range(7)])
        return (left_arm_qpos, right_arm_qpos)
