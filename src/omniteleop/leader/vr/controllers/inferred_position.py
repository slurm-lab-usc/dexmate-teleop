"""Inferred position controller for VR-based robot arm control.

Combines relative wrist deltas (from hand/paddle tracking via InterventionTracker)
with elbow orientation constraints (from VR body tracking) to resolve 7-DOF arm
redundancy during IK solving. The base InferredPositionController is robot-agnostic;
RealRobotInferredPositionController adds dexcontrol integration.

Unlike InterventionController (which uses EE-only IK), this controller feeds
elbow orientations from body tracking as orientation-only IK targets, pinning
the elbow direction and producing more natural arm configurations.
"""

import threading
from typing import Tuple

import numpy as np
from loguru import logger

from omniteleop.leader.vr.solvers.cartesian import (
    ElbowConstrainedIKController,
    RealRobotElbowConstrainedIKController,
)
from omniteleop.leader.vr.trackers.pose import InterventionTracker


# VR body tracking joint names corresponding to each arm's elbow/forearm
_ARM_SIDE_TO_BODY_JOINT = {"left": "left_arm_lower", "right": "right_arm_lower"}


class InferredPositionController:
    """Controller combining VR wrist deltas with body tracking elbow constraints.

    Takes an ElbowConstrainedIKController and InterventionTracker. Computes joint
    positions from VR wrist deltas via IK, with elbow orientation constraints
    from body tracking to resolve 7-DOF redundancy.

    Hand control uses pose interpolation between open and close poses based
    on the A (open) and B (close) button state.

    Attributes:
        tracker (InterventionTracker): VR controller tracker instance.
        ik_controller (ElbowConstrainedIKController): IK controller with elbow support.
    """

    _SIDES = ("left", "right")
    _EE_KEYS = {"left": "L_ee", "right": "R_ee"}
    GRIPPER_RATE = 0.8  # rad/s — incremental gripper speed when A/B held

    def __init__(
        self,
        ik_controller: ElbowConstrainedIKController,
        tracker: InterventionTracker,
        hand_open_poses: dict[str, np.ndarray] | None = None,
        hand_close_poses: dict[str, np.ndarray] | None = None,
    ):
        """Initialize the inferred position controller.

        Args:
            ik_controller: Elbow-constrained IK controller.
            tracker: VR controller tracker for wrist deltas. Its VRSubscriber
                must provide body tracking data for elbow orientations.
            hand_open_poses: Per-side open poses {"left": array, "right": array}.
                If None, hand control is disabled.
            hand_close_poses: Per-side close poses {"left": array, "right": array}.
                If None, hand control is disabled.
        """
        self.ik_controller = ik_controller
        self.tracker = tracker

        # Hand pose interpolation
        self._hand_open: dict[str, np.ndarray | None] = {s: None for s in self._SIDES}
        self._hand_close: dict[str, np.ndarray | None] = {s: None for s in self._SIDES}
        self._hand_enabled = (
            hand_open_poses is not None and hand_close_poses is not None
        )
        if self._hand_enabled:
            for side in self._SIDES:
                self._hand_open[side] = hand_open_poses[side]
                self._hand_close[side] = hand_close_poses[side]

        # Lock for MotionManager access (IK solving may not be thread-safe)
        self._motion_manager_lock = threading.Lock()

        # Per-side state
        self._reference_pose: dict[str, np.ndarray | None] = {
            s: None for s in self._SIDES
        }
        self._prev_tracking: dict[str, bool] = {s: False for s in self._SIDES}

        logger.info("Inferred position controller initialized")
        if self._hand_enabled:
            dof = len(self._hand_open["left"])
            logger.info(f"Hand control enabled ({dof} DOF per hand)")

    def set_joint_state(
        self, left_arm_qpos: np.ndarray, right_arm_qpos: np.ndarray
    ) -> None:
        """Update MotionManager joint state from external data (thread-safe).

        Args:
            left_arm_qpos: Joint positions for the left arm.
            right_arm_qpos: Joint positions for the right arm.
        """
        with self._motion_manager_lock:
            self.ik_controller.set_joint_state(left_arm_qpos, right_arm_qpos)

    def _capture_reference_pose(self, side: str) -> None:
        """Capture the current FK pose as the reference for delta computation.

        Must be called with _motion_manager_lock held.

        Args:
            side: "left" or "right".
        """
        current_qpos = self.ik_controller.motion_manager.get_joint_pos()
        ee_poses = self.ik_controller.motion_manager.fk(
            frame_names=self.ik_controller.motion_manager.target_frames,
            qpos=current_qpos,
            update_robot_state=False,
        )
        ee_key = self._EE_KEYS[side]
        self._reference_pose[side] = ee_poses[ee_key].np.copy()
        logger.info(
            f"{side.capitalize()} arm: Captured reference FK pose for delta computation"
        )

    def _get_elbow_orientations(self) -> dict[str, np.ndarray]:
        """Extract elbow orientations from VR body tracking, transformed to robot frame.

        Returns:
            Dict mapping robot elbow frame names to 3x3 orientation matrices
            in robot frame. E.g. {"L_arm_l4": R_3x3, "R_arm_l4": R_3x3}.
            Empty dict if body tracking is not available.
        """
        vr_sub = self.tracker._vr_sub
        if not vr_sub.has_body_tracking:
            return {}

        body = vr_sub.body
        orientations = {}
        for side in self._SIDES:
            body_joint_name = _ARM_SIDE_TO_BODY_JOINT[side]
            elbow_frame = self.ik_controller._ARM_SIDE_TO_ELBOW_KEY[side]

            # Get raw 4x4 pose in OpenXR space, transform to robot frame
            body_joint = body[body_joint_name]
            orientations[elbow_frame] = body_joint.rotation

        return orientations

    def _hand_pose_from_buttons(
        self, side: str, button_a: bool, button_b: bool
    ) -> list[float] | None:
        """Incremental hand control from button state.

        A = open (move toward open pose), B = close (move toward close pose).
        Steps by GRIPPER_RATE * dt per tick, clamped to [open, close] range.
        If neither pressed, returns None (no change).

        Must be called with _motion_manager_lock held.

        Args:
            side: "left" or "right".
            button_a: A button state (open).
            button_b: B button state (close).

        Returns:
            Hand joint positions, or None if no button pressed.
        """
        if not button_a and not button_b:
            return None

        dt = 1.0 / self.tracker.rate
        step = self.GRIPPER_RATE * dt

        mm = self.ik_controller.motion_manager
        hand = mm.left_hand if side == "left" else mm.right_hand
        current = np.array(hand.get_joint_pos())

        open_pos = self._hand_open[side]
        close_pos = self._hand_close[side]

        # Direction: toward open or toward close
        if button_a:
            direction = np.sign(open_pos - close_pos)
        else:
            direction = np.sign(close_pos - open_pos)

        target = current + direction * step

        # Clamp each joint to [min(open, close), max(open, close)]
        lo = np.minimum(open_pos, close_pos)
        hi = np.maximum(open_pos, close_pos)
        target = np.clip(target, lo, hi)

        return target.tolist()

    def _get_intervention_qpos(
        self,
    ) -> Tuple[
        bool,
        np.ndarray | None,
        np.ndarray | None,
        list[float] | None,
        list[float] | None,
    ]:
        """Get intervention joint positions and hand positions from VR tracking.

        Both arms are solved in a single IK call (with elbow constraints) to
        avoid sequential solving artifacts.

        Returns:
            Tuple containing:
            - tracking (bool): True if either arm is being tracked
            - left_arm_qpos (np.ndarray | None): Joint positions for left arm (7,)
            - right_arm_qpos (np.ndarray | None): Joint positions for right arm (7,)
            - left_hand (list[float] | None): Left hand joint positions, or None
            - right_hand (list[float] | None): Right hand joint positions, or None
        """
        actions = self.tracker.update()
        elbow_orientations = self._get_elbow_orientations()

        arm_qpos: dict[str, np.ndarray | None] = {s: None for s in self._SIDES}
        hand_positions: dict[str, list[float] | None] = {s: None for s in self._SIDES}

        try:
            with self._motion_manager_lock:
                mm = self.ik_controller.motion_manager
                active_deltas: dict[str, tuple[np.ndarray, np.ndarray]] = {}

                for side in self._SIDES:
                    action = actions[side]
                    if action is not None:
                        delta_xyz, delta_R, button_a, button_b = action

                        if not self._prev_tracking[side]:
                            self._capture_reference_pose(side)

                        if self._hand_enabled:
                            hp = self._hand_pose_from_buttons(side, button_a, button_b)
                            if hp is not None:
                                hand = mm.left_hand if side == "left" else mm.right_hand
                                hand.set_joint_pos(hp)
                            hand_positions[side] = hp

                        active_deltas[side] = (delta_xyz, delta_R)
                    elif self._prev_tracking[side]:
                        self._reference_pose[side] = None
                        logger.info(
                            f"{side.capitalize()} arm: Tracking stopped, cleared reference pose"
                        )

                if active_deltas:
                    left_qpos, right_qpos, _ = (
                        self.ik_controller.move_delta_cartesian_with_elbow_both(
                            left_delta=active_deltas.get("left"),
                            right_delta=active_deltas.get("right"),
                            elbow_orientations=elbow_orientations,
                            left_reference=self._reference_pose["left"],
                            right_reference=self._reference_pose["right"],
                        )
                    )
                    arm_qpos["left"] = left_qpos
                    arm_qpos["right"] = right_qpos

        except Exception as e:
            logger.warning(f"Error computing inferred position IK: {e}")
            arm_qpos = {s: None for s in self._SIDES}
            hand_positions = {s: None for s in self._SIDES}

        for side in self._SIDES:
            self._prev_tracking[side] = actions[side] is not None

        is_tracking = any(actions[s] is not None for s in self._SIDES)
        return (
            is_tracking,
            arm_qpos["left"],
            arm_qpos["right"],
            hand_positions.get("left"),
            hand_positions.get("right"),
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        self.tracker.cleanup()
        logger.info("Inferred position controller cleaned up")


class RealRobotInferredPositionController(InferredPositionController):
    """Inferred position controller for real robots via dexcontrol.

    Extends InferredPositionController with automatic robot initialization
    via RealRobotElbowConstrainedIKController.
    """

    def __init__(
        self,
        topic: str = "vr/controllers",
        rate: int = 40,
        visualize: bool = False,
        input_mode: str = "paddle",
    ):
        """Initialize the real robot inferred position controller.

        Args:
            topic: Zenoh topic for VR controller data.
            rate: Update rate for tracker in Hz.
            visualize: Whether to enable visualization.
            input_mode: Input mode — "paddle" for VR controllers,
                "hand" for hand tracking.
        """
        tracker = InterventionTracker(
            topic=topic,
            rate=rate,
            visualize=False,
            input_mode=input_mode,
        )

        ik_controller = RealRobotElbowConstrainedIKController(
            bot=None,
            visualize=visualize,
        )

        super().__init__(ik_controller=ik_controller, tracker=tracker)
