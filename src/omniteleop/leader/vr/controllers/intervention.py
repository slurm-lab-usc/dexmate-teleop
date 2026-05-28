# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Intervention controller for VR-based robot arm control.

Composes a tracker (VR input) with a solver (IK) to produce joint positions
from VR controller deltas. The base InterventionController is robot-agnostic;
RealRobotInterventionController adds dexcontrol integration.
"""

import threading
from typing import Tuple

import numpy as np
from loguru import logger

from omniteleop.leader.vr.solvers.cartesian import BaseIKController, RealRobotIKController
from omniteleop.leader.vr.trackers.pose import InterventionTracker


class InterventionController:
    """Controller for VR-based intervention control of robot arms.

    Takes a BaseIKController and InterventionTracker, computes joint positions
    from VR deltas via IK solving. Hand control uses pose interpolation between
    open and close poses based on the A (open) and B (close) button state.

    Attributes:
        tracker (InterventionTracker): VR controller tracker instance.
        ik_controller (BaseIKController): IK controller for motion planning.
    """

    _SIDES = ("left", "right")
    _EE_KEYS = {"left": "L_ee", "right": "R_ee"}
    GRIPPER_RATE = 0.8  # rad/s — incremental gripper speed when A/B held

    def __init__(
        self,
        ik_controller: BaseIKController,
        tracker: InterventionTracker,
        hand_open_poses: dict[str, np.ndarray] | None = None,
        hand_close_poses: dict[str, np.ndarray] | None = None,
    ):
        """Initialize the intervention controller.

        Args:
            ik_controller: IK controller for computing joint positions.
            tracker: VR controller tracker for getting intervention actions.
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

        logger.info("Intervention controller initialized")
        if self._hand_enabled:
            dof = len(self._hand_open["left"])
            logger.info(f"Hand control enabled ({dof} DOF per hand)")

    def set_joint_state(
        self,
        left_arm_qpos: np.ndarray,
        right_arm_qpos: np.ndarray,
        left_hand_qpos: np.ndarray | None = None,
        right_hand_qpos: np.ndarray | None = None,
        torso_qpos: np.ndarray | None = None,
    ) -> None:
        """Update MotionManager joint state from external data (thread-safe).

        Args:
            left_arm_qpos: Joint positions for the left arm.
            right_arm_qpos: Joint positions for the right arm.
            left_hand_qpos: Joint positions for the left hand (optional).
            right_hand_qpos: Joint positions for the right hand (optional).
            torso_qpos: Joint positions for the torso (optional). Ignored if
                the torso region is locked in MotionManager.
        """
        with self._motion_manager_lock:
            self.ik_controller.set_joint_state(left_arm_qpos, right_arm_qpos)
            if left_hand_qpos is not None or right_hand_qpos is not None:
                self.ik_controller.set_hand_positions(
                    left_hand_qpos.tolist() if left_hand_qpos is not None else None,
                    right_hand_qpos.tolist() if right_hand_qpos is not None else None,
                )
            if torso_qpos is not None:
                self.ik_controller.motion_manager.torso.set_joint_pos(torso_qpos)

    def _capture_reference_pose(self, side: str) -> None:
        """Capture current robot state as the reference for tracking.

        Must be called with _motion_manager_lock held.

        Args:
            side: "left" or "right".
        """
        mm = self.ik_controller.motion_manager
        current_qpos = mm.get_joint_pos()
        ee_poses = mm.fk(
            frame_names=mm.target_frames,
            qpos=current_qpos,
            update_robot_state=False,
        )
        ee_key = self._EE_KEYS[side]
        self._reference_pose[side] = ee_poses[ee_key].np.copy()
        logger.info(
            f"{side.capitalize()} arm: Captured reference FK pose for delta computation"
        )

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
        if button_b:
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

        Both arms are solved in a single IK call to avoid sequential solving
        artifacts (where the second arm's solve sees an already-updated state
        from the first arm's solve).

        Returns:
            Tuple containing:
            - tracking (bool): True if either arm is being tracked
            - left_arm_qpos (np.ndarray | None): Joint positions for left arm (7,)
            - right_arm_qpos (np.ndarray | None): Joint positions for right arm (7,)
            - left_hand (list[float] | None): Left hand joint positions, or None
            - right_hand (list[float] | None): Right hand joint positions, or None

        Torso joints written by the same IK solve (when the TORSO region is
        unlocked) can be read via :meth:`get_torso_qpos` after this call.
        """
        actions = self.tracker.update()

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

                        # Set hand positions BEFORE IK so they're included in
                        # the qpos_dict that _solve_and_apply writes.
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
                        self.ik_controller.move_delta_cartesian_both(
                            left_delta=active_deltas.get("left"),
                            right_delta=active_deltas.get("right"),
                            left_reference=self._reference_pose["left"],
                            right_reference=self._reference_pose["right"],
                        )
                    )
                    arm_qpos["left"] = left_qpos
                    arm_qpos["right"] = right_qpos

        except Exception as e:
            logger.warning(f"Error computing intervention IK: {e}")
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

    def get_torso_qpos(self) -> np.ndarray | None:
        """Return current torso joint positions from MotionManager.

        After an IK solve with the TORSO region unlocked, this reflects the
        positions the solver chose. Returns None when torso joints are
        locked (MM's torso accessor returns an empty array in that case).
        Thread-safe.
        """
        with self._motion_manager_lock:
            torso = self.ik_controller.motion_manager.torso.get_joint_pos()
        if torso is None or len(torso) == 0:
            return None
        return np.asarray(torso)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.tracker.cleanup()
        logger.info("Intervention controller cleaned up")


class RealRobotInterventionController(InterventionController):
    """Intervention controller for real robots via dexcontrol.

    Extends InterventionController with automatic MotionManager-to-robot
    synchronization via a background thread.
    """

    def __init__(
        self,
        topic: str = "vr/controllers",
        rate: int = 40,
        visualize: bool = False,
        hand_open_poses: dict[str, np.ndarray] | None = None,
        hand_close_poses: dict[str, np.ndarray] | None = None,
        mm_kwargs: dict | None = None,
        unlock_torso: bool = False,
        viz_only: bool = False,
    ):
        """Initialize the real robot intervention controller.

        Args:
            topic: Zenoh topic for VR controller data.
            rate: Update rate for tracker in Hz.
            visualize: Whether to enable visualization.
            hand_open_poses: Per-side open poses {"left": array, "right": array}.
                If None, hand control is disabled.
            hand_close_poses: Per-side close poses {"left": array, "right": array}.
                If None, hand control is disabled.
            mm_kwargs: Extra keyword arguments forwarded verbatim to
                MotionManager (e.g. ``custom_urdf_path``).
            unlock_torso: If True, leave the TORSO region unlocked so the IK
                solver can recruit torso joints. HEAD and BASE remain locked.
            viz_only: If True, skip the dexcontrol ``Robot`` instantiation and
                run the IK stack against a virtual MotionManager state seeded
                from the config's initial joint configuration. The SAPIEN
                visualizer reflects IK output; nothing is read from or written
                to real hardware. Implies ``visualize=True``.
        """
        tracker = InterventionTracker(
            topic=topic,
            rate=rate,
            visualize=False,  # Disable tracker's own visualization
        )

        regions_to_lock = ["HEAD", "BASE"] if unlock_torso else ["HEAD", "TORSO", "BASE"]
        if viz_only:
            ik_controller = BaseIKController(
                initial_joint_configuration_dict=None,  # MM uses config defaults
                visualize=True,
                joint_regions_to_lock=regions_to_lock,
                mm_kwargs=mm_kwargs,
            )
        else:
            ik_controller = RealRobotIKController(
                bot=None,  # Will create a new Robot instance
                visualize=visualize,
                joint_regions_to_lock=regions_to_lock,
                mm_kwargs=mm_kwargs,
            )

        super().__init__(
            ik_controller=ik_controller,
            tracker=tracker,
            hand_open_poses=hand_open_poses,
            hand_close_poses=hand_close_poses,
        )
