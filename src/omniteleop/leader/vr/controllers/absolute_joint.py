"""Absolute joint controller for VR body tracking-based robot arm control.

Composes an ActivationTracker (VR activation gating) with a BodyRetargetingController
(joint retargeting) to produce joint positions from VR body tracking data.
The base AbsoluteJointController is robot-agnostic;
RealRobotAbsoluteJointController adds dexcontrol integration.
"""

import threading
from typing import Tuple

import numpy as np
from loguru import logger

from omniteleop.leader.vr.solvers.cartesian import BaseIKController, RealRobotIKController
from omniteleop.leader.vr.solvers.joint import BodyRetargetingController
from omniteleop.leader.vr.trackers.activation import ActivationTracker


class AbsoluteJointController:
    """Controller for VR body tracking-based absolute joint control.

    Takes a BaseIKController (for MotionManager access and hand positions),
    an ActivationTracker, and a BodyRetargetingController. When activated,
    retargets VR body joints to robot arm joints and interpolates hand closure.

    Body tracking provides both arms simultaneously, so activation on EITHER
    side gates the entire body update. Hand closure is interpolated per-side
    based on each side's hand_t value.

    Attributes:
        tracker (ActivationTracker): VR activation tracker.
        body_controller (BodyRetargetingController): Joint retargeting solver.
        ik_controller (BaseIKController): Provides MotionManager for visualization
            and hand position control.
    """

    _SIDES = ("left", "right")

    def __init__(
        self,
        ik_controller: BaseIKController,
        tracker: ActivationTracker,
        body_controller: BodyRetargetingController,
        hand_open_poses: dict[str, np.ndarray] | None = None,
        hand_close_poses: dict[str, np.ndarray] | None = None,
    ):
        """Initialize the absolute joint controller.

        Args:
            ik_controller: IK controller providing MotionManager access.
            tracker: Activation tracker for VR gating and hand_t.
            body_controller: Body tracking joint retargeting solver.
            hand_open_poses: Per-side open poses {"left": array, "right": array}.
                If None, hand control is disabled.
            hand_close_poses: Per-side close poses {"left": array, "right": array}.
                If None, hand control is disabled.
        """
        self.ik_controller = ik_controller
        self.tracker = tracker
        self.body_controller = body_controller

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

        # Lock for MotionManager access (body retargeting may not be thread-safe)
        self._motion_manager_lock = threading.Lock()

        logger.info("Absolute joint controller initialized")
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

    def _interpolate_hand(self, side: str, hand_t: float) -> list[float]:
        """Interpolate between open and close hand poses.

        Args:
            side: "left" or "right".
            hand_t: Interpolation factor [0=open, 1=close].

        Returns:
            Interpolated hand joint positions.
        """
        positions = (
            self._hand_open[side] * (1.0 - hand_t) + self._hand_close[side] * hand_t
        )
        return positions.tolist()

    def _get_intervention_qpos(
        self,
    ) -> Tuple[
        bool,
        np.ndarray | None,
        np.ndarray | None,
        list[float] | None,
        list[float] | None,
    ]:
        """Get joint positions from body tracking with activation gating.

        Returns:
            Tuple containing:
            - tracking (bool): True if body tracking was applied
            - left_arm_qpos (np.ndarray | None): Joint positions for left arm (7,)
            - right_arm_qpos (np.ndarray | None): Joint positions for right arm (7,)
            - left_hand (list[float] | None): Left hand joint positions, or None
            - right_hand (list[float] | None): Right hand joint positions, or None
        """
        activation = self.tracker.update()
        either_active = activation["left"].is_active or activation["right"].is_active

        if not either_active:
            return (False, None, None, None, None)

        if not self.tracker.vr_sub.has_body_tracking:
            logger.warning("Activation detected but no body tracking data received yet")
            return (False, None, None, None, None)

        # Body tracking → joint retargeting (both arms simultaneously)
        try:
            with self._motion_manager_lock:
                # Set hand positions on MotionManager BEFORE body retargeting so
                # they're already applied when set_joint_pos triggers a render.
                # This way arms and hands update through the same code path.
                left_hand = None
                right_hand = None
                if self._hand_enabled:
                    mm = self.ik_controller.motion_manager
                    if activation["left"].is_active:
                        left_hand = self._interpolate_hand(
                            "left", activation["left"].hand_t
                        )
                        mm.left_hand.set_joint_pos(left_hand)
                    if activation["right"].is_active:
                        right_hand = self._interpolate_hand(
                            "right", activation["right"].hand_t
                        )
                        mm.right_hand.set_joint_pos(right_hand)

                result = self.body_controller.update(self.tracker.vr_sub.body)

                if result is None:
                    return (False, None, None, None, None)

                left_arm_qpos, right_arm_qpos = result
        except Exception as e:
            logger.warning(f"Error computing body tracking retarget: {e}")
            return (False, None, None, None, None)

        return (True, left_arm_qpos, right_arm_qpos, left_hand, right_hand)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.tracker.cleanup()
        logger.info("Absolute joint controller cleaned up")


class RealRobotAbsoluteJointController(AbsoluteJointController):
    """Absolute joint controller for real robots via dexcontrol.

    Extends AbsoluteJointController with automatic robot initialization
    via RealRobotIKController.
    """

    def __init__(
        self,
        topic: str = "vr/controllers",
        input_mode: str = "paddle",
        visualize: bool = False,
    ):
        """Initialize the real robot absolute joint controller.

        Args:
            topic: Zenoh topic for VR controller data.
            input_mode: Input mode — "paddle" or "hand".
            visualize: Whether to enable visualization.
        """
        tracker = ActivationTracker(
            topic=topic,
            input_mode=input_mode,
        )

        ik_controller = RealRobotIKController(
            bot=None,
            visualize=visualize,
        )

        body_controller = BodyRetargetingController(ik_controller.motion_manager)

        super().__init__(
            ik_controller=ik_controller,
            tracker=tracker,
            body_controller=body_controller,
        )
