"""VR Tracking with Trigger-Based Activation.

Tracks VR controller or hand wrist position and orientation when activated.
Both position and orientation are tracked relative to when tracking started (zeroed).

Supports two input modes:
- paddle: VR controllers with trigger-based activation, A/B buttons for gripper
- hand: Hand tracking with externally-controlled activation via set_intervening()

Uses dexstream's VRSubscriber for typed VR data access. Data arrives in robot
frame (Z-up); only the per-side EE local-frame correction is applied here.
"""

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tyro
from loguru import logger

from dexcomm import RateLimiter
from dexstream.retargeting.transforms import apply_ee_correction
from dexstream.subscribers.vr import VRSubscriber, ControllerState, HandState
from omniteleop.common.logging import setup_logging


@dataclass
class TrackedPose:
    """Tracked pose data for a single controller or hand."""

    position: np.ndarray  # Relative position (3,)
    orientation: np.ndarray  # Relative orientation as 3x3 rotation matrix
    is_tracking: bool  # Whether currently tracking
    button_a: bool = False  # A button state (gripper open)
    button_b: bool = False  # B button state (gripper close)


class InterventionTracker:
    """Tracks VR controller or hand poses with activation gating.

    Paddle mode: index trigger activates tracking, A/B buttons control gripper.
    Hand mode: activation is set externally via set_intervening(); wrist tracking
    runs when intervening is True.
    """

    TRIGGER_THRESHOLD = 0.5  # Threshold for considering trigger pressed
    DEFAULT_RATE = 40

    def __init__(
        self,
        topic: str = "vr/controllers",
        rate: Optional[int] = None,
        visualize: bool = True,
        input_mode: str = "paddle",
    ):
        """Initialize the intervention tracker.

        Args:
            topic: Zenoh topic to subscribe to for VR controller data
            rate: Update rate in Hz
            visualize: Whether to visualize with rerun
            input_mode: Input mode - "paddle" for VR controllers, "hand" for hand tracking
        """
        if input_mode not in ("paddle", "hand"):
            raise ValueError(
                f"Invalid input_mode: {input_mode}. Must be 'paddle' or 'hand'."
            )

        self.rate = rate or self.DEFAULT_RATE
        self.visualize = visualize
        self.input_mode = input_mode

        # Use VRSubscriber for typed access to controller data
        self._vr_sub = VRSubscriber(topic=topic)
        self._vr_sub.start()

        self.rate_limiter = RateLimiter(self.rate)

        # Per-side intervention state (paddle: auto from trigger, hand: set externally)
        self._intervening: dict[str, bool] = {"left": False, "right": False}

        # Per-side tracking state: origin pose captured when tracking starts
        self._tracking: dict[str, bool] = {"left": False, "right": False}
        self._origin_pos: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._origin_rot: dict[str, np.ndarray | None] = {"left": None, "right": None}

        # Last tracked poses
        self.left_pose = TrackedPose(
            position=np.zeros(3),
            orientation=np.eye(3),
            is_tracking=False,
        )
        self.right_pose = TrackedPose(
            position=np.zeros(3),
            orientation=np.eye(3),
            is_tracking=False,
        )

        logger.info(f"Intervention tracker initialized in '{input_mode}' mode")

        # Rerun visualization
        self._rr = None
        if self.visualize:
            self._setup_rerun()

    @property
    def is_intervening(self) -> dict[str, bool]:
        """Per-side intervention state. Read-only view."""
        return dict(self._intervening)

    def set_intervening(self, side: str, active: bool) -> None:
        """Set intervention state externally (for hand mode).

        Args:
            side: "left" or "right".
            active: Whether intervention is active.
        """
        self._intervening[side] = active

    def _setup_rerun(self):
        """Setup rerun visualization."""
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            self._rr = rr
            self._rrb = rrb

            blueprint = rrb.Blueprint(
                rrb.Spatial3DView(
                    name="Controller Tracking",
                    origin="/",
                ),
            )

            rr.init("intervention_sim", spawn=True)
            rr.send_blueprint(blueprint)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

            logger.info("Rerun visualization initialized")
        except ImportError:
            logger.warning("Rerun not installed. Visualization disabled.")
            logger.warning("Install with: pip install rerun-sdk")
            self.visualize = False

    def _process_single_controller(
        self,
        side: str,
        controller: ControllerState,
    ) -> TrackedPose:
        """Process one paddle controller.

        Activation: index trigger > threshold.
        Buttons A/B passed through for gripper control.
        """
        wrist_robot = apply_ee_correction(controller.wrist, side)
        triggered = controller.index_trigger > self.TRIGGER_THRESHOLD
        self._intervening[side] = triggered

        if triggered:
            if not self._tracking[side]:
                self._origin_pos[side] = wrist_robot[:3, 3].copy()
                self._origin_rot[side] = wrist_robot[:3, :3].copy()
                self._tracking[side] = True
                logger.info(f"{side.capitalize()} controller tracking started")

            relative_pos = wrist_robot[:3, 3] - self._origin_pos[side]
            relative_rot = wrist_robot[:3, :3] @ self._origin_rot[side].T

            return TrackedPose(
                position=relative_pos,
                orientation=relative_rot,
                is_tracking=True,
                button_a=controller.button_a,
                button_b=controller.button_b,
            )

        if self._tracking[side]:
            logger.info(f"{side.capitalize()} controller tracking stopped")
        self._tracking[side] = False
        self._origin_pos[side] = None
        self._origin_rot[side] = None
        return TrackedPose(
            position=np.zeros(3),
            orientation=np.eye(3),
            is_tracking=False,
        )

    def _process_hand_tracking(self, side: str, hand: HandState) -> TrackedPose:
        """Process one tracked hand when externally activated.

        Activation is controlled via set_intervening(), not pinch.
        Wrist tracking runs only when _intervening[side] is True.
        """
        wrist_robot = apply_ee_correction(hand.wrist, side)

        if self._intervening[side]:
            if not self._tracking[side]:
                self._origin_pos[side] = wrist_robot[:3, 3].copy()
                self._origin_rot[side] = wrist_robot[:3, :3].copy()
                self._tracking[side] = True
                logger.info(f"{side.capitalize()} hand tracking started")

            relative_pos = wrist_robot[:3, 3] - self._origin_pos[side]
            relative_rot = wrist_robot[:3, :3] @ self._origin_rot[side].T

            return TrackedPose(
                position=relative_pos,
                orientation=relative_rot,
                is_tracking=True,
            )

        if self._tracking[side]:
            logger.info(f"{side.capitalize()} hand tracking stopped")
        self._tracking[side] = False
        self._origin_pos[side] = None
        self._origin_rot[side] = None
        return TrackedPose(
            position=np.zeros(3),
            orientation=np.eye(3),
            is_tracking=False,
        )

    def _process_controller_data(self) -> tuple[TrackedPose, TrackedPose]:
        """Process both controllers/hands and update tracking state.

        Returns:
            Tuple of (left_pose, right_pose).
        """
        if self.input_mode == "paddle":
            if not self._vr_sub.connected or self._vr_sub.timestamp_ns == 0:
                return self.left_pose, self.right_pose
            self.left_pose = self._process_single_controller("left", self._vr_sub.left)
            self.right_pose = self._process_single_controller(
                "right", self._vr_sub.right
            )
        else:
            if not self._vr_sub.connected or self._vr_sub.hand_timestamp_ns == 0:
                return self.left_pose, self.right_pose
            self.left_pose = self._process_hand_tracking("left", self._vr_sub.left_hand)
            self.right_pose = self._process_hand_tracking(
                "right", self._vr_sub.right_hand
            )

        return self.left_pose, self.right_pose

    def _action_from_pose(
        self, pose: TrackedPose
    ) -> tuple[np.ndarray, np.ndarray, bool, bool] | None:
        """Convert a TrackedPose to (delta_xyz, delta_R, button_a, button_b).

        `delta_R` is the 3x3 relative rotation matrix carried through SO(3);
        it is not decomposed into Euler angles (gimbal-lock free).

        Returns None if the pose is not tracking.
        """
        if not pose.is_tracking:
            return None
        delta_xyz = pose.position.copy()
        delta_R = pose.orientation.copy()
        return (delta_xyz, delta_R, pose.button_a, pose.button_b)

    def _get_intervention_actions(
        self,
    ) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray, bool, bool]]]:
        """Get intervention actions (delta_xyz, delta_R, buttons) for each side.

        Returns:
            Dictionary with keys "left" and "right", each containing:
            - None if not tracking
            - Tuple of (delta_xyz, delta_R, button_a, button_b) if tracking
        """
        return {
            "left": self._action_from_pose(self.left_pose),
            "right": self._action_from_pose(self.right_pose),
        }

    def update(self) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray, bool, bool]]]:
        """Update tracking state and return intervention actions.

        This method should be called periodically. It always reads the LATEST
        data from VRSubscriber (which is continuously updated by its background
        callback thread), so intermediate updates are effectively skipped.

        Returns:
            Dictionary with keys "left" and "right", each containing:
            - None if not tracking
            - Tuple of (delta_xyz, delta_R, button_a, button_b) if tracking
        """
        left_pose, right_pose = self._process_controller_data()

        if self.visualize:
            self._visualize_poses(left_pose, right_pose)

        return self._get_intervention_actions()

    def _visualize_poses(self, left_pose: TrackedPose, right_pose: TrackedPose):
        """Visualize poses using rerun."""
        if not self.visualize or self._rr is None:
            return

        rr = self._rr
        axis_length = 0.1

        for name, pose, point_color in [
            ("left_controller", left_pose, [0, 150, 255]),
            ("right_controller", right_pose, [255, 150, 0]),
        ]:
            if pose.is_tracking:
                pos = pose.position
                rot = pose.orientation
                x_axis = rot @ np.array([axis_length, 0, 0])
                y_axis = rot @ np.array([0, axis_length, 0])
                z_axis = rot @ np.array([0, 0, axis_length])

                rr.log(
                    f"{name}/axes",
                    rr.Arrows3D(
                        origins=[pos, pos, pos],
                        vectors=[x_axis, y_axis, z_axis],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ),
                )
                rr.log(
                    f"{name}/position",
                    rr.Points3D([pos], colors=[point_color], radii=[0.02]),
                )
            else:
                rr.log(name, rr.Clear(recursive=True))

        rr.log(
            "world/origin",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                colors=[[200, 50, 50], [50, 200, 50], [50, 50, 200]],
            ),
        )

    def _print_status(self, left_pose: TrackedPose, right_pose: TrackedPose):
        """Print tracking status to console."""
        status_parts = []

        for label, pose in [("L", left_pose), ("R", right_pose)]:
            if pose.is_tracking:
                pos = pose.position
                status_parts.append(
                    f"{label}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
                )

        if status_parts:
            print(f"\r{' | '.join(status_parts):<80}", end="", flush=True)
        else:
            print(f"\r{'Waiting for trigger press...':<80}", end="", flush=True)

    def run(self):
        """Main tracking loop (standalone mode)."""
        logger.info(f"Starting intervention tracker at {self.rate}Hz")
        logger.info("Press index trigger on left or right controller to start tracking")
        logger.info("Position will be relative to where tracking started")

        if self.visualize:
            logger.info("Rerun visualization enabled - check rerun viewer")

        try:
            while True:
                self.update()
                self._print_status(self.left_pose, self.right_pose)
                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            print()  # New line after status
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self._vr_sub.stop()
        logger.info("Intervention tracker stopped")


def install_reverse_parity(tracker: InterventionTracker) -> None:
    """Wrap tracker.update() to swap L/R and conjugate deltas by 180° about Z.

    Enables teleop of a backward-facing robot as if front-facing (head cam also
    rotated 180°). Applied at the tracker boundary so downstream IK sees
    already-flipped deltas and stays parity-agnostic.

    Also swaps tracker._intervening in place each tick so consumers that read
    per-side gating flags see a view consistent with the swapped actions.
    _process_controller_data rewrites _intervening fresh every tick, so the
    in-place swap is self-consistent and does not leak across ticks.

    The tracker emits `(delta_xyz, delta_R, btn_a, btn_b)` with `delta_R` a
    3x3 rotation matrix. We conjugate the matrix directly — no Euler round
    trip. Feeding the matrix into `Rotation.from_euler` would broadcast its
    rows as three separate angle triples and silently produce garbage.

    Math:
      - flipped_xyz = R180z @ delta_xyz
      - flipped_R = R180z @ delta_R @ R180z.T
      - R180z = diag(-1, -1, 1), det=+1 (proper rotation), R180z @ R180z = I

    Idempotency: NOT idempotent. Calling twice wraps twice, which cancels the
    math back to identity (this is exercised in tests/test_reverse_parity.py).
    """
    _R180z = np.diag([-1.0, -1.0, 1.0])
    _original_update = tracker.update

    def _parity_update():
        actions = _original_update()
        raw = tracker._intervening
        raw["left"], raw["right"] = raw["right"], raw["left"]
        swapped = {"left": actions["right"], "right": actions["left"]}
        for side in ("left", "right"):
            if swapped[side] is not None:
                delta_xyz, delta_R, btn_a, btn_b = swapped[side]
                flipped_xyz = _R180z @ delta_xyz
                flipped_R = _R180z @ delta_R @ _R180z.T
                swapped[side] = (flipped_xyz, flipped_R, btn_a, btn_b)
        return swapped

    tracker.update = _parity_update
    logger.info("Reverse parity enabled — L/R swapped and deltas mirrored")


def main(
    topic: str = "vr/controllers",
    rate: Optional[int] = None,
    visualize: bool = True,
    debug: bool = False,
) -> int:
    """Intervention simulation - VR controller tracking with trigger activation.

    Args:
        topic: Zenoh topic to subscribe to for VR controller data
        rate: Update rate in Hz (default: 40)
        visualize: Enable rerun visualization for coordinate frame tracking
        debug: Enable debug logging

    Returns:
        Exit code (0 for success, 1 for error)
    """
    setup_logging(debug)
    logger.info("Starting Intervention Simulation")

    try:
        tracker = InterventionTracker(
            topic=topic,
            rate=rate,
            visualize=visualize,
        )
        tracker.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
