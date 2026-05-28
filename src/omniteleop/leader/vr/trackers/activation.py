"""VR activation tracking for body-based absolute joint control.

Monitors VR controller triggers or hand pinch gestures for activation gating
and hand closure. Does NOT compute relative pose deltas — that's
InterventionTracker's job. This tracker provides:

- Per-side activation state (active/inactive)
- Per-side hand_t closure factor [0=open, 1=close]
- Access to the underlying VRSubscriber for body tracking data

Supports two input modes:
- paddle: VR controllers with trigger-based activation
- hand: Hand tracking with pinch-based activation
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from dexstream.subscribers.vr import VRSubscriber


@dataclass
class ActivationState:
    """Activation state for a single side (left or right)."""

    is_active: bool
    hand_t: float  # Hand closure interpolation factor [0=open, 1=close]


class ActivationTracker:
    """Tracks VR activation state without pose computation.

    Reads VR controller triggers (paddle mode) or hand pinch gestures (hand
    mode) to determine activation gating and hand closure factor. Body tracking
    data is accessed through the public ``vr_sub`` property.

    Uses the same threshold constants as InterventionTracker for consistent
    behavior across control modes.
    """

    TRIGGER_THRESHOLD = 0.5
    PINCH_ACTIVATE_THRESHOLD = 0.04  # metres
    PINCH_OPEN_DISTANCE = 0.10  # metres — fully open (hand_t=0)
    PINCH_CLOSE_DISTANCE = 0.01  # metres — fully closed (hand_t=1)

    def __init__(
        self,
        topic: str = "vr/controllers",
        input_mode: str = "paddle",
    ):
        """Initialize the activation tracker.

        Args:
            topic: Zenoh topic to subscribe to for VR controller data.
            input_mode: Input mode — "paddle" for VR controllers,
                "hand" for hand tracking.
        """
        if input_mode not in ("paddle", "hand"):
            raise ValueError(
                f"Invalid input_mode: {input_mode}. Must be 'paddle' or 'hand'."
            )

        self.input_mode = input_mode
        self._vr_sub = VRSubscriber(topic=topic)
        self._vr_sub.start()

        logger.info(f"Activation tracker initialized in '{input_mode}' mode")

    @property
    def vr_sub(self) -> VRSubscriber:
        """The underlying VRSubscriber for body tracking data access."""
        return self._vr_sub

    def _pinch_distance_to_hand_t(self, pinch_distance: float) -> float:
        """Map pinch distance to hand closure factor [0=open, 1=close]."""
        t = (self.PINCH_OPEN_DISTANCE - pinch_distance) / (
            self.PINCH_OPEN_DISTANCE - self.PINCH_CLOSE_DISTANCE
        )
        return float(np.clip(t, 0.0, 1.0))

    def update(self) -> dict[str, ActivationState]:
        """Read latest VR data and return per-side activation state.

        Returns:
            Dictionary with keys "left" and "right", each an ActivationState.
        """
        if self.input_mode == "paddle":
            return self._update_paddle()
        return self._update_hand()

    def _update_paddle(self) -> dict[str, ActivationState]:
        """Read paddle controllers for activation and hand_t."""
        if not self._vr_sub.connected or self._vr_sub.timestamp_ns == 0:
            return {
                "left": ActivationState(is_active=False, hand_t=0.0),
                "right": ActivationState(is_active=False, hand_t=0.0),
            }

        left = self._vr_sub.left
        right = self._vr_sub.right
        return {
            "left": ActivationState(
                is_active=left.index_trigger > self.TRIGGER_THRESHOLD,
                hand_t=left.index_trigger,
            ),
            "right": ActivationState(
                is_active=right.index_trigger > self.TRIGGER_THRESHOLD,
                hand_t=right.index_trigger,
            ),
        }

    def _update_hand(self) -> dict[str, ActivationState]:
        """Read hand tracking for activation and hand_t."""
        if not self._vr_sub.connected or self._vr_sub.hand_timestamp_ns == 0:
            return {
                "left": ActivationState(is_active=False, hand_t=0.0),
                "right": ActivationState(is_active=False, hand_t=0.0),
            }

        left_hand = self._vr_sub.left_hand
        right_hand = self._vr_sub.right_hand
        return {
            "left": ActivationState(
                is_active=left_hand.fingertips.pinch_distance
                < self.PINCH_ACTIVATE_THRESHOLD,
                hand_t=self._pinch_distance_to_hand_t(
                    left_hand.fingertips.pinch_distance
                ),
            ),
            "right": ActivationState(
                is_active=right_hand.fingertips.pinch_distance
                < self.PINCH_ACTIVATE_THRESHOLD,
                hand_t=self._pinch_distance_to_hand_t(
                    right_hand.fingertips.pinch_distance
                ),
            ),
        }

    def cleanup(self) -> None:
        """Stop the VRSubscriber."""
        self._vr_sub.stop()
        logger.info("Activation tracker stopped")
