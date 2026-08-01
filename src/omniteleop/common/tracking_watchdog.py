"""Latency-aware measured joint tracking watchdog."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TrackingFailureDetail:
    """The joint responsible for an arm-level tracking failure."""

    joint_index: int
    error_rad: float
    actual_rad: float
    reference_rad: float


class DelayedTrackingWatchdog:
    """Detect sustained tracking failure without treating normal latency as failure.

    Measured state may match any command from the recent
    ``reference_delay_s`` window. This accepts both low-latency tracking of the
    newest command and normal lag behind it. A mismatch must remain above
    tolerance for ``violation_duration_s`` before it is reported.
    """

    def __init__(
        self,
        tolerance_rad: float,
        reference_delay_s: float = 0.12,
        violation_duration_s: float = 0.20,
        minimum_progress_rad: float | None = None,
    ) -> None:
        self.tolerance_rad = tolerance_rad
        self.reference_delay_s = reference_delay_s
        self.violation_duration_s = violation_duration_s
        self.minimum_progress_rad = minimum_progress_rad
        self._history: dict[str, deque[tuple[float, np.ndarray]]] = defaultdict(deque)
        self._violation_since: dict[str, float] = {}
        self._progress_anchor: dict[str, tuple[int, float, float]] = {}
        self._stream_started_at: dict[str, float] = {}
        self.last_failure_details: dict[str, TrackingFailureDetail] = {}

    def reset(self) -> None:
        self._history.clear()
        self._violation_since.clear()
        self._progress_anchor.clear()
        self._stream_started_at.clear()
        self.last_failure_details.clear()

    def update(
        self,
        targets: Mapping[str, np.ndarray],
        actuals: Mapping[str, np.ndarray],
        *,
        now: float | None = None,
    ) -> dict[str, float]:
        """Record commands and return only sustained tracking violations."""
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - self.reference_delay_s
        failures: dict[str, float] = {}
        self.last_failure_details = {}

        for name, target in targets.items():
            history = self._history[name]
            history.append((timestamp, np.asarray(target, dtype=float).copy()))
            started_at = self._stream_started_at.setdefault(name, timestamp)
            while len(history) > 1 and history[1][0] <= cutoff:
                history.popleft()

            if (
                timestamp - started_at < self.reference_delay_s
                or name not in actuals
            ):
                self._violation_since.pop(name, None)
                continue

            actual = np.asarray(actuals[name], dtype=float)
            reference_errors = [
                (
                    float(np.max(np.abs(actual - reference))),
                    reference,
                )
                for _sample_time, reference in history
            ]
            error, best_reference = min(
                reference_errors,
                key=lambda item: item[0],
            )
            if error <= self.tolerance_rad:
                self._violation_since.pop(name, None)
                self._progress_anchor.pop(name, None)
                continue

            differences = np.abs(actual - best_reference)
            joint_index = int(np.argmax(differences))
            if self.minimum_progress_rad is not None:
                direction = float(
                    np.sign(best_reference[joint_index] - actual[joint_index])
                )
                anchor = self._progress_anchor.get(name)
                if (
                    anchor is None
                    or anchor[0] != joint_index
                    or anchor[2] != direction
                ):
                    self._progress_anchor[name] = (
                        joint_index,
                        float(actual[joint_index]),
                        direction,
                    )
                    self._violation_since[name] = timestamp
                    continue

                progress = direction * (float(actual[joint_index]) - anchor[1])
                if progress >= self.minimum_progress_rad:
                    # The worst joint is still following the command. Start a
                    # fresh observation window instead of mistaking normal
                    # dynamic lag for a stalled actuator.
                    self._progress_anchor[name] = (
                        joint_index,
                        float(actual[joint_index]),
                        direction,
                    )
                    self._violation_since[name] = timestamp
                    continue

            started = self._violation_since.setdefault(name, timestamp)
            if timestamp - started >= self.violation_duration_s:
                failures[name] = error
                self.last_failure_details[name] = TrackingFailureDetail(
                    joint_index=joint_index,
                    error_rad=error,
                    actual_rad=float(actual[joint_index]),
                    reference_rad=float(best_reference[joint_index]),
                )

        active_names = set(targets)
        for name in tuple(self._history):
            if name not in active_names:
                self._history.pop(name, None)
                self._violation_since.pop(name, None)
                self._progress_anchor.pop(name, None)
                self._stream_started_at.pop(name, None)

        return failures
