"""Safety helpers for trajectories executed on physical robot joints."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrajectoryScaling:
    """Diagnostics for a joint-trajectory time-scaling operation."""

    original_waypoints: int
    scaled_waypoints: int
    original_peak_speed_rad_s: float
    scale_factor: float


def limit_sampled_joint_speed(
    trajectory: np.ndarray,
    *,
    control_frequency: float,
    max_joint_speed_rad_s: float,
) -> tuple[np.ndarray, TrajectoryScaling]:
    """Time-stretch a sampled joint path without changing its geometric path.

    The input and output are both sampled at ``control_frequency``. Extra
    linearly interpolated samples are inserted until every per-joint step is at
    or below ``max_joint_speed_rad_s``. Endpoints are preserved exactly.
    """
    samples = np.asarray(trajectory, dtype=float)
    if samples.ndim != 2:
        raise ValueError("trajectory must be a 2-D array")
    if samples.shape[0] == 0:
        raise ValueError("trajectory must contain at least one waypoint")
    if not bool(np.isfinite(samples).all()):
        raise ValueError("trajectory must contain only finite values")
    if not math.isfinite(control_frequency) or control_frequency <= 0:
        raise ValueError("control_frequency must be positive and finite")
    if not math.isfinite(max_joint_speed_rad_s) or max_joint_speed_rad_s <= 0:
        raise ValueError("max_joint_speed_rad_s must be positive and finite")

    if samples.shape[0] == 1:
        diagnostics = TrajectoryScaling(
            original_waypoints=1,
            scaled_waypoints=1,
            original_peak_speed_rad_s=0.0,
            scale_factor=1.0,
        )
        return samples.copy(), diagnostics

    peak_speed = float(
        np.max(np.abs(np.diff(samples, axis=0))) * control_frequency
    )
    scale_factor = max(1.0, peak_speed / max_joint_speed_rad_s)
    original_intervals = samples.shape[0] - 1
    scaled_intervals = max(
        original_intervals,
        int(math.ceil(original_intervals * scale_factor)),
    )

    if scaled_intervals == original_intervals:
        scaled = samples.copy()
    else:
        old_phase = np.linspace(0.0, 1.0, samples.shape[0])
        new_phase = np.linspace(0.0, 1.0, scaled_intervals + 1)
        scaled = np.column_stack(
            [
                np.interp(new_phase, old_phase, samples[:, joint_index])
                for joint_index in range(samples.shape[1])
            ]
        )
        # Avoid any interpolation roundoff at the safety-critical endpoints.
        scaled[0] = samples[0]
        scaled[-1] = samples[-1]

    diagnostics = TrajectoryScaling(
        original_waypoints=samples.shape[0],
        scaled_waypoints=scaled.shape[0],
        original_peak_speed_rad_s=peak_speed,
        scale_factor=scale_factor,
    )
    return scaled, diagnostics
