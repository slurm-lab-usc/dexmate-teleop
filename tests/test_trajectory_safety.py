from __future__ import annotations

import numpy as np
import pytest

from omniteleop.common.trajectory_safety import limit_sampled_joint_speed


def test_slow_trajectory_is_not_resampled() -> None:
    trajectory = np.array([[0.0, 0.0], [0.01, -0.02], [0.02, -0.04]])

    scaled, diagnostics = limit_sampled_joint_speed(
        trajectory,
        control_frequency=10.0,
        max_joint_speed_rad_s=0.5,
    )

    np.testing.assert_array_equal(scaled, trajectory)
    assert diagnostics.scale_factor == 1.0
    assert diagnostics.original_waypoints == diagnostics.scaled_waypoints == 3


def test_fast_trajectory_is_time_stretched_to_speed_limit() -> None:
    trajectory = np.array([[0.0, -1.0], [1.0, 1.0]])

    scaled, diagnostics = limit_sampled_joint_speed(
        trajectory,
        control_frequency=10.0,
        max_joint_speed_rad_s=0.5,
    )

    np.testing.assert_array_equal(scaled[0], trajectory[0])
    np.testing.assert_array_equal(scaled[-1], trajectory[-1])
    peak_speed = float(np.max(np.abs(np.diff(scaled, axis=0))) * 10.0)
    assert peak_speed <= 0.5 + 1e-12
    assert diagnostics.original_peak_speed_rad_s == pytest.approx(20.0)
    assert diagnostics.scale_factor == pytest.approx(40.0)
    assert diagnostics.scaled_waypoints == 41


@pytest.mark.parametrize(
    ("trajectory", "frequency", "speed"),
    [
        (np.array([]), 100.0, 0.5),
        (np.array([[0.0, np.nan]]), 100.0, 0.5),
        (np.zeros((2, 2)), 0.0, 0.5),
        (np.zeros((2, 2)), 100.0, 0.0),
    ],
)
def test_invalid_trajectory_scaling_inputs_fail_closed(
    trajectory: np.ndarray,
    frequency: float,
    speed: float,
) -> None:
    with pytest.raises(ValueError):
        limit_sampled_joint_speed(
            trajectory,
            control_frequency=frequency,
            max_joint_speed_rad_s=speed,
        )
