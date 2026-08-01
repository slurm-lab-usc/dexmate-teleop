import numpy as np

from omniteleop.common.tracking_watchdog import DelayedTrackingWatchdog


def test_normal_command_latency_does_not_trip_watchdog() -> None:
    watchdog = DelayedTrackingWatchdog(
        tolerance_rad=0.05,
        reference_delay_s=0.1,
        violation_duration_s=0.2,
    )

    assert not watchdog.update(
        {"right_arm": np.zeros(7)},
        {"right_arm": np.zeros(7)},
        now=0.0,
    )
    assert not watchdog.update(
        {"right_arm": np.full(7, 0.2)},
        {"right_arm": np.zeros(7)},
        now=0.11,
    )
    assert not watchdog.update(
        {"right_arm": np.full(7, 0.4)},
        {"right_arm": np.full(7, 0.2)},
        now=0.22,
    )


def test_low_latency_tracking_of_newest_command_does_not_trip_watchdog() -> None:
    watchdog = DelayedTrackingWatchdog(
        tolerance_rad=0.05,
        reference_delay_s=0.1,
        violation_duration_s=0.2,
    )

    assert not watchdog.update(
        {"right_arm": np.zeros(7)},
        {"right_arm": np.zeros(7)},
        now=0.0,
    )
    assert not watchdog.update(
        {"right_arm": np.full(7, 0.2)},
        {"right_arm": np.full(7, 0.2)},
        now=0.11,
    )
    assert not watchdog.update(
        {"right_arm": np.full(7, 0.4)},
        {"right_arm": np.full(7, 0.4)},
        now=0.22,
    )


def test_sustained_tracking_failure_trips_watchdog() -> None:
    watchdog = DelayedTrackingWatchdog(
        tolerance_rad=0.05,
        reference_delay_s=0.1,
        violation_duration_s=0.2,
    )
    target = {"right_arm": np.full(7, 0.3)}
    stuck = {"right_arm": np.zeros(7)}

    assert not watchdog.update(target, stuck, now=0.0)
    assert not watchdog.update(target, stuck, now=0.11)
    failures = watchdog.update(target, stuck, now=0.32)

    assert failures == {"right_arm": 0.3}
    detail = watchdog.last_failure_details["right_arm"]
    assert detail.joint_index == 0
    assert detail.error_rad == 0.3
    assert detail.actual_rad == 0.0
    assert detail.reference_rad == 0.3


def test_large_dynamic_lag_does_not_trip_while_joint_makes_progress() -> None:
    watchdog = DelayedTrackingWatchdog(
        tolerance_rad=0.2,
        reference_delay_s=0.1,
        violation_duration_s=0.75,
        minimum_progress_rad=0.01,
    )
    target = {"left_arm": np.full(7, 0.5)}

    assert not watchdog.update(target, {"left_arm": np.zeros(7)}, now=0.0)
    assert not watchdog.update(target, {"left_arm": np.zeros(7)}, now=0.11)
    assert not watchdog.update(
        target,
        {"left_arm": np.full(7, 0.02)},
        now=0.90,
    )
    assert not watchdog.update(
        target,
        {"left_arm": np.full(7, 0.04)},
        now=1.70,
    )

    failures = watchdog.update(
        target,
        {"left_arm": np.full(7, 0.04)},
        now=2.46,
    )
    assert failures == {"left_arm": 0.46}
