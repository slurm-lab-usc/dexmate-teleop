#!/usr/bin/env python3
"""Best-effort arm hold, disable transition, and post-disable drift check."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

ARM_DOF = 7
ARM_NAMES = ("left_arm", "right_arm")
DEFAULT_HOLD_SECONDS = 0.6
DEFAULT_SETTLE_TIMEOUT_SECONDS = 6.0
DEFAULT_DWELL_SECONDS = 0.4
DEFAULT_SETTLE_VEL_TOL = 0.05  # rad/s
DEFAULT_DRIFT_TOL = 0.02  # rad, ~1.1 deg


@dataclass
class ArmParkReport:
    """Outcome of parking one arm."""

    arm: str
    attempted: bool = False
    hold_commands: int = 0
    settled: bool = False
    brake_release_was_enabled: bool | None = None
    brake_release_disabled: bool = False
    disabled: bool = False
    start_pos: np.ndarray | None = None
    end_pos: np.ndarray | None = None
    max_drift_rad: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.attempted
            and self.disabled
            and not self.warnings
            and self.max_drift_rad is not None
            and self.max_drift_rad <= DEFAULT_DRIFT_TOL
        )

    def summary(self) -> str:
        drift = "n/a" if self.max_drift_rad is None else f"{self.max_drift_rad:.4f} rad"
        return (
            f"{self.arm}: held={self.hold_commands} cmds, settled={self.settled}, "
            f"disabled={self.disabled}, drift={drift}"
            + (f", warnings={self.warnings}" if self.warnings else "")
        )


def park_arms_safely(
    robot: Any,
    *,
    control_rate: float = 100.0,
    hold_seconds: float = DEFAULT_HOLD_SECONDS,
    settle_timeout_seconds: float = DEFAULT_SETTLE_TIMEOUT_SECONDS,
    dwell_seconds: float = DEFAULT_DWELL_SECONDS,
) -> dict[str, ArmParkReport]:
    """Hold the measured pose, disable both arms in parallel, then verify.

    The function is idempotent and best-effort: failures are returned in the
    reports instead of being raised from the caller's cleanup path. Dexcontrol
    service calls have their own timeouts; the process supervisor supplies the
    final SIGTERM-to-SIGKILL bound.

    ``disable`` only confirms that active control was disabled. It must not be
    interpreted as proof that a mechanical holding brake engaged; the measured
    post-disable drift is the authoritative result.
    """
    reports = {name: ArmParkReport(arm=name) for name in ARM_NAMES}
    targets: dict[str, np.ndarray] = {}
    live: dict[str, Any] = {}

    for name in ARM_NAMES:
        arm = getattr(robot, name, None)
        if arm is None:
            continue
        report = reports[name]
        try:
            target = np.asarray(arm.get_joint_pos(), dtype=float).copy()
            if target.shape != (ARM_DOF,) or not bool(np.isfinite(target).all()):
                raise ValueError(f"expected {ARM_DOF} finite joint positions")
        except Exception as exc:  # noqa: BLE001 - cleanup must continue
            report.warnings.append(f"could not read joint position: {exc}")
            continue
        report.attempted = True
        report.start_pos = target.copy()
        targets[name] = target
        live[name] = arm

    estop_active = _software_estop_active(robot)
    if estop_active:
        for report in reports.values():
            if report.attempted:
                report.warnings.append("hold skipped (software E-Stop active)")
    elif live:
        _hold_until_settled(
            live,
            targets,
            reports,
            control_rate=control_rate,
            hold_seconds=hold_seconds,
            settle_timeout_seconds=settle_timeout_seconds,
        )

    if live:
        _disable_arms_in_parallel(
            live,
            targets,
            reports,
            control_rate=control_rate,
            keep_holding=not estop_active,
        )

    if dwell_seconds > 0 and live:
        time.sleep(dwell_seconds)
    for name, arm in live.items():
        report = reports[name]
        try:
            end_pos = np.asarray(arm.get_joint_pos(), dtype=float)
            if end_pos.shape != (ARM_DOF,) or not bool(np.isfinite(end_pos).all()):
                raise ValueError(f"expected {ARM_DOF} finite joint positions")
            report.end_pos = end_pos
            report.max_drift_rad = float(np.max(np.abs(end_pos - targets[name])))
            if report.max_drift_rad > DEFAULT_DRIFT_TOL:
                report.warnings.append(
                    f"drifted {report.max_drift_rad:.4f} rad after disabling"
                )
        except Exception as exc:  # noqa: BLE001 - cleanup must continue
            report.warnings.append(f"could not verify final position: {exc}")

    for report in reports.values():
        if not report.attempted:
            continue
        log = logger.success if report.ok else logger.warning
        log(f"Arm park {'complete' if report.ok else 'incomplete'} — {report.summary()}")
    return reports


def _hold_until_settled(
    live: dict[str, Any],
    targets: dict[str, np.ndarray],
    reports: dict[str, ArmParkReport],
    *,
    control_rate: float,
    hold_seconds: float,
    settle_timeout_seconds: float,
) -> None:
    period = 1.0 / max(1.0, control_rate)
    started = time.monotonic()
    hold_until = started + max(0.0, hold_seconds)
    deadline = started + max(0.0, settle_timeout_seconds)
    settled = {name: False for name in live}

    while time.monotonic() < deadline:
        _publish_holds(live, targets, reports)
        now = time.monotonic()
        if now >= hold_until:
            settled = {
                name: _is_settled(arm, DEFAULT_SETTLE_VEL_TOL)
                for name, arm in live.items()
            }
            if all(settled.values()):
                break
        time.sleep(min(period, max(0.0, deadline - now)))

    for name, is_settled in settled.items():
        reports[name].settled = is_settled
        if not is_settled:
            reports[name].warnings.append("arm did not settle before timeout")


def _disable_arms_in_parallel(
    live: dict[str, Any],
    targets: dict[str, np.ndarray],
    reports: dict[str, ArmParkReport],
    *,
    control_rate: float,
    keep_holding: bool,
) -> None:
    """Transition both arms concurrently while holding every arm not yet disabled."""
    locks = {name: threading.Lock() for name in live}

    def transition(name: str, arm: Any) -> None:
        report = reports[name]
        try:
            status = arm.get_brake_status()
            report.brake_release_was_enabled = bool(status.get("enabled", False))
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"brake status query failed: {exc}")

        if report.brake_release_was_enabled:
            try:
                response = arm.release_brake(False)
                report.brake_release_disabled = bool(response.get("success", False))
                if not report.brake_release_disabled:
                    report.warnings.append(f"release_brake(False) failed: {response}")
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(f"release_brake(False) failed: {exc}")

        try:
            with locks[name]:
                response = arm.set_modes(["disable"] * ARM_DOF)
                report.disabled = bool(
                    isinstance(response, dict) and response.get("success") is True
                )
                if not report.disabled:
                    report.warnings.append(
                        f"set_modes(disable) failed: {response}"
                    )
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"set_modes(disable) failed: {exc}")

    threads = {
        name: threading.Thread(
            target=transition,
            args=(name, arm),
            daemon=True,
            name=f"park-{name}",
        )
        for name, arm in live.items()
    }
    for thread in threads.values():
        thread.start()

    period = 1.0 / max(1.0, control_rate)
    while any(thread.is_alive() for thread in threads.values()):
        if keep_holding:
            for name, arm in live.items():
                if reports[name].disabled:
                    continue
                with locks[name]:
                    if not reports[name].disabled:
                        _publish_hold(name, arm, targets[name], reports[name])
        for thread in threads.values():
            thread.join(timeout=period / max(1, len(threads)))


def _publish_holds(
    live: dict[str, Any],
    targets: dict[str, np.ndarray],
    reports: dict[str, ArmParkReport],
) -> None:
    for name, arm in live.items():
        _publish_hold(name, arm, targets[name], reports[name])


def _publish_hold(
    name: str,
    arm: Any,
    target: np.ndarray,
    report: ArmParkReport,
) -> None:
    try:
        arm.set_joint_pos(target)
        report.hold_commands += 1
    except Exception as exc:  # noqa: BLE001
        warning = f"hold command failed: {exc}"
        if warning not in report.warnings:
            report.warnings.append(warning)
            logger.error(f"{name} {warning}")


def _is_settled(arm: Any, tolerance: float) -> bool:
    try:
        velocity = np.asarray(arm.get_joint_vel(), dtype=float)
    except Exception:  # noqa: BLE001
        return False
    return (
        velocity.shape == (ARM_DOF,)
        and bool(np.isfinite(velocity).all())
        and float(np.max(np.abs(velocity))) <= tolerance
    )


def _software_estop_active(robot: Any) -> bool:
    estop = getattr(robot, "estop", None)
    if estop is None:
        return False
    try:
        return bool(estop.is_software_estop_enabled())
    except Exception:  # noqa: BLE001
        return False
