"""A command-stream E-Stop request must not outlive its producer.

CommandProcessor forces emergency_stop=True on the same command that carries
exit_requested, so the last message published before teleop shuts down latches
_cmd_estop. A joycon R3+L3 exit does not go through the Stop endpoint, so
nothing cleared that latch and the operator could never clear E-Stop again.
"""

from __future__ import annotations

import time

from omniteleop.app.backend.app_backend import TeleopApp


def app_with_cmd_estop(age_s: float | None) -> TeleopApp:
    app = TeleopApp.__new__(TeleopApp)
    app._cmd_estop = True
    app._cmd_pending_realign = False
    app._last_command_at = None if age_s is None else time.time() - age_s
    return app


def test_live_stream_still_blocks_the_clear() -> None:
    """A running producer asserting E-Stop is a real objection."""
    app = app_with_cmd_estop(age_s=0.0)

    assert app._cmd_estop_is_live() is True


def test_latched_request_from_an_exited_producer_does_not_block() -> None:
    app = app_with_cmd_estop(age_s=TeleopApp.CMD_ESTOP_STALE_S + 1.0)

    assert app._cmd_estop_is_live() is False


def test_no_command_ever_received_does_not_block() -> None:
    app = app_with_cmd_estop(age_s=None)

    assert app._cmd_estop_is_live() is False


def test_cleared_flag_is_never_live() -> None:
    app = app_with_cmd_estop(age_s=0.0)
    app._cmd_estop = False

    assert app._cmd_estop_is_live() is False


def test_stale_window_covers_many_publish_cycles() -> None:
    """40 Hz command rate: the window must be a real outage, not a hiccup."""
    assert TeleopApp.CMD_ESTOP_STALE_S >= 1.0
