"""Per-session zenoh endpoints must not outlive their teleop session.

The backend's Node outlives every session, so subscribers and publishers
declared per start have to be shut down explicitly. They were not, and each
start/stop cycle left another live subscriber on the shared node.
"""

from __future__ import annotations

from omniteleop.app.backend.app_backend import TeleopApp


class FakeEndpoint:
    def __init__(self, raises: bool = False) -> None:
        self.shutdowns = 0
        self._raises = raises

    def shutdown(self) -> None:
        self.shutdowns += 1
        if self._raises:
            raise RuntimeError("endpoint already gone")


def app_with(subs: list, pub=None) -> TeleopApp:
    app = TeleopApp.__new__(TeleopApp)
    app._teleop_subs = list(subs)
    app._recorder_ctrl_pub = pub
    return app


def test_subscribers_are_shut_down_and_dropped() -> None:
    joints, commands = FakeEndpoint(), FakeEndpoint()
    app = app_with([joints, commands])

    app._retire_teleop_endpoints()

    assert joints.shutdowns == 1
    assert commands.shutdowns == 1
    assert app._teleop_subs == []


def test_recorder_publisher_is_released_too() -> None:
    pub = FakeEndpoint()
    app = app_with([], pub)

    app._retire_teleop_endpoints()

    assert pub.shutdowns == 1
    assert app._recorder_ctrl_pub is None


def test_one_bad_endpoint_does_not_strand_the_others() -> None:
    first, second, pub = FakeEndpoint(raises=True), FakeEndpoint(), FakeEndpoint()
    app = app_with([first, second], pub)

    app._retire_teleop_endpoints()

    assert second.shutdowns == 1
    assert pub.shutdowns == 1
    assert app._teleop_subs == []
    assert app._recorder_ctrl_pub is None


def test_retiring_twice_is_a_no_op() -> None:
    """Called from both _stop_teleop_procs and _start_teleop_procs."""
    sub, pub = FakeEndpoint(), FakeEndpoint()
    app = app_with([sub], pub)

    app._retire_teleop_endpoints()
    app._retire_teleop_endpoints()

    assert sub.shutdowns == 1
    assert pub.shutdowns == 1


def test_repeated_sessions_do_not_accumulate_subscribers() -> None:
    """Three start/stop cycles must leave exactly one generation live."""
    app = app_with([])
    live: list[FakeEndpoint] = []

    for _ in range(3):
        app._retire_teleop_endpoints()  # what _start_teleop_procs does first
        generation = [FakeEndpoint(), FakeEndpoint()]
        app._teleop_subs.extend(generation)
        live.append(generation[0])

    assert len(app._teleop_subs) == 2
    # Every earlier generation was shut down exactly once.
    assert [e.shutdowns for e in live] == [1, 1, 0]
