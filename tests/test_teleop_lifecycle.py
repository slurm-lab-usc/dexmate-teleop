from __future__ import annotations

import asyncio

import omniteleop.app.backend.app_backend as backend_module
from omniteleop.app.backend.app_backend import TeleopApp


class FakeStateManager:
    def __init__(self) -> None:
        self.estop_calls = 0

    async def estop(self) -> None:
        self.estop_calls += 1


def test_doctor_robot_disables_worker_thread_signal_registration(
    monkeypatch,
) -> None:
    captured = {}
    sentinel = object()

    def fake_robot(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(backend_module, "Robot", fake_robot)

    assert backend_module._create_doctor_robot() is sentinel
    assert captured == {
        "auto_shutdown": False,
        "configure_default_state": False,
    }


def test_stop_latches_estop_until_explicit_clear(monkeypatch) -> None:
    async def run() -> None:
        app = TeleopApp.__new__(TeleopApp)
        app._stop_teleop_procs = lambda: None
        app._gui_estop = False
        app._cmd_estop = True
        app.teleop_lifecycle = "stopping"
        hardware_estop: list[bool] = []

        async def set_software_estop(enabled: bool) -> None:
            hardware_estop.append(enabled)

        app._set_software_estop = set_software_estop
        state = FakeStateManager()
        monkeypatch.setattr(backend_module, "state_manager", state)

        async def run_inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(backend_module.asyncio, "to_thread", run_inline)

        await app._stop_teleop_background()

        assert app.teleop_lifecycle == "stopped"
        assert app._gui_estop is True
        assert app._cmd_estop is False
        assert state.estop_calls == 1
        assert hardware_estop == [True]

    asyncio.run(run())
