from __future__ import annotations

import asyncio

import omniteleop.app.backend.backend_utils.video_publisher as video_module
from omniteleop.app.backend.backend_utils.state_publisher import StatePublisher
from omniteleop.app.backend.backend_utils.video_publisher import VideoPublisher


class FakeWebSocket:
    def __init__(self) -> None:
        self.json_messages: list[dict] = []
        self.frames: list[bytes] = []

    async def send_json(self, value: dict) -> None:
        self.json_messages.append(value)

    async def send_bytes(self, value: bytes) -> None:
        self.frames.append(value)


def test_state_tick_builds_once_for_multiple_clients() -> None:
    async def run() -> None:
        publisher = StatePublisher()
        first = FakeWebSocket()
        second = FakeWebSocket()
        publisher.active_connections.update((first, second))
        calls = 0

        async def get_state() -> dict:
            nonlocal calls
            calls += 1
            return {"tick": calls}

        publisher.set_state_callback(get_state)
        await publisher.broadcast_state()

        assert calls == 1
        assert first.json_messages == [{"tick": 1}]
        assert second.json_messages == [{"tick": 1}]

    asyncio.run(run())


def test_video_tick_encodes_once_for_multiple_clients(monkeypatch) -> None:
    async def run() -> None:
        publisher = VideoPublisher()
        first = FakeWebSocket()
        second = FakeWebSocket()
        publisher.active_streams["head"].update((first, second))
        calls = 0

        def get_frame(camera_id: str) -> bytes:
            nonlocal calls
            calls += 1
            assert camera_id == "head"
            return b"jpeg"

        publisher.set_frame_callback(get_frame)
        await publisher.publish_frame("head")

        assert calls == 1
        assert first.frames == [b"jpeg"]
        assert second.frames == [b"jpeg"]

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(video_module.asyncio, "to_thread", run_inline)
    asyncio.run(run())


def test_camera_rates_match_ui_budget() -> None:
    assert VideoPublisher._fps("head") == 30.0
    assert VideoPublisher._fps("left_wrist") == 10.0
    assert VideoPublisher._fps("right_wrist") == 10.0
