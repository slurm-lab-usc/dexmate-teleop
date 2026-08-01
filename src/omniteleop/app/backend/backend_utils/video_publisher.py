"""Latest-frame WebSocket video fan-out without blocking the event loop."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable

from fastapi import WebSocket, WebSocketDisconnect


class VideoPublisher:
    """Encode each camera once per frame and fan it out to current viewers."""

    def __init__(self) -> None:
        self.active_streams: dict[str, set[WebSocket]] = defaultdict(set)
        self._producer_tasks: dict[str, asyncio.Task] = {}
        self._get_frame_callback: Callable[[str], bytes] | None = None
        self.send_timeout = 0.2

    def set_frame_callback(self, callback: Callable[[str], bytes]) -> None:
        self._get_frame_callback = callback

    @staticmethod
    def _fps(camera_id: str) -> float:
        return 10.0 if camera_id in {"left_wrist", "right_wrist"} else 30.0

    async def start_stream(self, websocket: WebSocket, camera_id: str) -> None:
        if not self._get_frame_callback:
            await websocket.close(code=1011, reason="Frame callback not set")
            return

        await websocket.accept()
        self.active_streams[camera_id].add(websocket)
        task = self._producer_tasks.get(camera_id)
        if task is None or task.done():
            self._producer_tasks[camera_id] = asyncio.create_task(
                self._produce(camera_id)
            )

        try:
            while websocket in self.active_streams.get(camera_id, set()):
                await websocket.receive()
        except (WebSocketDisconnect, RuntimeError, ConnectionError, OSError):
            pass
        finally:
            self._remove(camera_id, websocket)

    async def _produce(self, camera_id: str) -> None:
        interval = 1.0 / self._fps(camera_id)
        try:
            while self.active_streams.get(camera_id):
                started = asyncio.get_running_loop().time()
                await self.publish_frame(camera_id)

                elapsed = asyncio.get_running_loop().time() - started
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            pass
        finally:
            self._producer_tasks.pop(camera_id, None)

    async def publish_frame(self, camera_id: str) -> None:
        """Fetch/encode one frame once, then fan it out to all viewers."""
        callback = self._get_frame_callback
        frame = (
            await asyncio.to_thread(callback, camera_id)
            if callback is not None
            else b""
        )
        if not frame:
            return

        disconnected: set[WebSocket] = set()

        async def send(ws: WebSocket) -> None:
            try:
                await asyncio.wait_for(
                    ws.send_bytes(frame),
                    timeout=self.send_timeout,
                )
            except Exception:
                disconnected.add(ws)

        await asyncio.gather(
            *(
                send(ws)
                for ws in tuple(self.active_streams.get(camera_id, set()))
            ),
            return_exceptions=True,
        )
        for ws in disconnected:
            self._remove(camera_id, ws)

    def _remove(self, camera_id: str, websocket: WebSocket) -> None:
        streams = self.active_streams.get(camera_id)
        if streams is None:
            return
        streams.discard(websocket)
        if not streams:
            self.active_streams.pop(camera_id, None)

    async def stop_stream(self, camera_id: str) -> None:
        streams = tuple(self.active_streams.pop(camera_id, set()))
        for ws in streams:
            try:
                await ws.close()
            except Exception:
                pass
        task = self._producer_tasks.pop(camera_id, None)
        if task is not None:
            task.cancel()

    async def stop_all_streams(self) -> None:
        for camera_id in tuple(self.active_streams):
            await self.stop_stream(camera_id)


video_publisher = VideoPublisher()
