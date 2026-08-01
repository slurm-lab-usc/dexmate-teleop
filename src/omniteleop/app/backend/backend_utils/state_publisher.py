"""State publisher module - handles state streaming via WebSocket."""

import asyncio
from fastapi import WebSocket, WebSocketDisconnect


class StatePublisher:
    """Handles state streaming."""

    def __init__(self):
        """Initialize state publisher."""
        self.active_connections = set()
        self._broadcast_task = None
        self._get_state_callback = None
        self._on_connection_callback = None
        self._on_disconnection_callback = None
        self.interval = 0.1
        self.send_timeout = 0.25

    def set_state_callback(self, callback):
        """Set callback function to get current state."""
        self._get_state_callback = callback

    def set_connection_callback(self, callback):
        """Set callback function to be called when frontend connects."""
        self._on_connection_callback = callback

    def set_disconnection_callback(self, callback):
        """Set callback function to be called when all frontend connections close."""
        self._on_disconnection_callback = callback

    async def add_connection(self, websocket: WebSocket):
        """Add a WebSocket connection for state streaming."""
        await websocket.accept()
        self.active_connections.add(websocket)
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

        # Notify when all connections are closed
        if len(self.active_connections) == 0 and self._on_disconnection_callback:
            self._on_disconnection_callback()
        if len(self.active_connections) == 0 and self._broadcast_task is not None:
            self._broadcast_task.cancel()
            self._broadcast_task = None

    async def broadcast_state(self):
        """Broadcast state to all connected clients."""
        if not self._get_state_callback:
            return

        state_dict = await self._get_state_callback()
        disconnected = set()

        async def send(ws):
            try:
                await asyncio.wait_for(
                    ws.send_json(state_dict),
                    timeout=self.send_timeout,
                )
            except Exception:
                disconnected.add(ws)

        await asyncio.gather(
            *(send(ws) for ws in tuple(self.active_connections)),
            return_exceptions=True,
        )
        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections.discard(ws)

        # Notify when all connections are closed
        if (
            len(self.active_connections) == 0
            and disconnected
            and self._on_disconnection_callback
        ):
            self._on_disconnection_callback()

    async def _broadcast_loop(self):
        try:
            while self.active_connections:
                started = asyncio.get_running_loop().time()
                await self.broadcast_state()
                elapsed = asyncio.get_running_loop().time() - started
                await asyncio.sleep(max(0.0, self.interval - elapsed))
        except asyncio.CancelledError:
            pass

    async def start_stream(self, websocket: WebSocket):
        """Start streaming state to a WebSocket."""
        await self.add_connection(websocket)

        # Notify that frontend has connected (triggers 1-second delay before real states)
        if self._on_connection_callback:
            asyncio.create_task(self._on_connection_callback())

        try:
            while websocket in self.active_connections:
                await websocket.receive()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"State stream error: {e}")
        finally:
            await self.remove_connection(websocket)


# Global state publisher instance
state_publisher = StatePublisher()
