#!/usr/bin/env python3
"""Replay script for recorded episodes (MDP pkl or MCAP format).

Loads an episode via :mod:`omniteleop.record.episode_loader` and publishes
its recorded actions on ``robot_commands`` at the original recording rate,
following the same initialization pattern as command_processor.py. Alongside
each command it publishes the episode's recorded camera frames (JPEG bytes,
``replay/video/<camera>``) and a progress dict (``replay/status``) so the
app backend can mirror the replay in the web UI.

The start gate reads one line from stdin ("Press Enter"): interactively
that's the safety prompt; when spawned by the app backend, stdin is a pipe
and the backend writes a newline when the user clicks "Begin Motion".
"""

import select
import sys
import time
from typing import Any, Dict, Optional

import tyro
from loguru import logger

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.record.episode_loader import ReplayFrame, load_episode


class ReplayRecorder:
    """Replays a recorded episode by re-publishing its commands.

    Handles both MDP (transitions.pkl) and MCAP (episode.mcap) episodes via
    the episode_loader abstraction. Follows the same initialization pattern
    as the command processor to ensure proper robot controller state
    management.
    """

    # Constants for timing and initialization
    ESTOP_ACTIVATION_DELAY_SEC: float = 1.0
    INITIAL_POSITION_DELAY_SEC: float = 6.0
    DEBUG_LOG_INTERVAL: int = 20

    def __init__(
        self,
        episode_path: str,
        namespace: str = "",
        debug: bool = False,
    ) -> None:
        """Initialize replay recorder.

        Args:
            episode_path: Path to the episode directory (or a legacy
                transitions.pkl path).
            namespace: Namespace for Zenoh topics.
            debug: Enable debug output.

        Raises:
            FileNotFoundError: If the path is not a recorded episode.
        """
        self.node = Node(name="replay_recorder", namespace=namespace)

        self.debug = debug
        self.config = get_config()

        logger.info(f"📂 Loading episode: {episode_path}")
        self.loader = load_episode(episode_path)
        logger.info(
            f"✅ Episode loaded: format={self.loader.format}, "
            f"{self.loader.num_frames} frames @ {self.loader.rate_hz}Hz, "
            f"cameras={self.loader.cameras}"
        )

        # Publish at the episode's recorded rate so replayed motion matches
        # the original timing (falls back to 20Hz inside the loader).
        self.publish_rate = self.loader.rate_hz
        self.rate_limiter = RateLimiter(self.publish_rate)

        # State tracking
        self.running = False
        self._frame_idx = 0

        # Setup communication channels
        self._setup_communication()

        logger.info(
            f"🎬 ReplayRecorder initialized, will publish at {self.publish_rate}Hz"
        )

    def _setup_communication(self) -> None:
        """Setup Zenoh publishers.

        Creates the robot_commands publisher (same topic structure as
        command_processor), a replay status publisher for the app backend,
        and one raw-bytes JPEG publisher per recorded camera.
        """
        # Publisher for commands
        commands_topic = self.config.get_topic("robot_commands")
        self.command_pub = self.node.create_publisher(
            commands_topic, encoder=DictDataCodec.encode
        )

        resolved_commands = self.node.resolve_topic(commands_topic)
        logger.info(f"📡 Publishing commands to: {resolved_commands}")

        # Replay progress/status for the app backend & UI.
        status_topic = self.config.get_topic("replay_status", "replay/status")
        self.status_pub = self.node.create_publisher(
            status_topic, encoder=DictDataCodec.encode
        )

        # Recorded camera frames — raw JPEG bytes, one topic per camera.
        self.video_pubs = {
            cam: self.node.create_publisher(f"replay/video/{cam}")
            for cam in self.loader.cameras
        }

    def _publish_status(self, state: str) -> None:
        """Publish a replay status dict (state machine + progress)."""
        try:
            self.status_pub.publish(
                {
                    "state": state,
                    "frame_idx": self._frame_idx,
                    "total": self.loader.num_frames,
                    "episode": str(self.loader.episode_dir),
                    "format": self.loader.format,
                    "cameras": self.loader.cameras,
                    "rate_hz": self.publish_rate,
                }
            )
        except Exception as e:  # noqa: BLE001 — status is best-effort
            logger.warning(f"Failed to publish replay status: {e}")

    def _publish_images(self, frame: ReplayFrame) -> None:
        """Publish the frame's recorded camera JPEGs (raw bytes per topic)."""
        for cam, jpeg in frame.images.items():
            pub = self.video_pubs.get(cam)
            if pub is not None and jpeg:
                pub.publish(jpeg)

    def _wait_for_user_start(self) -> None:
        """Block until one line arrives on stdin.

        Interactively this is the "Press Enter to start" safety prompt.
        Under the app backend, stdin is a pipe and the backend writes a
        newline on POST /replay/begin.

        The "waiting" status is republished once a second while blocked:
        zenoh pub/sub is not durable, and the backend's subscriber may
        still be completing peer discovery when the first publish fires.
        """
        logger.info("⏸️ Press Enter to start replay...")
        try:
            while True:
                self._publish_status("waiting")
                ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                if ready:
                    line = sys.stdin.readline()
                    if line == "":
                        # EOF — stdin closed (e.g. `< /dev/null`), no gate.
                        logger.warning("stdin closed; starting replay without gate")
                    break
        except (OSError, ValueError):
            # stdin unusable (closed descriptor / not selectable).
            logger.warning("stdin unavailable; starting replay without gate")
        logger.info("▶️ Starting replay!")

    def _sleep_publishing(self, seconds: float, state: str) -> None:
        """Sleep while republishing the given status state every 0.5s."""
        deadline = time.time() + seconds
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._publish_status(state)
            time.sleep(min(0.5, remaining))

    def _initialize_emergency_stop_state(self, first_frame: ReplayFrame) -> None:
        """Initialize robot controller emergency stop state.

        This method ensures the robot controller is in the proper state
        by first activating emergency stop, then sending the first command
        to disable it and move to the initial position. This two-step
        process ensures reliable state initialization.
        """
        first_action = first_frame.components
        first_safety_flags = first_frame.safety_flags

        if not first_action:
            logger.warning("⚠️ No first action available, skipping estop initialization")
            return

        # Step 1: Activate emergency stop to initialize robot controller state
        logger.info(
            "🛑 Step 1: Activating emergency stop to initialize robot controller state..."
        )
        estop_activate_command = {
            "timestamp_ns": time.time_ns(),
            "components": first_action,
            "safety_flags": {
                "emergency_stop": True,
                "exit_requested": False,
            },
        }
        self.command_pub.publish(estop_activate_command)
        self._sleep_publishing(self.ESTOP_ACTIVATION_DELAY_SEC, "initializing")

        # Step 2: Send first command to disable emergency stop
        logger.info(
            "✅ Step 2: Sending first command to disable emergency stop and "
            "move to initial position..."
        )
        first_command = {
            "timestamp_ns": time.time_ns(),
            "components": first_action,
            "safety_flags": first_safety_flags,
        }
        self.command_pub.publish(first_command)
        self._publish_images(first_frame)

        logger.info(
            "⏳ First command sent, waiting for robot to reach initial position..."
        )
        if self.debug:
            logger.debug(f"🔍 First command safety flags: {first_safety_flags}")

        # Wait for robot to reach the initial position
        self._sleep_publishing(self.INITIAL_POSITION_DELAY_SEC, "initializing")
        logger.info("🚀 Proceeding with remaining commands...")

    def _create_command_from_frame(
        self, frame: ReplayFrame
    ) -> Optional[Dict[str, Any]]:
        """Create a robot_commands dict from a replay frame.

        Returns None if the frame has no action data.
        """
        if not frame.components:
            return None
        return {
            "timestamp_ns": time.time_ns(),
            "components": frame.components,
            "safety_flags": frame.safety_flags,
        }

    def _log_frame_debug(self, command_dict: Dict[str, Any]) -> None:
        """Log debug information for a frame."""
        logger.info(
            f"📤 Published frame {self._frame_idx}/{self.loader.num_frames}"
        )
        logger.debug(
            f"🔧 Command components: {list(command_dict.get('components', {}).keys())}"
        )
        logger.debug(f"🔒 Safety flags: {command_dict.get('safety_flags', {})}")

    def run(self) -> None:
        """Main replay loop.

        Orchestrates the entire replay sequence:
        1. Wait for the start gate (stdin newline)
        2. Initialize emergency stop state + move to initial position
        3. Replay recorded frames at the recording rate
        """
        self.running = True

        logger.info(f"🚀 Starting replay system at {self.publish_rate}Hz")

        # Wait for the start gate before any robot motion.
        self._wait_for_user_start()

        frames = self.loader.frames()
        first_frame = next(frames, None)
        if first_frame is None:
            logger.warning("⚠️ Episode has no frames — nothing to replay")
            self._publish_status("done")
            self.running = False
            return

        # Initialize robot controller estop state + initial pose.
        self._publish_status("initializing")
        self._initialize_emergency_stop_state(first_frame)

        # Replay the remaining frames. Reset the rate limiter so it paces
        # from now — it was constructed before the start gate and the
        # initial-position delays, and would otherwise burst to "catch up".
        self.rate_limiter.reset()
        total = self.loader.num_frames
        logger.info(f"🔄 Replaying remaining {total - 1} frames...")
        for frame in frames:
            if not self.running:
                break
            self._frame_idx = frame.index
            try:
                command_dict = self._create_command_from_frame(frame)
                if command_dict:
                    self.command_pub.publish(command_dict)
                    self._publish_images(frame)
                    if self.debug and frame.index % self.DEBUG_LOG_INTERVAL == 0:
                        self._log_frame_debug(command_dict)
                self._publish_status("playing")
            except Exception as e:
                logger.error(f"❌ Error in replay loop at frame {frame.index}: {e}")
                self._publish_status("error")
                break
            self.rate_limiter.sleep()
        else:
            self._frame_idx = total
            self._publish_status("done")

        logger.info(f"✅ Replay completed! Published {self._frame_idx} frames")
        self.running = False

    def stop(self) -> None:
        """Stop replay and cleanup resources.

        Shuts down all communication channels.
        """
        self.running = False
        self.node.shutdown()  # Dexcomm Node cleanup
        logger.info("🛑 ReplayRecorder stopped")


def main(
    episode_path: str,
    namespace: str = "",
    debug: bool = False,
) -> int:
    """Main entry point for replay recorder.

    Args:
        episode_path: Path to the recorded episode directory — either an
            MDP episode (contains transitions.pkl) or an MCAP episode
            (contains episode.mcap). A direct path to a transitions.pkl
            also works (legacy CLI compat).
        namespace: Zenoh namespace for topic isolation.
        debug: Enable debug output.

    Returns:
        Exit code (0 for success, 1 for error).

    Example:
        python -m omniteleop.record.replay_record \\
            --episode-path ~/recordings/07-17-2026/episode_0001_20260717_143052 --debug
    """
    setup_logging(debug)

    recorder = None
    try:
        recorder = ReplayRecorder(
            episode_path=episode_path,
            namespace=namespace,
            debug=debug,
        )
        recorder.run()
    except KeyboardInterrupt:
        logger.info("⚡ Shutting down...")
        if recorder:
            recorder.stop()
    except Exception as e:
        logger.error(f"❌ Replay error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
