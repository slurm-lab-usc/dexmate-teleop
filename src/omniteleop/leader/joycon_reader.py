#!/usr/bin/env python3
"""Leader program for reading JoyCon controller inputs."""

import os
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional
import tyro

from joycon_lib import DualJoyCon, Button

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm import RateLimiter
from omniteleop.common import JoyConData, get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.platform import default_joycon_backend
from loguru import logger


class JoyConReader:
    """Reads JoyCon inputs and publishes via Zenoh using Node interface."""

    def __init__(
        self,
        namespace: str = "",
        publish_rate: Optional[int] = None,
        topic: Optional[str] = None,
        deadzone: float = 0.1,
        debug: bool = False,
        backend: Optional[str] = None,
    ):
        """Initialize JoyCon reader.

        Args:
            namespace: Optional namespace prefix (e.g., "robot1")
            publish_rate: Override publish rate (uses config if None)
            topic: Override topic (uses config if None)
            deadzone: Analog stick deadzone threshold
            debug: Enable debug output
            backend: joycon_lib backend ("evdev", "hid"). None/"auto" picks the
                one that works on this OS.
        """
        # Initialize Node (namespace is handled automatically by Node)
        self.node = Node(name="joycon_reader", namespace=namespace)

        # Load configuration
        config = get_config()

        # Use config values with overrides
        self.publish_rate = publish_rate or config.get_rate("input_rate", 40)
        self.topic = topic or config.get_topic("exo_joycon")
        self.deadzone = deadzone

        # Backend priority: CLI flag > config > per-OS default. evdev needs the
        # Linux hid-nintendo driver; macOS has no such driver and talks raw HID.
        configured_backend = backend or config.get_joycon_config().get("backend")
        if configured_backend and str(configured_backend).lower() != "auto":
            self.backend = str(configured_backend).lower()
        else:
            self.backend = default_joycon_backend()

        # JoyCon interface
        self.joycons = None

        # Set to True when run() exits due to a detected disconnect.
        self._disconnect_exit = False

        # Zenoh publisher (will be created through Node)
        self.publisher = None

        # Rate limiter for precise timing
        self.rate_limiter = RateLimiter(self.publish_rate)

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "JoyCon", self.publish_rate, refresh_rate=10
            )

        logger.info(
            f"JoyCon reader configured: {self.publish_rate}Hz -> {self.topic} "
            f"(backend: {self.backend})"
        )

    def initialize(self) -> bool:
        """Initialize JoyCon and Zenoh communication."""
        try:
            # Initialize Zenoh publisher through Node (namespace handled automatically)
            self.publisher = self.node.create_publisher(
                self.topic, encoder=DictDataCodec.encode
            )

            # Initialize JoyCon controllers
            logger.info(f"Connecting to JoyCon controllers ({self.backend} backend)...")
            self.joycons = DualJoyCon(use_backend=self.backend)

            # Connect to controllers
            self.joycons.connect(auto_find=True)

            # Test connection
            if not self._test_connection():
                logger.error("JoyCon connection test failed")
                return False

            # Start polling
            self.joycons.start_polling(rate=self.publish_rate)

            logger.success("JoyCon controllers connected")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _test_connection(self) -> bool:
        """Test JoyCon connection."""
        try:
            # Check connection status
            status = self.joycons.is_connected()

            if not status["left"] and not status["right"]:
                logger.warning("No JoyCon controllers found")
                return False

            if status["left"]:
                logger.info("Left JoyCon connected")
            if status["right"]:
                logger.info("Right JoyCon connected")

            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to analog value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _check_joycon_liveness(self) -> Dict[str, bool]:
        """Return per-side liveness, whichever backend is in use.

        ``DualJoyCon.is_connected()`` only checks whether the Python handle is
        non-None, which never flips when a Joy-Con drops off Bluetooth, so we
        need a real liveness signal.

        Newer joycon_lib versions expose ``is_alive()`` — the HID backend
        reports it from failed reads, evdev from the device node. Older
        versions don't, so fall back to the evdev device-node check: the kernel
        removes ``/dev/input/eventN`` immediately on radio disconnect.
        """
        probe = getattr(self.joycons, "is_alive", None)
        if callable(probe):
            try:
                status = probe()
                return {
                    "left": bool(status.get("left", False)),
                    "right": bool(status.get("right", False)),
                }
            except Exception as e:  # noqa: BLE001 — fall through to the path check
                logger.debug(f"is_alive() probe failed: {e}")

        result = {"left": False, "right": False}
        for side in ("left", "right"):
            joycon = getattr(self.joycons, f"{side}_joycon", None)
            if joycon is None:
                continue
            evdev_device = getattr(getattr(joycon, "device", None), "device", None)
            path = getattr(evdev_device, "path", None)
            if path:
                result[side] = os.path.exists(path)
            else:
                # No device node to inspect (non-evdev backend on an older
                # joycon_lib): trust the handle rather than reporting a
                # disconnect we cannot actually observe.
                result[side] = True
        return result

    def _zero_state(self) -> dict:
        """Return a neutral JoyCon state with all buttons released and sticks centered."""
        return {
            "left": {
                "buttons": {
                    "up": False,
                    "down": False,
                    "left": False,
                    "right": False,
                    "l": False,
                    "zl": False,
                    "minus": False,
                    "capture": False,
                    "stick": False,
                },
                "stick": {"x": 0.0, "y": 0.0},
            },
            "right": {
                "buttons": {
                    "a": False,
                    "b": False,
                    "x": False,
                    "y": False,
                    "r": False,
                    "zr": False,
                    "plus": False,
                    "home": False,
                    "stick": False,
                },
                "stick": {"x": 0.0, "y": 0.0},
            },
        }

    def _read_joycon_state(self) -> dict:
        """Read current JoyCon state."""
        data = {
            "left": {"buttons": {}, "stick": {"x": 0.0, "y": 0.0}},
            "right": {"buttons": {}, "stick": {"x": 0.0, "y": 0.0}},
        }

        try:
            # Get combined button state
            button_state = self.joycons.get_button_state()

            # Get both sticks
            left_stick, right_stick = self.joycons.get_sticks()

            # Process left side (left JoyCon)
            data["left"]["buttons"] = {
                "up": button_state.is_pressed(Button.UP),
                "down": button_state.is_pressed(Button.DOWN),
                "left": button_state.is_pressed(Button.LEFT),
                "right": button_state.is_pressed(Button.RIGHT),
                "l": button_state.is_pressed(Button.L),
                "zl": button_state.is_pressed(Button.ZL),
                "minus": button_state.is_pressed(Button.MINUS),
                "capture": button_state.is_pressed(Button.CAPTURE),
                "stick": button_state.is_pressed(Button.L_STICK),
            }

            # Left stick values
            data["left"]["stick"]["x"] = self._apply_deadzone(left_stick.x)
            data["left"]["stick"]["y"] = self._apply_deadzone(left_stick.y)

            # Process right side (right JoyCon)
            data["right"]["buttons"] = {
                "a": button_state.is_pressed(Button.A),
                "b": button_state.is_pressed(Button.B),
                "x": button_state.is_pressed(Button.X),
                "y": button_state.is_pressed(Button.Y),
                "r": button_state.is_pressed(Button.R),
                "zr": button_state.is_pressed(Button.ZR),
                "plus": button_state.is_pressed(Button.PLUS),
                "home": button_state.is_pressed(Button.HOME),
                "stick": button_state.is_pressed(Button.R_STICK),
            }

            # Right stick values
            data["right"]["stick"]["x"] = self._apply_deadzone(right_stick.x)
            data["right"]["stick"]["y"] = self._apply_deadzone(right_stick.y)

        except Exception as e:
            logger.warning(f"Error reading JoyCon state: {e}")

        return data

    def run(self):
        """Main reading loop - runs directly without threading."""
        resolved_topic = self.node.resolve_topic(self.topic)
        logger.info(f"Started publishing to {resolved_topic} at {self.publish_rate}Hz")

        # Start the live display if debug mode
        if self._debug_display:
            logger.info("Starting debug display...")
            self._debug_display.start()
            logger.info("Debug display started")

        try:
            while True:
                status = self._check_joycon_liveness()
                if not (status["left"] and status["right"]):
                    dropped = [s for s in ("left", "right") if not status[s]]
                    logger.error(
                        f"JoyCon disconnect detected ({', '.join(dropped)}); "
                        "publishing zero state and exiting"
                    )
                    zero = self._zero_state()
                    self.publisher.publish(
                        asdict(
                            JoyConData(
                                timestamp_ns=time.time_ns(),
                                left=zero["left"],
                                right=zero["right"],
                            )
                        )
                    )
                    self._disconnect_exit = True
                    break

                # Read JoyCon state
                state = self._read_joycon_state()

                # Create and publish data
                data = JoyConData(
                    timestamp_ns=time.time_ns(),
                    left=state["left"],
                    right=state["right"],
                )

                # Publish via Zenoh (convert dataclass to dict for DictDataCodec)
                self.publisher.publish(asdict(data))

                # Efficient debug output using rich
                if self._debug_display:
                    # Pass the left and right controller data directly
                    debug_data = {"left": data.left, "right": data.right}
                    self._debug_display.print_joycon(debug_data)

                # Use RateLimiter for precise timing
                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        if self.joycons:
            try:
                self.joycons.stop_polling()
                self.joycons.disconnect()
            except Exception:
                pass

        # Node handles cleanup
        self.node.shutdown()

        logger.info("JoyCon reader stopped")


def main(
    namespace: str = "",
    publish_rate: Optional[int] = None,
    topic: Optional[str] = None,
    deadzone: float = 0.1,
    debug: bool = False,
    backend: Optional[str] = None,
):
    """Main entry point for JoyCon reader.

    Configuration is loaded from ROBOT_CONFIG env var (e.g., "vega_1_f5d6" -> vega_1_f5d6.yaml).

    Args:
        namespace: Optional namespace prefix (e.g., "robot1")
        publish_rate: Override publish rate (uses config if None)
        topic: Override topic (uses config if None)
        deadzone: Analog stick deadzone threshold
        debug: Enable debug output
        backend: Force a joycon_lib backend - "evdev" (Linux hid-nintendo) or
            "hid" (raw HID, required on macOS). Defaults to per-OS auto-select.
    """
    # Setup logging
    logger = setup_logging(debug)
    logger.info(
        f"Starting Dexexo JoyCon Reader{f' (namespace: {namespace})' if namespace else ''}"
    )

    reader = JoyConReader(
        namespace=namespace,
        publish_rate=publish_rate,
        topic=topic,
        deadzone=deadzone,
        debug=debug,
        backend=backend,
    )

    if not reader.initialize():
        logger.error("Failed to initialize JoyCon reader")
        return 1

    # Run directly (no threading needed)
    reader.run()

    if reader._disconnect_exit:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
