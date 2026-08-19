"""BaseRecorder: shared skeleton for MDPRecorder and MCAPRecorder.

Owns control modes (keyboard/pedal/joycon), action tracking from the robot
commands topic, lifecycle methods (start/end/discard episode), and the
record-loop thread skeleton. Four subclass hooks plug in storage:
_setup_storage, _collect_observation, _write_frame, _finalize_storage.

Subclasses implement their own initialize() that sets up the robot (and may
call super().initialize() to attach the shared subscribers).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter

from omniteleop.common import get_config
from omniteleop.record.errors import RequiredSensorError

# Optional deps — keyboard pedal mode and rerun visual indicator.
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


JOINT_COMPONENTS = ("left_arm", "right_arm", "torso", "head", "left_hand", "right_hand")

# Fallbacks for toggles the robot YAML's ``recorder.components`` omits.
#
# Core joints (arms/torso/head) default on; optional sensors default off. End
# effectors default ON to match the local pre-0.5.x behaviour — the deployment
# configs (``*_gripper``, ``*_f5d6``) all carry hands. A hand toggle should be
# turned off explicitly when the robot variant has no end effector.
DEFAULT_RECORD_COMPONENTS: Dict[str, bool] = {
    "left_arm": True,
    "right_arm": True,
    "torso": True,
    "head": True,
    "left_hand": True,
    "right_hand": True,
    "head_left_rgb": True,
    "head_right_rgb": True,
    "head_left_depth": False,
    "left_wrist_rgb": False,
    "right_wrist_rgb": False,
    "left_wrist_wrench": False,
    "right_wrist_wrench": False,
}


class BaseRecorder:
    """Shared recorder skeleton. See module docstring."""

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        record_mode: str = "joycon",
        show_rerun: bool = False,
    ):
        self.debug = debug
        self.record_mode = record_mode
        self.show_rerun = show_rerun and RERUN_AVAILABLE

        # Zenoh node (subscribers attached in initialize()).
        self._node = Node(name=self._node_name(), namespace=namespace)
        self.subscriber = None
        self.control_subscriber = None

        # Config.
        self.config = get_config()
        recorder_config = self.config.get("recorder", {})

        save_dir_str = recorder_config.get("save_dir", "recordings")
        save_path = Path(save_dir_str)
        if not save_path.is_absolute():
            save_path = Path.home() / save_path
        self.save_dir = save_path

        self.episode_prefix = recorder_config.get("episode_prefix", "episode")
        self.record_rate = recorder_config.get("record_rate", 20.0)
        resolution = recorder_config.get("image_resolution", [640, 480])
        self.image_resolution = tuple(resolution) if isinstance(resolution, list) else (640, 480)
        self.jpeg_quality = recorder_config.get("jpeg_quality", 90)
        self.auto_stop_on_estop = recorder_config.get("auto_stop_on_estop", True)
        wrist_adapter_config = recorder_config.get("wrist_camera_adapter", {}) or {}
        self.sensor_abort_after_s = float(
            wrist_adapter_config.get("abort_after_s", 1.0)
        )

        components_config = recorder_config.get("components", {}) or {}
        self.record_components: Dict[str, bool] = {
            name: bool(components_config.get(name, default))
            for name, default in DEFAULT_RECORD_COMPONENTS.items()
        }

        # Episode state.
        self.episode_num = 0
        self.episode_dir: Optional[Path] = None
        self.is_recording = False
        self.episode_start_time: Optional[float] = None
        self.episode_start_timestamp_ns: Optional[int] = None

        # Command + action tracking.
        self.latest_command = None
        self.command_lock = threading.Lock()
        self.current_action: Dict[str, Any] = {}
        self.last_action: Dict[str, Any] = {}
        self.action_lock = threading.Lock()

        # Record thread.
        self.record_thread: Optional[threading.Thread] = None
        self.record_running = False
        self.rate_limiter = RateLimiter(self.record_rate)
        self._episode_finish_lock = threading.Lock()
        self._auto_discard_pending = False
        self.pending_metadata: Dict[str, Any] = {}

        # Pedal-mode state.
        self.pedal_key_press_times: Dict[str, float] = {}
        self.pedal_key_lock = threading.Lock()
        self.pedal_hold_duration = 1.0
        self.pedal_running = False

        # Stats.
        self.total_transitions = 0
        self.total_episodes = 0
        self.transitions_in_episode = 0

        # Rerun init (best-effort; disable on failure).
        if self.show_rerun:
            try:
                rr.init(f"{self.__class__.__name__} Status", spawn=True)
            except Exception as e:
                logger.warning(f"Failed to initialize rerun: {e}")
                self.show_rerun = False

    # Subclasses may override to name their node differently.
    def _node_name(self) -> str:
        return self.__class__.__name__.lower()

    # ─── Component availability ────────────────────────────────────────────

    def _drop_unavailable_components(self, robot: Any) -> None:
        """Turn off joint toggles whose hardware this robot doesn't have.

        ``record_components`` is what the YAML *asks* for; the live robot is
        the authority on what exists. Reading an absent component raises in
        dexcontrol ("Component 'left_hand' is not available on this robot"),
        which would otherwise blow up every record-loop tick, so reconcile
        once at init and warn instead.

        Subclasses call this right after building their ``Robot``; that is the
        point where ``record_components`` is final, so this also logs the
        effective set.
        """
        has_component = getattr(robot, "has_component", None)
        if has_component is None:
            return
        for comp in JOINT_COMPONENTS:
            if not self.record_components.get(comp, False):
                continue
            try:
                present = bool(has_component(comp))
            except Exception as e:  # noqa: BLE001 — never block startup on this
                logger.debug(f"has_component({comp}) failed: {e}")
                continue
            if present:
                continue
            self.record_components[comp] = False
            logger.warning(
                f"recorder.components.{comp} is enabled but this robot has no "
                f"{comp} — not recording its state or actions"
            )

        enabled = [k for k, v in self.record_components.items() if v]
        disabled = [k for k, v in self.record_components.items() if not v]
        logger.info(f"Recording: {', '.join(enabled) or 'nothing'}")
        logger.info(f"Not recording: {', '.join(disabled) or 'nothing'}")

    # ─── Action resolution (pure logic) ────────────────────────────────────

    def _resolve_action(self, joint_pos: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve action per the three-step priority:
        current_action > last_action > observation (joint_pos). Mutates last_action.
        """
        resolved: Dict[str, Any] = {}
        with self.action_lock:
            for comp in JOINT_COMPONENTS:
                if not self.record_components.get(comp, False):
                    continue
                if comp in self.current_action:
                    resolved[comp] = self.current_action[comp]
                elif comp in self.last_action:
                    resolved[comp] = self.last_action[comp]
                elif comp in joint_pos:
                    resolved[comp] = {"pos": joint_pos[comp]}
                if comp in resolved:
                    self.last_action[comp] = resolved[comp]

            if "chassis" in self.current_action:
                resolved["chassis"] = self.current_action["chassis"]
                self.last_action["chassis"] = self.current_action["chassis"]
            elif "chassis" in self.last_action:
                resolved["chassis"] = self.last_action["chassis"]
        return resolved

    # ─── Subclass hooks (abstract) ─────────────────────────────────────────

    def _setup_storage(self, metadata: Dict[str, Any]) -> None:
        """Open the episode's storage. Subclasses must set `self.episode_dir`."""
        raise NotImplementedError

    def _collect_observation(self) -> Dict[str, Any]:
        """Collect one frame of robot state.

        The returned dict must contain at least `"joint_pos"` (the record-loop
        reads it for the observation-as-action fallback). All other keys are
        subclass-defined and passed through to `_write_frame` unchanged.
        """
        raise NotImplementedError

    def _write_frame(
        self,
        timestamp_ns: int,
        resolved_action: Dict[str, Any],
        observation: Dict[str, Any],
        safety_flags: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def _finalize_storage(self, success: bool) -> None:
        raise NotImplementedError

    # ─── Initialization (shared subscribers) ───────────────────────────────

    def initialize(self) -> None:
        """Attach zenoh subscribers. Subclasses override to set up the robot;
        they should call super().initialize() first."""
        commands_topic = self.config.get_topic("robot_commands")
        self.subscriber = self._node.create_subscriber(
            commands_topic,
            callback=self._on_command_received,
            decoder=DictDataCodec.decode,
        )

        # Attach recorder_control subscriber in every mode so an external
        # orchestrator (e.g. the app backend GUI) can drive start/stop/discard
        # via zenoh alongside whatever physical input the mode uses.
        control_topic = self.config.get_topic("recorder_control")
        self.control_subscriber = self._node.create_subscriber(
            control_topic,
            callback=self._on_control_received,
            decoder=DictDataCodec.decode,
        )
        logger.info(f"Subscribed to recorder_control topic ({self.record_mode} mode)")

        if self.record_mode == "keyboard":
            logger.info("Keyboard mode: start/stop via stdin (s/e/d/q)")
        elif self.record_mode == "pedal":
            if not PYNPUT_AVAILABLE:
                raise ImportError("pynput is required for pedal mode")
            logger.info("Pedal mode: hold a=start, b=end, c=discard for 1s")

    # ─── Zenoh callbacks ───────────────────────────────────────────────────

    def _on_command_received(self, data: Dict[str, Any]) -> None:
        with self.command_lock:
            self.latest_command = data
        with self.action_lock:
            components = data.get("components", {})
            for name, value in components.items():
                self.current_action[name] = value
        safety_flags = data.get("safety_flags", {})
        if self.auto_stop_on_estop and self.is_recording and safety_flags.get("emergency_stop", False):
            logger.info("Emergency stop detected, ending episode")
            self.end_episode()
        if safety_flags.get("exit_requested", False) and self.is_recording:
            logger.info("Exit signal received")
            self.end_episode()

    def _on_control_received(self, data: Dict[str, Any]) -> None:
        command = data.get("command", "")
        if command == "start" and not self.is_recording:
            self.start_episode(data.get("metadata", {}))
        elif command == "stop" and self.is_recording:
            self.end_episode()
        elif command == "toggle":
            if self.is_recording:
                self.end_episode()
            else:
                self.start_episode(data.get("metadata", {}))
        elif command == "discard" and self.is_recording:
            self.discard_episode()
        elif command == "set_metadata":
            self.pending_metadata = dict(data.get("metadata") or {})
            logger.info(f"Updated pending episode metadata: {self.pending_metadata}")

    # ─── Episode lifecycle ─────────────────────────────────────────────────

    def start_episode(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.is_recording:
            logger.warning("Already recording, ending current episode first")
            self.end_episode()

        # Merge the latest backend-provided pending metadata (e.g. task label,
        # instruction, annotator) with any metadata carried on the control
        # message itself. JoyCon-originated starts only carry source/timestamp,
        # so the pending metadata is what makes webpage annotations show up.
        merged_metadata = dict(self.pending_metadata or {})
        merged_metadata.update(metadata or {})

        self.is_recording = True
        self._auto_discard_pending = False
        self.transitions_in_episode = 0
        self.episode_start_time = time.time()
        self.episode_start_timestamp_ns = time.time_ns()

        with self.action_lock:
            self.current_action.clear()
            self.last_action.clear()

        self._setup_storage(merged_metadata)

        self.record_running = True
        self.record_thread = threading.Thread(
            target=self._record_loop, daemon=True, name=f"{self.__class__.__name__}Loop"
        )
        self.record_thread.start()

        logger.opt(colors=True).info(
            "\n" + "=" * 80 + "\n"
            f"<white><bold><bg green>🎬 RECORDING STARTED - Episode {self.episode_num}</bg green></bold></white>\n"
            f"  📁 Directory: {self.episode_dir.name if self.episode_dir else 'N/A'}\n"
            f"  📊 Record Rate: {self.record_rate} Hz\n"
            f"  📝 Metadata: {merged_metadata if merged_metadata else 'None'}\n" + "=" * 80
        )

        self._show_rerun_indicator("start", {
            "episode_num": self.episode_num,
            "directory": self.episode_dir.name if self.episode_dir else "N/A",
            "record_rate": self.record_rate,
            "metadata": merged_metadata if merged_metadata else "None",
        })

    def end_episode(self) -> None:
        with self._episode_finish_lock:
            self._end_episode_unlocked()

    def _end_episode_unlocked(self) -> None:
        if not self.is_recording:
            logger.warning("Not currently recording")
            return
        self.record_running = False
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        self.is_recording = False
        duration = time.time() - (self.episode_start_time or time.time())
        avg_rate = self.transitions_in_episode / duration if duration > 0 else 0.0

        self._finalize_storage(success=True)

        self.total_episodes += 1
        self.episode_num += 1

        logger.opt(colors=True).info(
            "\n" + "=" * 80 + "\n"
            f"<white><bold><bg blue>💾 EPISODE SAVED - Episode {self.episode_num - 1}</bg blue></bold></white>\n"
            f"  📁 Directory: {self.episode_dir.name if self.episode_dir else 'N/A'}\n"
            f"  📊 Transitions: {self.transitions_in_episode}\n"
            f"  ⏱️  Duration: {duration:.1f}s\n"
            f"  📈 Avg Rate: {avg_rate:.1f} Hz\n"
            f"  🎯 Total Episodes: {self.total_episodes}\n"
            f"  📦 Total Transitions: {self.total_transitions}\n" + "=" * 80
        )

        self._show_rerun_indicator("end", {
            "episode_num": self.episode_num - 1,
            "directory": self.episode_dir.name if self.episode_dir else "N/A",
            "transitions": self.transitions_in_episode,
            "duration": f"{duration:.1f}",
            "avg_rate": f"{avg_rate:.1f}",
            "total_episodes": self.total_episodes,
            "total_transitions": self.total_transitions,
        })

    def discard_episode(self, reason: str = "") -> None:
        with self._episode_finish_lock:
            self._discard_episode_unlocked(reason)

    def _discard_episode_unlocked(self, reason: str = "") -> None:
        if not self.is_recording:
            logger.warning("Not currently recording")
            return
        self.record_running = False
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        self.is_recording = False
        duration = time.time() - (self.episode_start_time or time.time())

        avg_rate = self.transitions_in_episode / duration if duration > 0 else 0.0
        discarded_dir_name = self.episode_dir.name if self.episode_dir else "N/A"

        self._finalize_storage(success=False)

        self.episode_num += 1
        reason_line = f"  ❌ Reason: {reason}\n" if reason else ""

        logger.opt(colors=True).info(
            "\n" + "=" * 80 + "\n"
            f"<white><bold><bg red>🗑️ EPISODE DISCARDED - Episode {self.episode_num - 1}</bg red></bold></white>\n"
            f"  📁 Directory: {discarded_dir_name}\n"
            f"  📊 Transitions: {self.transitions_in_episode}\n"
            f"  ⏱️  Duration: {duration:.1f}s\n"
            f"  📈 Avg Rate: {avg_rate:.1f} Hz\n"
            f"  ⚠️  Data was NOT saved\n"
            + reason_line
            + "=" * 80
        )

        self._show_rerun_indicator("discard", {
            "episode_num": self.episode_num - 1,
            "directory": discarded_dir_name,
            "transitions": self.transitions_in_episode,
            "duration": f"{duration:.1f}",
            "avg_rate": f"{avg_rate:.1f}",
        })

    # ─── Record loop ───────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        logger.info(f"Record loop started at {self.record_rate}Hz")
        sensor_failure_started_at: Optional[float] = None
        while self.record_running and self.is_recording:
            try:
                timestamp_ns = time.time_ns()
                with self.command_lock:
                    safety_flags = (self.latest_command or {}).get("safety_flags", {})
                observation = self._collect_observation()
                resolved = self._resolve_action(observation.get("joint_pos", {}))
                if self.transitions_in_episode == 0:
                    logger.info(
                        f"📊 First action - from command: {list(self.current_action.keys())}, "
                        f"resolved: {list(resolved.keys())}"
                    )
                self._write_frame(timestamp_ns, resolved, observation, safety_flags)
                self.transitions_in_episode += 1
                self.total_transitions += 1
                sensor_failure_started_at = None
            except RequiredSensorError as e:
                now = time.monotonic()
                if sensor_failure_started_at is None:
                    sensor_failure_started_at = now
                    logger.error(f"Required recording sensor unhealthy: {e}")
                elif now - sensor_failure_started_at >= self.sensor_abort_after_s:
                    if not self._auto_discard_pending:
                        self._auto_discard_pending = True
                        reason = str(e)
                        logger.error(
                            f"Required sensor unhealthy for "
                            f"{self.sensor_abort_after_s:.1f}s; discarding episode"
                        )
                        threading.Thread(
                            target=self.discard_episode,
                            args=(reason,),
                            daemon=True,
                            name=f"{self.__class__.__name__}AutoDiscard",
                        ).start()
                    return
            except Exception as e:
                logger.error(f"Error in record loop: {e}")
            self.rate_limiter.sleep()
        logger.info("Record loop stopped")

    # ─── Control modes ─────────────────────────────────────────────────────

    def run(self) -> None:
        if self.record_mode == "keyboard":
            self._run_keyboard_mode()
        elif self.record_mode == "pedal":
            self._run_pedal_mode()
        else:
            self._run_subscriber_mode()

    def _run_keyboard_mode(self) -> None:
        logger.info("⌨️ Recorder running in keyboard mode")
        logger.info(
            "🎮 Commands: 's' = start recording, 'e' = end episode, 'd' = delete/discard, 'q' = quit"
        )
        try:
            while True:
                if self.is_recording:
                    stats = self.get_statistics()
                    logger.info(
                        f"📹 Recording episode {stats['current_episode_num']}: "
                        f"{stats['transitions_in_episode']} transitions"
                    )
                else:
                    logger.info("⏸️  Idle - Press 's' to start recording")

                cmd = input("Command (s/e/d/q): ").strip().lower()
                if cmd == "s" and not self.is_recording:
                    logger.info("▶️ Starting new episode...")
                    self.start_episode()
                elif cmd == "e" and self.is_recording:
                    logger.info("⏹️ Ending current episode...")
                    self.end_episode()
                elif cmd == "d" and self.is_recording:
                    logger.info("🗑️ Discarding current episode...")
                    self.discard_episode()
                elif cmd == "q":
                    logger.info("👋 Quitting recorder...")
                    break
                elif cmd == "s" and self.is_recording:
                    logger.warning(
                        "⚠️ Already recording! Press 'e' to end or 'd' to discard current episode first."
                    )
                elif cmd == "e" and not self.is_recording:
                    logger.warning("⚠️ Not recording! Press 's' to start a new episode.")
                elif cmd == "d" and not self.is_recording:
                    logger.warning("⚠️ Not recording! Press 's' to start a new episode.")
                else:
                    logger.warning(f"❓ Unknown command: '{cmd}'")
        except KeyboardInterrupt:
            logger.info("⚡ Recorder interrupted by user")

    def _run_subscriber_mode(self) -> None:
        logger.info("🎧 Recorder running (waiting for commands)")
        try:
            while True:
                if self.is_recording:
                    stats = self.get_statistics()
                    logger.info(
                        f"📹 Recording episode {stats['current_episode_num']}: "
                        f"{stats['transitions_in_episode']} transitions"
                    )
                time.sleep(5.0)
        except KeyboardInterrupt:
            logger.info("⚡ Recorder interrupted by user")

    def _run_pedal_mode(self) -> None:
        if not PYNPUT_AVAILABLE:
            logger.error("❌ Pedal mode requires pynput. Install with: pip install pynput")
            return
        logger.info("🎹 Recorder running in pedal mode")
        logger.info(
            "🎮 Pedal controls: Hold 'a' for 1s = start, 'b' for 1s = end, 'c' for 1s = discard"
        )
        logger.info("Press 'esc' to quit\n")
        self.pedal_running = True
        check_thread = threading.Thread(target=self._check_pedal_hold_duration, daemon=True)
        check_thread.start()
        try:
            with pynput_keyboard.Listener(
                on_press=self._on_pedal_key_press, on_release=self._on_pedal_key_release
            ) as listener:
                while self.pedal_running:
                    if self.is_recording:
                        stats = self.get_statistics()
                        logger.info(
                            f"📹 Recording episode {stats['current_episode_num']}: "
                            f"{stats['transitions_in_episode']} transitions"
                        )
                    else:
                        logger.info("⏸️  Idle - Hold 'a' for 1s to start recording")
                    time.sleep(5.0)
                    if not listener.running:
                        break
        except KeyboardInterrupt:
            logger.info("⚡ Recorder interrupted by user")
        finally:
            self.pedal_running = False
            logger.info("👋 Exiting pedal mode...")

    def _on_pedal_key_press(self, key) -> None:
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                if c in {"a", "b", "c"}:
                    with self.pedal_key_lock:
                        self.pedal_key_press_times.setdefault(c, time.time())
        except AttributeError:
            pass

    def _on_pedal_key_release(self, key) -> bool:
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                with self.pedal_key_lock:
                    self.pedal_key_press_times.pop(c, None)
        except AttributeError:
            pass
        if PYNPUT_AVAILABLE and key == pynput_keyboard.Key.esc:
            self.pedal_running = False
            return False
        return True

    def _check_pedal_hold_duration(self) -> None:
        while self.pedal_running:
            now = time.time()
            with self.pedal_key_lock:
                fired = []
                for c, t0 in list(self.pedal_key_press_times.items()):
                    if now - t0 >= self.pedal_hold_duration:
                        if c == "a" and not self.is_recording:
                            self.start_episode()
                        elif c == "b" and self.is_recording:
                            self.end_episode()
                        elif c == "c" and self.is_recording:
                            self.discard_episode()
                        fired.append(c)
                for c in fired:
                    self.pedal_key_press_times.pop(c, None)
            time.sleep(0.1)

    # ─── Rerun indicator ───────────────────────────────────────────────────

    def _show_rerun_indicator(self, status: str, info: Dict[str, Any]) -> None:
        if not self.show_rerun or not RERUN_AVAILABLE:
            return
        color = {"start": [0, 255, 0], "end": [0, 128, 255], "discard": [255, 0, 0]}.get(status, [128, 128, 128])
        rect_size = 5000.0

        lines = []
        if status == "start":
            lines.append(f"🎬 RECORDING STARTED - Episode {info.get('episode_num', 'N/A')}")
            lines.append(f"📁 Directory: {info.get('directory', 'N/A')}")
            lines.append(f"📊 Record Rate: {info.get('record_rate', 'N/A')} Hz")
            lines.append(f"📝 Metadata: {info.get('metadata', 'None')}")
            label = f"🎬 RECORDING STARTED - Episode {info.get('episode_num', 'N/A')}"
        elif status == "end":
            lines.append(f"💾 EPISODE SAVED - Episode {info.get('episode_num', 'N/A')}")
            lines.append(f"📁 Directory: {info.get('directory', 'N/A')}")
            lines.append(f"📊 Transitions: {info.get('transitions', 'N/A')}")
            lines.append(f"⏱️  Duration: {info.get('duration', 'N/A')}s")
            lines.append(f"📈 Avg Rate: {info.get('avg_rate', 'N/A')} Hz")
            lines.append(f"🎯 Total Episodes: {info.get('total_episodes', 'N/A')}")
            lines.append(f"📦 Total Transitions: {info.get('total_transitions', 'N/A')}")
            label = f"💾 EPISODE SAVED - Episode {info.get('episode_num', 'N/A')}"
        elif status == "discard":
            lines.append(f"🗑️ EPISODE DISCARDED - Episode {info.get('episode_num', 'N/A')}")
            lines.append(f"📁 Directory: {info.get('directory', 'N/A')}")
            lines.append(f"📊 Transitions: {info.get('transitions', 'N/A')}")
            lines.append(f"⏱️  Duration: {info.get('duration', 'N/A')}s")
            lines.append(f"📈 Avg Rate: {info.get('avg_rate', 'N/A')} Hz")
            lines.append("⚠️  Data was NOT saved")
            label = f"🗑️ EPISODE DISCARDED - Episode {info.get('episode_num', 'N/A')}"
        else:
            label = f"[{status.upper()}] ep={info.get('episode_num', '?')}"

        try:
            rr.log("status_indicator/background", rr.Boxes3D(
                half_sizes=[[rect_size, rect_size, rect_size]],
                centers=[[0, 0, 0]],
                colors=[color],
                quaternions=[0, 0, 0, 1],
                fill_mode="solid",
                labels=[label],
                show_labels=True,
            ))
            rr.log("status_indicator/text", rr.TextLog(
                "\n".join(lines),
                level=rr.TextLogLevel.INFO,
            ))
        except Exception as e:
            logger.debug(f"Failed to show rerun indicator: {e}")

    # ─── Stats + cleanup ───────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_transitions": self.total_transitions,
            "total_episodes": self.total_episodes,
            "current_episode_num": self.episode_num,
            "is_recording": self.is_recording,
            "transitions_in_episode": self.transitions_in_episode,
        }

    def cleanup(self) -> None:
        if self.record_mode == "pedal":
            self.pedal_running = False
        if self.is_recording:
            self.end_episode()
        # Subclasses shut down the robot; base does not hold a reference.
        logger.info(f"{self.__class__.__name__} shutdown: {self.total_episodes} episodes, "
                    f"{self.total_transitions} transitions")
