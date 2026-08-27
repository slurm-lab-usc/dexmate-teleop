#!/usr/bin/env python3
"""OmniTeleop App Backend.

Orchestrates the full teleoperation stack:
  - Launches leader readers (arm, joycon) and follower processes
    (command_processor, robot_controller) as managed subprocesses.
  - Streams camera images and robot state to the frontend via WebSocket.
  - Optionally runs the MDP recorder when --record-mode is set.

Usage:
    python -m omniteleop.app.backend.app_backend                 # teleop only
    python -m omniteleop.app.backend.app_backend --record-mode   # teleop + recording
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2  # type: ignore
import numpy as np
import tyro
from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcontrol.robot import Robot
from dexcontrol.core.config import get_robot_config
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from omniteleop.common import get_config
from omniteleop.common.joint_state_safety import has_active_joint_error
from omniteleop.common.logging import setup_logging
from omniteleop.read_only_robot import ReadOnlyRobot, SoftwareEStopControl

from omniteleop.app.backend.state_checker import StateChecker
from omniteleop.app.backend.backend_utils.state_manager import state_manager
from omniteleop.app.backend.backend_utils.state_publisher import state_publisher
from omniteleop.app.backend.backend_utils.video_publisher import video_publisher
from omniteleop.record.base_recorder import DEFAULT_RECORD_COMPONENTS
from omniteleop.record.episode_loader import detect_format, inspect_episode


# Recording is now delegated to the mcap_recorder subprocess (spawned when
# record_mode=true at /teleop/start). The backend only publishes control
# commands on the recorder_control topic and reads status from the
# subprocess's stdout — no in-process MCAP writing.


def _create_doctor_robot() -> Robot:
    """Create a Doctor control client that is safe to construct in a worker thread."""
    return Robot(
        auto_shutdown=False,
        configure_default_state=False,
    )


_ARM_NAMES = ("left_arm", "right_arm")
_ARM_DOF = 7


def _read_arm_brake_status(bot: Robot) -> Dict[str, Any]:
    """Return per-joint brake-release state and live position for both arms."""
    arms: Dict[str, Any] = {}
    any_released = False
    for arm_name in _ARM_NAMES:
        arm = getattr(bot, arm_name, None)
        if arm is None:
            continue
        position = np.asarray(arm.get_joint_pos(), dtype=float)
        limits = arm.joint_pos_limit
        if position.shape != (_ARM_DOF,) or not bool(np.isfinite(position).all()):
            raise RuntimeError(f"{arm_name} returned invalid position state")
        if limits is None or limits.shape != (_ARM_DOF, 2):
            raise RuntimeError(f"{arm_name} joint limits are unavailable")

        status = arm.get_brake_status()
        if not isinstance(status, dict) or status.get("success") is not True:
            raise RuntimeError(f"{arm_name} brake status query failed: {status}")
        enabled = bool(status.get("enabled", False))
        raw_joints = status.get("joints") or []
        try:
            released = {
                int(index)
                for index in raw_joints
                if 0 <= int(index) < _ARM_DOF
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{arm_name} returned invalid released-joint data: {raw_joints}"
            ) from exc
        # Some firmware versions report enabled=true with an empty list for
        # an all-joint release. Treat that conservatively as every joint.
        if enabled and not released:
            released = set(range(_ARM_DOF))

        joint_names = list(getattr(arm, "joint_name", []) or [])
        side_prefix = "L" if arm_name == "left_arm" else "R"
        joints = []
        for index in range(_ARM_DOF):
            lower = float(limits[index, 0])
            upper = float(limits[index, 1])
            value = float(position[index])
            joints.append(
                {
                    "index": index,
                    "name": (
                        str(joint_names[index])
                        if len(joint_names) == _ARM_DOF
                        else f"{side_prefix}_arm_j{index + 1}"
                    ),
                    "position_rad": value,
                    "limits_rad": [lower, upper],
                    "outside_limits": not lower <= value <= upper,
                    "released": enabled and index in released,
                }
            )
        any_released = any_released or enabled
        arms[arm_name] = {
            "enabled": enabled,
            "released_joints": sorted(released) if enabled else [],
            "message": str(status.get("message", "")),
            "joints": joints,
        }

    return {
        "ok": True,
        "any_released": any_released,
        "arms": arms,
    }


def _require_arm_brakes_engaged(report: Dict[str, Any]) -> None:
    """Refuse motion whenever any arm reports brake release enabled."""
    released = [
        f"{arm_name} joints={arm_status.get('released_joints', [])}"
        for arm_name, arm_status in report.get("arms", {}).items()
        if arm_status.get("enabled")
    ]
    if released:
        # Precondition violation → ValueError so handlers map it to 409,
        # not a server-side 500.
        raise ValueError(
            "Brake release is active; disable all brake release before movement ("
            + "; ".join(released)
            + ")"
        )


def _control_arm_brake(
    bot: Robot,
    action: str,
    *,
    arm_name: str | None = None,
    joint_index: int | None = None,
    confirmed: bool = False,
) -> Dict[str, Any]:
    """Release one selected joint or disable release on every active arm."""
    before = _read_arm_brake_status(bot)
    selected: Dict[str, Any] | None = None

    if action == "release":
        if confirmed is not True:
            raise ValueError("Explicit physical-support confirmation is required")
        if before["any_released"]:
            raise ValueError(
                "A joint is already released; disable all brake release before "
                "selecting another joint"
            )
        if arm_name not in _ARM_NAMES:
            raise ValueError("arm must be 'left_arm' or 'right_arm'")
        if isinstance(joint_index, bool) or not isinstance(joint_index, int):
            raise ValueError("joint_index must be an integer from 0 to 6")
        if not 0 <= joint_index < _ARM_DOF:
            raise ValueError("joint_index must be an integer from 0 to 6")
        arm = getattr(bot, arm_name, None)
        if arm is None:
            raise ValueError(f"{arm_name} is unavailable")
        selected = before["arms"][arm_name]["joints"][joint_index]
        response = arm.release_brake(True, joints=[joint_index])
        if not isinstance(response, dict) or response.get("success") is not True:
            raise RuntimeError(
                f"{arm_name} joint {joint_index + 1} brake release failed: {response}"
            )
    elif action == "engage":
        for current_arm_name, arm_status in before["arms"].items():
            if not arm_status.get("enabled"):
                continue
            arm = getattr(bot, current_arm_name)
            response = arm.release_brake(False)
            if not isinstance(response, dict) or response.get("success") is not True:
                raise RuntimeError(
                    f"{current_arm_name} brake engage failed: {response}"
                )
    elif action != "status":
        raise ValueError("action must be 'release', 'engage', or 'status'")

    result = _read_arm_brake_status(bot)
    result["action"] = action
    if selected is not None:
        result["selected_joint"] = selected
        result["selected_arm"] = arm_name
    if action == "engage" and result["any_released"]:
        raise RuntimeError("Brake release remained active after engage request")
    return result


def _resolve_zenoh_cert(zenoh_base: Path, name: str) -> Optional[Path]:
    """Resolve a certificate name to its config file path.

    Tries, in order:
      1. <zenoh_base>/<name>/zenoh_peer_config.json5  (directory bundle)
      2. <zenoh_base>/<name>.dzcfg                    (flat file)

    Returns the resolved Path if found, else None.
    """
    candidate_dir = zenoh_base / name / "zenoh_peer_config.json5"
    if candidate_dir.exists():
        return candidate_dir
    candidate_flat = zenoh_base / f"{name}.dzcfg"
    if candidate_flat.exists():
        return candidate_flat
    return None


def _autodetect_zenoh_cert(zenoh_base: Path) -> Optional[Path]:
    """Pick the first available cert under zenoh_base, preferring directory bundles.

    Considers:
      - Directories containing zenoh_peer_config.json5
      - Flat .dzcfg files

    Returns the config file Path, or None if nothing is found.
    """
    dirs = sorted(
        p / "zenoh_peer_config.json5"
        for p in zenoh_base.iterdir()
        if p.is_dir() and (p / "zenoh_peer_config.json5").exists()
    )
    if dirs:
        return dirs[0]
    flat = sorted(zenoh_base.glob("*.dzcfg"))
    if flat:
        return flat[0]
    return None


class TeleopApp:
    """Manages the full teleop stack and exposes a FastAPI server to the frontend."""

    # Subprocess that owns the arms. It is stopped last and given extra time for
    # measured-pose hold, mode transition, and drift verification; see
    # omniteleop.follower.safe_shutdown. A successful disable response is not
    # evidence of mechanical brake engagement.
    ARM_HOLDER_PROC = "robot_controller"
    # release_brake(False) may take 45 s, followed by the mode service timeout.
    ARM_HOLDER_STOP_GRACE_S = 70.0
    PROC_STOP_GRACE_S = 5.0
    # A command-stream E-Stop request older than this has no live producer
    # behind it. CommandProcessor publishes at 40 Hz, so this is many missed
    # cycles, not a hiccup.
    CMD_ESTOP_STALE_S = 2.0

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        rpi_mode: bool = False,
        lock_hand_collision_avoidance: bool = False,
        unlock_torso: bool = False,
    ):
        self.namespace = namespace
        self.debug = debug
        self.rpi_mode = rpi_mode
        self.lock_hand_collision_avoidance = lock_hand_collision_avoidance
        self.unlock_torso = unlock_torso
        self.record_mode = False  # set per-launch from /teleop/start body
        # "mcap" or "mdp" — which recorder subprocess to spawn when record_mode=True.
        self.recorder_format: str = "mcap"
        # Leader stack: "exoskeleton" | "vr" | "vr_sim". Picked per /teleop/start.
        self.leader_mode: str = "exoskeleton"

        self.config = get_config()
        recorder_cfg = self.config.get("recorder", {})

        # ---- paths & rates ------------------------------------------------
        save_dir_str = recorder_cfg.get("save_dir", "recordings")
        save_path = Path(save_dir_str)
        if not save_path.is_absolute():
            save_path = Path.home() / save_path
        self.save_dir = save_path
        # Saved-episode-count index: day ("MM-DD-YYYY") → task_label → count.
        # Persisted next to save_dir so totals survive backend restarts; updated
        # on the recorder's "EPISODE SAVED" marker (discards never count).
        self._record_counts_path = self.save_dir / ".record_counts.json"
        self._record_counts: Dict[str, Dict[str, int]] = self._load_record_counts()
        self._current_task_label: str = ""
        # Metadata saved from the web UI and used as the default for JoyCon
        # initiated recording (so L+R starts carry task label/instruction/etc).
        self._pending_record_metadata: Dict[str, Any] = {}
        self.episode_prefix = recorder_cfg.get("episode_prefix", "episode")
        self.record_rate = recorder_cfg.get("record_rate", 20.0)
        resolution = recorder_cfg.get("image_resolution", [640, 480])
        self.image_resolution = (
            tuple(resolution) if isinstance(resolution, list) else (640, 480)
        )
        self.auto_stop_on_estop = recorder_cfg.get("auto_stop_on_estop", True)

        # ---- component flags -----------------------------------------------
        # Load all component flags from config; any key present in the yaml is
        # respected. Keys not in the yaml fall back to the recorders' shared
        # DEFAULT_RECORD_COMPONENTS so the GUI reports exactly what the
        # recorder subprocess will actually write.
        components_cfg = recorder_cfg.get("components", {}) or {}
        self.record_components: Dict[str, bool] = {
            **DEFAULT_RECORD_COMPONENTS,
            **{k: bool(v) for k, v in components_cfg.items()},
        }
        self.save_images = any(
            self.record_components.get(k)
            for k in [
                "head_left_rgb",
                "head_right_rgb",
                "head_left_depth",
                "left_wrist_rgb",
                "right_wrist_rgb",
            ]
        )

        # ---- subprocess handles --------------------------------------------
        self._procs: List[subprocess.Popen] = []
        self._proc_names: List[str] = []  # parallel name list for each proc
        # name -> returncode of subprocesses that exited unexpectedly while the
        # stack was running (populated by _tail_proc, surfaced via error_state).
        self._proc_crashed: Dict[str, int] = {}
        self._stop_in_progress = False  # True while _stop_teleop_procs runs
        self._teleop_running = False  # True once start_teleop() succeeds
        self.teleop_lifecycle = "stopped"
        self._lifecycle_task: Optional[asyncio.Task] = None
        self._doctor_lock = asyncio.Lock()
        self._teleop_env: Dict[str, str] = {}  # env passed from frontend at start
        # Last resolved robot/zenoh env from any successful launch (teleop or
        # replay). Survives _stop_teleop_procs so a subsequent replay can reuse
        # the known-good robot connection when its own form fields are blank.
        self._last_launch_env: Dict[str, str] = {}

        # ---- replay state ----------------------------------------------------
        # Replay and teleop are mutually exclusive: starting one kills the
        # other's subprocesses. The replay stack = robot_controller +
        # replay_record; replay_record is held at its stdin start gate until
        # POST /replay/begin.
        self._replay_running = False
        self._replay_proc: Optional[subprocess.Popen] = None  # stdin gate handle
        self._replay_lock = threading.Lock()
        self._replay_status: Dict[str, Any] = {}  # latest replay/status message
        self._replay_frames: Dict[str, bytes] = {}  # camera -> latest JPEG bytes
        self._replay_episode: Dict[str, Any] = {}  # inspect_episode() of current
        self._replay_sub_topics: set = set()  # zenoh topics already subscribed

        # ---- per-process log ring-buffers & subscribers -------------------
        # _proc_logs[name] = deque of {"source": name, "line": text}
        self._proc_logs: Dict[str, List[Dict[str, str]]] = {}
        self._log_lock = threading.Lock()
        self._log_subscribers: List[asyncio.Queue] = []
        self._log_loop: Optional[asyncio.AbstractEventLoop] = None

        # ---- robot / state -------------------------------------------------
        self.robot: Optional[ReadOnlyRobot] = None
        self._estop_control: Optional[SoftwareEStopControl] = None
        self._node = Node(name="app_backend", namespace=namespace)
        # Dual estop tracking: GUI and command_processor are independent sources.
        # Effective estop = gui_estop OR cmd_estop.
        self._gui_estop = False
        self._cmd_estop = False
        # True while CommandProcessor is holding E-Stop because the exo drifted
        # away from the robot during a pause and must be re-aligned.
        self._cmd_pending_realign = False
        # When the last robot_commands message arrived. Used to tell a live
        # E-Stop request from one latched by a producer that has since exited.
        self._last_command_at: Optional[float] = None
        # Per-session subscribers on the shared node, retired between runs.
        self._teleop_subs: List[Any] = []

        self.state_checker = None  # initialized in _start_teleop_procs

        self.robot_state = "BOOT"
        self._robot_state_lock = threading.Lock()
        self._state_checker_thread: Optional[threading.Thread] = None
        self._state_checker_running = False

        self.frontend_ready = False

        # Tracks whether we've already called state_checker.mark_motion_started()
        # for the current teleop session. Reset in _stop_teleop_procs.
        self._motion_started_signaled: bool = False

        # ---- recording state (advisory — authoritative state is in the
        #      mcap_recorder subprocess; these mirrors are driven by stdout) --
        self.is_recording = False
        # One of: "idle", "recording", "saved", "discarded", "crashed", "stopped"
        self.recording_status: str = "idle"
        self.recorder_ready: bool = False
        self.episode_dir_name: Optional[str] = None
        self._recorder_ctrl_pub = None  # lazily created in _start_teleop_procs

        # Action tracking from robot_commands topic — used for UI
        # active_components ring (not for recording anymore).
        self._action_lock = threading.Lock()
        self.current_action: Dict[str, Any] = {}
        self._component_last_seen: Dict[str, float] = {}
        self._active_component_window = 0.5  # seconds
        self._latest_robot_joints: Dict[str, Any] = {}  # from robot/joints topic

        # ---- FastAPI -------------------------------------------------------
        self.app = FastAPI(title="OmniTeleop App Backend")
        self._setup_fastapi()

    # =========================================================================
    # FastAPI setup
    # =========================================================================

    def _setup_fastapi(self):
        app = self.app

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # ------------------------------------------------------------------
        # State callback fed into state_publisher
        # ------------------------------------------------------------------
        async def build_state_dict() -> Dict[str, Any]:
            if not self.frontend_ready:
                return {
                    "timestamp_posix": time.time(),
                    "robots": {
                        "robot": [
                            {
                                "topic": "control_state",
                                "data_type": "string",
                                "data": "BOOT",
                            },
                            {"topic": "episode_id", "data_type": "string", "data": ""},
                            {
                                "topic": "error_state",
                                "data_type": "json",
                                "data": {
                                    "state": "BOOT",
                                    "message": "Waiting for frontend…",
                                },
                            },
                        ]
                    },
                }

            is_recording = self.is_recording
            episode_id = self.episode_dir_name

            is_estop = await state_manager.is_estop()

            with self._robot_state_lock:
                robot_state = self.robot_state

            # Compute unified control_state.
            # StateChecker returns the raw robot state (BOOT/DIAGNOSIS/ALIGN/ACTIVE).
            # The backend maps this to a control_state that the frontend displays.
            #
            # Core transitions:
            #   BOOT → DIAGNOSIS → ALIGN → (alignment done) → PAUSED → ACTIVE
            #
            # During ALIGN, CommandProcessor forces emergency_stop=True — but we
            # still show ALIGN (not PAUSE) so the user knows they need to match
            # the robot's pose. Once StateChecker says ACTIVE (alignment done),
            # CommandProcessor STILL forces emergency_stop=True until the user
            # explicitly cycles estop. At that point we show PAUSED — alignment
            # work is done, the user just needs to release estop to start motion.
            if robot_state == "ALIGN":
                control_state = (
                    "ALIGN"
                    if not self._motion_started_signaled
                    else ("PAUSE" if is_estop else "ACTIVE")
                )
            elif robot_state == "ACTIVE" and not self._motion_started_signaled:
                # First time ACTIVE — StateChecker confirmed alignment passed.
                # CommandProcessor still has emergency_stop=True (post-align
                # fresh-engagement gate), so the effective estop IS on. Show
                # PAUSED, not ALIGN — the aligning work is done.
                control_state = "PAUSE" if is_estop else "ACTIVE"
            elif is_estop:
                control_state = (
                    "ALIGN" if not self._motion_started_signaled else "PAUSE"
                )
            elif is_recording:
                control_state = "RECORD"
            elif robot_state == "BOOT":
                control_state = "BOOT"
            elif robot_state == "DIAGNOSIS":
                control_state = "DIAGNOSIS"
            elif robot_state == "ACTIVE":
                control_state = "ACTIVE"
            else:
                control_state = "BOOT"

            # A re-alignment hold outranks the mapping above. CommandProcessor
            # is forcing E-Stop until the exo comes back to the robot's pose,
            # so show ALIGN — PAUSE would tell the operator to just release the
            # joycon, which does nothing here. get_alignment_status() below
            # does not depend on the motion-started latch, so the per-joint
            # guidance is live again without un-latching the state machine.
            if self._cmd_pending_realign:
                control_state = "ALIGN"

            # First time control_state reaches ACTIVE (estop released after
            # alignment), permanently latch the state_checker out of ALIGN.
            # This prevents ALIGN from re-appearing when the robot moves away
            # from its starting position during normal operation.
            if (
                control_state == "ACTIVE"
                and not self._motion_started_signaled
                and self.state_checker
            ):
                self._motion_started_signaled = True
                self.state_checker.mark_motion_started()

            # Robot observations
            robot_states = [
                {
                    "topic": "control_state",
                    "data_type": "string",
                    "data": control_state,
                },
                {
                    "topic": "episode_id",
                    "data_type": "string",
                    "data": episode_id or "",
                },
                {
                    "topic": "recording_status",
                    "data_type": "string",
                    "data": self.recording_status,
                },
                {
                    "topic": "recorder_format",
                    "data_type": "string",
                    "data": self.recorder_format,
                },
                {
                    "topic": "recorder_ready",
                    "data_type": "boolean",
                    "data": self.recorder_ready,
                },
                {
                    "topic": "record_mode",
                    "data_type": "boolean",
                    "data": self.record_mode,
                },
                {
                    "topic": "record_counts",
                    "data_type": "json",
                    "data": self._record_counts_status(),
                },
                {
                    "topic": "teleop_running",
                    "data_type": "boolean",
                    "data": self._teleop_running,
                },
                {
                    "topic": "teleop_lifecycle",
                    "data_type": "string",
                    "data": self.teleop_lifecycle,
                },
                {
                    "topic": "leader_mode",
                    "data_type": "string",
                    "data": self.leader_mode,
                },
            ]

            if self.robot:
                try:
                    obs = await self._get_robot_state()
                    for comp, positions in obs.get("joint_pos", {}).items():
                        robot_states.append(
                            {
                                "topic": f"observation/state/{comp}",
                                "data_type": "array",
                                "data": positions,
                            }
                        )
                except Exception:
                    pass

            if self.state_checker and self.leader_mode == "exoskeleton":
                try:
                    exo = self.state_checker.get_latest_exo_joints()
                    if exo:
                        for side in ("left", "right"):
                            joints = exo.get(side, [])
                            if joints:
                                robot_states.append(
                                    {
                                        "topic": f"observation/exo/{side}_arm",
                                        "data_type": "array",
                                        "data": joints,
                                    }
                                )
                except Exception:
                    pass

            # VR command stream — populated by _on_command_received from the
            # robot_commands topic that paddle_leader publishes. Surfaced as
            # observation/command/<comp> so the Observations tab can show
            # what the operator is currently commanding when leader_mode is
            # vr or vr_sim. Skipped in exoskeleton mode to keep the WS state
            # dict lean.
            if self.leader_mode in ("vr", "vr_sim"):
                with self._action_lock:
                    for comp in ("left_arm", "right_arm", "torso"):
                        comp_data = self.current_action.get(comp)
                        if not comp_data:
                            continue
                        pos = comp_data.get("pos")
                        if pos:
                            robot_states.append(
                                {
                                    "topic": f"observation/command/{comp}",
                                    "data_type": "array",
                                    "data": list(pos),
                                }
                            )

            # Diagnostics / error_state
            if self.state_checker:
                if control_state == "BOOT":
                    topic_info = self.state_checker.get_missing_topics()
                    error_data = {
                        "state": "BOOT",
                        "message": "Required topics not available",
                        "missing_topics": topic_info["missing"],
                        "found_topics": topic_info["found"],
                    }
                elif control_state == "DIAGNOSIS":
                    diag = self.state_checker.get_diagnosis_details()
                    error_data = {
                        "state": "DIAGNOSIS",
                        "message": "Robot not ready — diagnostic checks failed",
                        "diagnostic_checks": diag,
                    }
                    if diag.get("no_component_errors") is False:
                        errors, _ = self.state_checker.get_component_errors()
                        if errors:
                            error_data["component_errors"] = errors
                elif control_state == "ALIGN":
                    error_data = self.state_checker.get_out_of_limit_joints()
                    error_data["alignment"] = (
                        self.state_checker.get_alignment_status()
                    )
                    error_data["state"] = "ALIGN"
                else:
                    error_data = {
                        "state": control_state,
                        "message": "Robot is running"
                        if control_state == "ACTIVE"
                        else control_state,
                    }

                # Subprocesses that died mid-run (see _tail_proc). Surfaced to
                # the frontend immediately instead of waiting for StateChecker.
                if self._proc_crashed:
                    error_data["process_crashed"] = dict(self._proc_crashed)

                robot_states.append(
                    {"topic": "error_state", "data_type": "json", "data": error_data}
                )
            else:
                robot_states.append(
                    {
                        "topic": "error_state",
                        "data_type": "json",
                        "data": {"state": control_state, "message": "No state checker"},
                    }
                )

            robot_states.append(
                {
                    "topic": "estop_state",
                    "data_type": "json",
                    "data": {"software_estop": is_estop},
                }
            )

            if self.state_checker:
                try:
                    sensor_status = self.state_checker.get_sensor_topics_status()
                    robot_states.append(
                        {
                            "topic": "sensor_topics",
                            "data_type": "json",
                            "data": sensor_status,
                        }
                    )
                except Exception:
                    pass

            with self._action_lock:
                cutoff = time.time() - self._active_component_window
                active_components = [
                    c for c, t in self._component_last_seen.items() if t >= cutoff
                ]
            robot_states.append(
                {
                    "topic": "active_components",
                    "data_type": "json",
                    "data": active_components,
                }
            )

            # Replay state — mirrors the replay_record subprocess's
            # replay/status topic plus the episode metadata captured at
            # /replay/start. Drives the Replay tab in the frontend.
            with self._replay_lock:
                replay_status = dict(self._replay_status)
                replay_episode = dict(self._replay_episode)
            robot_states.append(
                {
                    "topic": "replay",
                    "data_type": "json",
                    "data": {
                        "running": self._replay_running,
                        "state": replay_status.get("state", "idle"),
                        "frame_idx": replay_status.get("frame_idx", 0),
                        "total_frames": replay_status.get(
                            "total", replay_episode.get("num_frames", 0)
                        ),
                        "episode": replay_episode.get("path", ""),
                        "episode_name": replay_episode.get("name", ""),
                        "format": replay_episode.get("format", ""),
                        "cameras": replay_episode.get("cameras", []),
                        "rate_hz": replay_status.get(
                            "rate_hz", replay_episode.get("rate_hz", 0)
                        ),
                    },
                }
            )

            # Last-launch robot/zenoh env — lets the Replay tab prefill its
            # form so replays reuse the known-good robot connection without
            # retyping it. ROBOT_NAME is intentionally omitted (may be blank /
            # auto-detected); only the fields the UI form exposes are surfaced.
            robot_states.append(
                {
                    "topic": "launch_env",
                    "data_type": "json",
                    "data": {
                        k: self._last_launch_env.get(k, "")
                        for k in ("ROBOT_NAME", "ROBOT_CONFIG", "ZENOH_CONFIG")
                    },
                }
            )

            return {
                "timestamp_posix": time.time(),
                "robots": {"robot": robot_states},
            }

        @app.on_event("startup")
        async def _capture_loop():
            self._log_loop = asyncio.get_running_loop()

            # Sink backend's own loguru output into the log stream
            def _backend_log_sink(msg):
                entry = {"source": "app_backend", "line": msg.rstrip("\n")}
                with self._log_lock:
                    buf = self._proc_logs.setdefault("app_backend", [])
                    buf.append(entry)
                    if len(buf) > 500:
                        buf.pop(0)
                    subs = list(self._log_subscribers)
                loop = self._log_loop
                if loop and loop.is_running():
                    for q in subs:
                        try:
                            loop.call_soon_threadsafe(q.put_nowait, entry)
                        except Exception:
                            pass

            logger.add(_backend_log_sink, format="{level}: {message}")

        state_publisher.set_state_callback(build_state_dict)
        state_publisher.set_connection_callback(self.on_frontend_connected)
        state_publisher.set_disconnection_callback(self.on_frontend_disconnected)

        # ------------------------------------------------------------------
        # REST endpoints
        # ------------------------------------------------------------------

        @app.get("/health")
        async def health():
            return {"status": "ok", "timestamp": time.time()}

        @app.get("/state")
        async def get_state():
            return JSONResponse(content=await build_state_dict())

        @app.get("/sensors")
        async def get_sensors():
            sensors = [
                {"id": "camera_1", "data_type": "video/x-motion-jpeg"},
                {"id": "camera_2", "data_type": "video/x-motion-jpeg"},
                {"id": "camera_3", "data_type": "video/x-motion-jpeg"},
                {"id": "camera_4", "data_type": "video/x-motion-jpeg"},
            ]
            return JSONResponse(content=sensors)

        # -- Teleop controls ------------------------------------------------

        @app.post("/teleop/start")
        async def teleop_start(request: Request):
            """Launch the teleop subprocess stack.

            Body (all fields optional):
            {
                "record_mode": false,
                "env": {
                    "ROBOT_NAME": "...",
                    "ROBOT_CONFIG": "vega_1u_gripper",
                    "ZENOH_CONFIG": "/path/to/zenoh.json"
                }
            }
            """
            if self.teleop_lifecycle not in {"stopped", "failed"}:
                raise HTTPException(status_code=409, detail="Teleop already running")
            if await state_manager.is_estop():
                raise HTTPException(
                    status_code=403, detail="Cannot start: E-Stop is active"
                )
            # Mutual exclusion: teleop and replay never run together. If a
            # replay stack is up (or finished procs linger), kill it cleanly
            # before launching teleop.
            if self._replay_running or self._procs:
                self._stop_teleop_procs()
            try:
                body = await request.json()
            except Exception:
                body = {}
            self.record_mode = bool(body.get("record_mode", False))
            # Recorder format — "mcap" (default) or "mdp". Only consulted
            # when record_mode is True.
            fmt = str(body.get("recording_format", "mcap")).lower()
            self.recorder_format = fmt if fmt in ("mcap", "mdp") else "mcap"
            # Leader stack selection. "exoskeleton" (default) launches the
            # historical arm_reader+joycon_reader+command_processor chain.
            # "vr" launches paddle_leader → robot_controller directly (paddle
            # bypasses command_processor). "vr_sim" launches paddle_leader in
            # visualize-only mode with no follower.
            leader_mode = str(body.get("leader_mode", "exoskeleton")).lower()
            if leader_mode not in ("exoskeleton", "vr", "vr_sim"):
                leader_mode = "exoskeleton"
            self.leader_mode = leader_mode
            # vr_sim has no robot in the loop — recording is meaningless.
            if leader_mode == "vr_sim":
                self.record_mode = False
            user_env: Dict[str, str] = {
                k: str(v) for k, v in (body.get("env") or {}).items() if v
            }
            self.teleop_lifecycle = "starting"
            self._lifecycle_task = asyncio.create_task(
                self._start_teleop_background(user_env)
            )
            return JSONResponse(
                content=await build_state_dict(),
                status_code=202,
            )

        @app.post("/teleop/stop")
        async def teleop_stop():
            """Stop the teleop subprocess stack.

            The recorder subprocess dies with the rest on SIGTERM; its stdout
            EOF branch in _tail_proc will flip is_recording/recording_status
            if we were mid-episode, so nothing special to do here.
            """
            if self.teleop_lifecycle == "starting":
                raise HTTPException(
                    status_code=409,
                    detail="Teleop is still starting; wait for running or failed state",
                )
            if self.teleop_lifecycle == "stopping":
                return JSONResponse(
                    content=await build_state_dict(),
                    status_code=202,
                )
            if self.teleop_lifecycle == "stopped":
                raise HTTPException(status_code=409, detail="Teleop is not running")
            self.teleop_lifecycle = "stopping"
            self._lifecycle_task = asyncio.create_task(
                self._stop_teleop_background()
            )
            return JSONResponse(
                content=await build_state_dict(),
                status_code=202,
            )

        # -- Replay controls -------------------------------------------------
        #
        # Replay drives the robot from a recorded episode: robot_controller
        # (follower) + replay_record (publisher of recorded robot_commands and
        # camera JPEGs). Mutually exclusive with teleop — each start endpoint
        # kills the other stack's processes first.

        @app.get("/replay/episodes")
        async def replay_episodes():
            """List recorded episodes under the recorder save_dir, newest first."""
            return JSONResponse(content=self._list_episodes())

        @app.post("/replay/validate")
        async def replay_validate(request: Request):
            """Validate a user-supplied episode path.

            Body: {"path": "/path/to/episode"}
            Returns episode metadata (format, num_frames, rate_hz, cameras).
            """
            try:
                body = await request.json()
            except Exception:
                body = {}
            path = str(body.get("path", "")).strip()
            if not path:
                raise HTTPException(status_code=400, detail="Missing 'path'")
            try:
                info = inspect_episode(path)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Not a valid episode: {exc}"
                )
            return JSONResponse(content=info)

        @app.post("/replay/start")
        async def replay_start(request: Request):
            """Launch the replay stack for an episode.

            Body: {"path": "/path/to/episode", "env": {...}}  (env optional,
            same keys as /teleop/start).

            Cleanly kills any running teleop/replay subprocesses first, then
            spawns robot_controller + replay_record. The robot does NOT move
            until POST /replay/begin releases the start gate.
            """
            if await state_manager.is_estop():
                raise HTTPException(
                    status_code=403, detail="Cannot start: E-Stop is active"
                )
            try:
                body = await request.json()
            except Exception:
                body = {}
            path = str(body.get("path", "")).strip()
            if not path:
                raise HTTPException(status_code=400, detail="Missing 'path'")
            try:
                info = inspect_episode(path)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Not a valid episode: {exc}"
                )
            user_env = {
                k: str(v) for k, v in (body.get("env") or {}).items() if v
            }
            # Mutual exclusion: kill everything (teleop or a previous replay).
            if self._teleop_running or self._replay_running or self._procs:
                self._stop_teleop_procs()
            self._start_replay_procs(info, user_env)
            return JSONResponse(content=await build_state_dict())

        @app.post("/replay/begin")
        async def replay_begin():
            """Release the replay start gate — the robot starts moving.

            Writes a newline to replay_record's stdin, firing its
            "Press Enter to start" gate.
            """
            with self._replay_lock:
                replay_state = self._replay_status.get("state")
            proc = self._replay_proc
            if not self._replay_running or proc is None or proc.stdin is None:
                raise HTTPException(status_code=409, detail="Replay is not running")
            if replay_state != "waiting":
                raise HTTPException(
                    status_code=409,
                    detail=f"Replay not waiting for begin (state={replay_state or 'unknown'})",
                )
            try:
                proc.stdin.write("\n")
                proc.stdin.flush()
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to signal replay: {exc}"
                )
            return JSONResponse(content=await build_state_dict())

        @app.post("/replay/stop")
        async def replay_stop():
            """Stop the replay subprocess stack.

            Mirrors /teleop/stop's estop reset so the user can start again
            cleanly afterwards.
            """
            self._stop_teleop_procs()
            self._gui_estop = False
            self._cmd_estop = False
            self._cmd_pending_realign = False
            await state_manager.clear_estop()
            if self.robot and hasattr(self.robot, "estop"):
                try:
                    self.robot.estop.deactivate()
                except Exception as exc:
                    logger.warning(
                        f"Hardware estop release during /replay/stop failed: {exc}"
                    )
            return JSONResponse(content=await build_state_dict())

        # -- Doctor actions (manual robot recovery from the UI) --
        #
        # Each endpoint builds a fresh Robot() inside a context manager that
        # overlays the user-submitted env (ROBOT_NAME / ROBOT_CONFIG /
        # ZENOH_CONFIG from the last /teleop/start) onto os.environ. This
        # mirrors the semantics of running clear_error.py / open_close_hand.py
        # as CLI scripts with those env vars set, and ensures the Doctor
        # actions target the robot variant the user actually configured in
        # the launch form — not whatever the backend process was started with.

        _DOCTOR_CLEAR_COMPONENTS = [
            "left_arm", "right_arm", "head", "chassis", "left_hand", "right_hand",
        ]
        _DOCTOR_ENV_KEYS = ("ROBOT_NAME", "ROBOT_CONFIG", "ZENOH_CONFIG")

        @contextmanager
        def _doctor_env_scope():
            """Overlay self._teleop_env onto os.environ for the block."""
            previous = {k: os.environ.get(k) for k in _DOCTOR_ENV_KEYS}
            for k in _DOCTOR_ENV_KEYS:
                v = (self._teleop_env or {}).get(k)
                if v:
                    os.environ[k] = v
            try:
                yield
            finally:
                for k, v in previous.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        @app.post("/doctor/clear_error")
        async def doctor_clear_error():
            """Clear error state on all robot components."""
            self._require_doctor_exclusive()
            failed: list[str] = []

            def _run() -> None:
                with _doctor_env_scope():
                    with _create_doctor_robot() as bot:
                        for component in _DOCTOR_CLEAR_COMPONENTS:
                            try:
                                bot.clear_error(component)
                            except Exception as exc:
                                failed.append(f"{component}: {exc}")

            try:
                async with self._doctor_lock:
                    await asyncio.to_thread(_run)
            except Exception as exc:
                logger.exception("doctor/clear_error failed")
                raise HTTPException(status_code=500, detail=str(exc))
            return JSONResponse(
                content={"ok": not failed, "failed": failed}
            )

        @app.post("/doctor/init_arm_safe")
        async def doctor_init_arm_safe():
            """Move both arms to the zero pose via collision-aware planning."""
            self._require_doctor_exclusive()
            result: Dict[str, Any] = {}

            def _run() -> None:
                from omniteleop.app.backend.backend_utils.arm_safe_initializer import (
                    ArmSafeInitializer,
                )

                with _doctor_env_scope():
                    with _create_doctor_robot() as bot:
                        _require_arm_brakes_engaged(
                            _read_arm_brake_status(bot)
                        )
                        arms = {
                            name: getattr(bot, name)
                            for name in _ARM_NAMES
                            if getattr(bot, name, None) is not None
                        }
                        starts: Dict[str, np.ndarray] = {}
                        for name, arm in arms.items():
                            state = arm.get_state()  # noqa: SLF001
                            position = np.asarray(state.get("pos", []), dtype=float)
                            limits = arm.joint_pos_limit
                            if (
                                position.shape != (7,)
                                or not bool(np.isfinite(position).all())
                                or limits is None
                                or limits.shape != (7, 2)
                            ):
                                raise RuntimeError(
                                    f"{name} has invalid position or limit data"
                                )
                            outside = np.flatnonzero(
                                (position < limits[:, 0] - 1e-3)
                                | (position > limits[:, 1] + 1e-3)
                            )
                            if outside.size:
                                raise RuntimeError(
                                    f"{name} remains outside joint limits at "
                                    f"indices {outside.tolist()}"
                                )
                            starts[name] = position.copy()
                            if has_active_joint_error(state.get("error")):
                                bot.clear_error(name)
                                clear_deadline = time.monotonic() + 2.0
                                while time.monotonic() < clear_deadline:
                                    if not has_active_joint_error(
                                        arm.get_state().get("error")  # noqa: SLF001
                                    ):
                                        break
                                    time.sleep(0.02)
                                else:
                                    raise RuntimeError(
                                        f"{name} error remained latched after clear"
                                    )

                        bot.estop.deactivate()
                        position_mode_ready = False
                        try:
                            for name, arm in arms.items():
                                response = arm.set_modes(["position"] * 7)
                                if (
                                    not isinstance(response, dict)
                                    or response.get("success") is not True
                                ):
                                    raise RuntimeError(
                                        f"{name} position mode failed: {response}"
                                    )
                            position_mode_ready = True

                            hold_deadline = time.monotonic() + 0.5
                            while time.monotonic() < hold_deadline:
                                for name, arm in arms.items():
                                    arm.set_joint_pos(starts[name])
                                time.sleep(0.01)
                            hold_drift = {
                                name: float(
                                    np.max(
                                        np.abs(
                                            np.asarray(
                                                arm.get_joint_pos(), dtype=float
                                            )
                                            - starts[name]
                                        )
                                    )
                                )
                                for name, arm in arms.items()
                            }
                            if any(value > 0.02 for value in hold_drift.values()):
                                raise RuntimeError(
                                    f"Measured-pose hold failed: {hold_drift}"
                                )

                            initializer = ArmSafeInitializer(
                                visualize=False,
                                bot=bot,
                            )
                            initializer.run(
                                target="zero",
                                shutdown_after_execution=False,
                            )
                            goal = initializer._create_goal_configuration("zero")
                            final_error = {}
                            final_position = {}
                            final_targets = {}
                            for name, arm in arms.items():
                                actual = np.asarray(
                                    arm.get_joint_pos(), dtype=float
                                )
                                target = np.asarray(
                                    [goal[joint] for joint in arm.joint_name],
                                    dtype=float,
                                )
                                final_position[name] = actual.tolist()
                                final_targets[name] = target
                                final_error[name] = float(
                                    np.max(np.abs(actual - target))
                                )
                            if any(value > 0.05 for value in final_error.values()):
                                raise RuntimeError(
                                    f"Safe init final pose verification failed: "
                                    f"{final_error}"
                                )

                            # Doctor init is a transition into the Teleop start pose,
                            # not a terminal shutdown operation. Keep position mode
                            # active and leave the final target latched for the next
                            # control owner instead of disabling the arms here.
                            final_hold_deadline = time.monotonic() + 0.5
                            while time.monotonic() < final_hold_deadline:
                                for name, arm in arms.items():
                                    arm.set_joint_pos(final_targets[name])
                                time.sleep(0.01)
                            result.update(
                                {
                                    "ok": True,
                                    "start_position": {
                                        name: value.tolist()
                                        for name, value in starts.items()
                                    },
                                    "final_position": final_position,
                                    "hold_drift_rad": hold_drift,
                                    "max_final_error_rad": final_error,
                                    "terminal_mode": "position",
                                }
                            )
                        except Exception:
                            # Never transition to disable here: hardware tests
                            # showed that both arms can fall after disable. If
                            # position mode was established, briefly latch the
                            # last measured pose before the client disconnects.
                            if position_mode_ready:
                                failure_targets = {
                                    name: np.asarray(
                                        arm.get_joint_pos(), dtype=float
                                    ).copy()
                                    for name, arm in arms.items()
                                }
                                failure_hold_deadline = time.monotonic() + 0.5
                                while time.monotonic() < failure_hold_deadline:
                                    for name, arm in arms.items():
                                        arm.set_joint_pos(failure_targets[name])
                                    time.sleep(0.01)
                            raise

            try:
                async with self._doctor_lock:
                    await asyncio.to_thread(_run)
            except ValueError as exc:
                # Precondition violations (brake release active, joints out of
                # limits) — same 409 semantics as doctor/arm_brakes.
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("doctor/init_arm_safe failed")
                raise HTTPException(status_code=500, detail=str(exc))
            return JSONResponse(content=result)

        @app.post("/doctor/arm_brakes")
        async def doctor_arm_brakes(request: Request):
            """Read both arms, release one selected joint, or engage all brakes."""
            self._require_doctor_exclusive()
            try:
                body = await request.json()
            except Exception:
                body = {}
            action = str(body.get("action", "status")).lower()
            arm_name = body.get("arm")
            raw_joint_index = body.get("joint_index")
            joint_index = (
                raw_joint_index
                if isinstance(raw_joint_index, int)
                and not isinstance(raw_joint_index, bool)
                else None
            )
            confirmed = body.get("confirmed") is True
            result: Dict[str, Any] = {}

            def _run() -> None:
                nonlocal result
                with _doctor_env_scope():
                    with _create_doctor_robot() as bot:
                        result = _control_arm_brake(
                            bot,
                            action,
                            arm_name=arm_name,
                            joint_index=joint_index,
                            confirmed=confirmed,
                        )

            try:
                async with self._doctor_lock:
                    await asyncio.to_thread(_run)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception(f"doctor/arm_brakes ({action}) failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return JSONResponse(content=result)

        @app.post("/doctor/hand")
        async def doctor_hand(request: Request):
            """Open or close both hands."""
            self._require_doctor_exclusive()
            body = await request.json()
            action = body.get("action")
            if action not in ("open", "close"):
                raise HTTPException(
                    status_code=400,
                    detail="action must be 'open' or 'close'",
                )

            def _run() -> None:
                with _doctor_env_scope():
                    with _create_doctor_robot() as bot:
                        _require_arm_brakes_engaged(
                            _read_arm_brake_status(bot)
                        )
                        if action == "open":
                            bot.left_hand.open_hand()
                            bot.right_hand.open_hand()
                        else:
                            bot.left_hand.close_hand()
                            bot.right_hand.close_hand()

            try:
                async with self._doctor_lock:
                    await asyncio.to_thread(_run)
            except ValueError as exc:
                # Brake release active — precondition violation, same 409
                # semantics as doctor/arm_brakes.
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception(f"doctor/hand ({action}) failed")
                raise HTTPException(status_code=500, detail=str(exc))
            return JSONResponse(content={"ok": True, "action": action})

        # -- Recording controls (only active when record_mode was set at /teleop/start) --

        @app.post("/record/start")
        async def record_start(request: Request):
            if not self.record_mode:
                raise HTTPException(
                    status_code=403,
                    detail="Record mode not enabled — set record_mode=true in /teleop/start",
                )
            if await state_manager.is_estop():
                raise HTTPException(
                    status_code=403, detail="Robot is in emergency stop"
                )
            if self.is_recording:
                raise HTTPException(status_code=409, detail="Already recording")
            if self._recorder_ctrl_pub is None:
                raise HTTPException(
                    status_code=503,
                    detail="Recorder control publisher not ready",
                )
            recorder_name = f"{self.recorder_format}_recorder"
            if not self.recorder_ready or not self._is_process_running(recorder_name):
                raise HTTPException(
                    status_code=503,
                    detail="Recorder is not ready; check recorder and wrist camera logs",
                )
            try:
                body = await request.json()
            except Exception:
                body = {}
            metadata = body.get("metadata") or {}
            self._set_pending_record_metadata(metadata)
            self._recorder_ctrl_pub.publish(
                {"command": "start", "metadata": metadata}
            )
            # Optimistic flip — without this the UI button visibly reverts to
            # "Start" for the 200-500 ms between click and the recorder's
            # "RECORDING STARTED" stdout line. The stdout tail remains
            # authoritative: if the recorder crashes/never confirms, the
            # finally-block in _tail_proc will flip this back to
            # "crashed"/"stopped".
            self.is_recording = True
            self.recording_status = "recording"
            self.episode_dir_name = None  # set by the Directory: log line
            return JSONResponse(content=await build_state_dict())

        @app.post("/record/metadata")
        async def record_metadata(request: Request):
            # This is allowed before record mode is enabled so the operator can
            # pre-fill task metadata before launching teleop.
            try:
                body = await request.json()
            except Exception:
                body = {}
            metadata = body.get("metadata") or {}
            self._set_pending_record_metadata(metadata)
            # Push the pending metadata to the recorder so a later JoyCon
            # L+R start automatically carries these fields.
            if self._recorder_ctrl_pub is not None:
                self._recorder_ctrl_pub.publish(
                    {"command": "set_metadata", "metadata": metadata}
                )
            return JSONResponse(content={"ok": True, "metadata": metadata})

        @app.post("/record/stop")
        async def record_stop(request: Request):
            if not self.record_mode:
                raise HTTPException(status_code=403, detail="Record mode not enabled")
            if not self.is_recording:
                raise HTTPException(status_code=409, detail="Not currently recording")
            if self._recorder_ctrl_pub is None:
                raise HTTPException(
                    status_code=503,
                    detail="Recorder control publisher not ready",
                )
            try:
                body = await request.json()
                is_success = body.get("is_success", True)
            except Exception:
                is_success = True
            # "stop" saves, "discard" drops the episode. Mirrors pedal 'b'/'c'.
            command = "stop" if is_success else "discard"
            self._recorder_ctrl_pub.publish({"command": command})
            # Optimistic flip (same rationale as /record/start). The stdout
            # tail will later confirm via "EPISODE SAVED" / "EPISODE DISCARDED".
            self.is_recording = False
            self.recording_status = "saved" if is_success else "discarded"
            return JSONResponse(
                content={
                    "timestamp_posix": time.time(),
                    "is_success": is_success,
                    "episode_id": self.episode_dir_name or "",
                },
                status_code=201,
            )

        # -- E-Stop ----------------------------------------------------------

        @app.post("/robots/estop")
        async def robots_estop():
            self._gui_estop = True
            await self._sync_effective_estop()
            await video_publisher.stop_all_streams()
            # Recorder stops itself on estop via its robot_commands subscriber
            # (auto_stop_on_estop in BaseRecorder) — no need to do it here.
            try:
                await self._set_software_estop(True)
            except Exception as exc:
                logger.error(f"Hardware estop failed: {exc}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Hardware E-stop activation failed: {exc}",
                ) from exc
            return JSONResponse(content=await build_state_dict())

        @app.delete("/robots/estop")
        async def robots_estop_clear():
            if not await state_manager.is_estop():
                raise HTTPException(status_code=409, detail="Not in emergency stop")
            self._gui_estop = False
            await self._sync_effective_estop()
            if not self._cmd_estop_is_live():
                # Nothing is producing commands any more (teleop exited, or the
                # stack died). Drop the latched request so it cannot block this
                # clear or the next session's.
                self._cmd_estop = False
                self._cmd_pending_realign = False
                await self._sync_effective_estop()
            if self._cmd_estop:
                # Two very different reasons land here. Say which one, or the
                # operator has no way to know what action would clear it.
                detail = (
                    (
                        "Exoskeleton drifted away from the robot while paused. "
                        "Match the leader arms back to the robot pose (see the "
                        "alignment guide) — E-Stop clears automatically once "
                        "they are within range."
                    )
                    if self._cmd_pending_realign
                    else (
                        "Command input still requests E-stop; "
                        "hardware E-stop remains active"
                    )
                )
                raise HTTPException(status_code=409, detail=detail)
            try:
                await self._set_software_estop(False)
            except Exception as exc:
                self._gui_estop = True
                await self._sync_effective_estop()
                raise HTTPException(
                    status_code=503,
                    detail=f"Hardware E-stop clear failed: {exc}",
                ) from exc
            return JSONResponse(content=await build_state_dict())

        # -- WebSocket endpoints --------------------------------------------

        @app.websocket("/ws/state")
        async def ws_state(websocket: WebSocket):
            await state_publisher.start_stream(websocket)

        @app.websocket("/ws/sensors/stream/{camera_id}")
        async def ws_camera(websocket: WebSocket, camera_id: str):
            # Camera preview is a pure read-only stream (never commands the
            # robot), so it stays available during E-Stop — the ALIGN phase
            # needs live vision of the robot while E-Stop is still latched.
            await video_publisher.start_stream(websocket, camera_id)

        @app.websocket("/ws/logs")
        async def ws_logs(websocket: WebSocket):
            """Stream per-process log lines as JSON: {"source": "omni-arm", "line": "..."}"""
            await websocket.accept()
            import json as _json

            q: asyncio.Queue = asyncio.Queue(maxsize=500)
            with self._log_lock:
                # Send buffered history first
                history = []
                for name, lines in self._proc_logs.items():
                    history.extend(lines[-200:])
                self._log_subscribers.append(q)
            try:
                for entry in history:
                    await websocket.send_text(_json.dumps(entry))
                while True:
                    entry = await q.get()
                    await websocket.send_text(_json.dumps(entry))
            except Exception:
                pass
            finally:
                with self._log_lock:
                    try:
                        self._log_subscribers.remove(q)
                    except ValueError:
                        pass

        # -- Serve frontend from the sibling frontend/ directory -------------
        _frontend_dir = Path(__file__).parents[2] / "app" / "frontend"
        if _frontend_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(_frontend_dir / "assets")),
                name="assets",
            )

            @app.get("/")
            async def serve_index():
                _index = _frontend_dir / "index.html"
                if _index.exists():
                    # The UI is a single local HTML file and changes alongside
                    # the backend. Prevent a browser 304/cache hit from hiding
                    # newly added safety guidance after a restart.
                    return FileResponse(
                        str(_index),
                        headers={"Cache-Control": "no-store"},
                    )
                raise HTTPException(status_code=404, detail="Frontend not found")

    # =========================================================================
    # Frontend connection lifecycle
    # =========================================================================

    async def on_frontend_connected(self):
        await asyncio.sleep(1)
        self.frontend_ready = True
        logger.info("Frontend ready")

    def on_frontend_disconnected(self):
        self.frontend_ready = False
        logger.info("Frontend disconnected")

    # =========================================================================
    # Teleop subprocess management
    # =========================================================================

    def _require_doctor_exclusive(self) -> None:
        if (
            self.teleop_lifecycle != "stopped"
            or self._teleop_running
            or any(proc.poll() is None for proc in self._procs)
        ):
            raise HTTPException(
                status_code=409,
                detail="Doctor actions require Teleop to be fully stopped",
            )
        if self._doctor_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="Another Doctor action is already running",
            )

    async def _start_teleop_background(
        self, user_env: Dict[str, str]
    ) -> None:
        try:
            await asyncio.to_thread(self._start_teleop_procs, user_env)
            self.teleop_lifecycle = "running"
        except Exception as exc:
            logger.exception(f"Teleop startup failed: {exc}")
            self.teleop_lifecycle = "failed"
            self._teleop_running = False
            if self._procs:
                await asyncio.to_thread(self._stop_teleop_procs)

    async def _stop_teleop_background(self) -> None:
        try:
            await asyncio.to_thread(self._stop_teleop_procs)
        except Exception as exc:
            logger.exception(f"Teleop shutdown failed: {exc}")
            self.teleop_lifecycle = "failed"
            return

        # Stopping is a terminal safety transition. Keep software E-Stop
        # latched until the operator explicitly clears it.
        self._gui_estop = True
        self._cmd_estop = False
        self._cmd_pending_realign = False
        await state_manager.estop()
        try:
            await self._set_software_estop(True)
        except Exception as exc:
            logger.warning(f"Hardware E-Stop activation after stop failed: {exc}")
        self.teleop_lifecycle = "stopped"

    async def _set_software_estop(self, enabled: bool) -> None:
        if (
            self.robot is None
            or not hasattr(self.robot, "estop")
            or self._estop_control is None
        ):
            raise RuntimeError("E-stop control is not initialized")
        await asyncio.to_thread(
            self._estop_control.set_enabled,
            enabled,
            self.robot.estop,
        )

    def _build_proc_env(self, user_env: Dict[str, str]) -> Dict[str, str]:
        """Overlay user env overrides onto os.environ for subprocess launch.

        Resolves ROBOT_NAME / ROBOT_CONFIG / ZENOH_CONFIG the same way for
        teleop and replay stacks. Mutates ``user_env`` in place (resolved
        values) and stores it as ``self._teleop_env`` for state reporting.

        Blank fields fall back to the last successful launch's resolved env
        (``self._last_launch_env``) before defaults kick in, so a replay
        started without re-entering the robot/zenoh form reuses the same
        robot connection the last teleop/replay used — avoiding a mismatched
        auto-detected cert that can't reach the robot.
        """
        env = os.environ.copy()
        prev = self._last_launch_env

        # --- Resolve ROBOT_NAME ---
        if not user_env.get("ROBOT_NAME"):
            if prev.get("ROBOT_NAME"):
                user_env["ROBOT_NAME"] = prev["ROBOT_NAME"]
            else:
                # Fall back to whatever is already in the process environment
                user_env.pop("ROBOT_NAME", None)  # let os.environ value pass through

        # --- Resolve ROBOT_CONFIG ---
        if not user_env.get("ROBOT_CONFIG"):
            user_env["ROBOT_CONFIG"] = prev.get("ROBOT_CONFIG") or "vega_1u_gripper"

        # --- Resolve ZENOH_CONFIG ---
        zenoh_val = user_env.get("ZENOH_CONFIG", "").strip()
        if not zenoh_val and prev.get("ZENOH_CONFIG"):
            # Reuse the last launch's already-resolved absolute cert path.
            # Write it back so it propagates into the resolved env below (it
            # bypasses the resolve branches since it's already absolute).
            zenoh_val = prev["ZENOH_CONFIG"]
            user_env["ZENOH_CONFIG"] = zenoh_val
        zenoh_base = Path.home() / ".dexmate" / "comm" / "zenoh"
        if zenoh_val and not zenoh_val.startswith("/"):
            # Treat as a certificate name — resolve to a full path by trying
            # both supported formats:
            #   1. <name>/zenoh_peer_config.json5  (directory bundle)
            #   2. <name>.dzcfg                    (flat file)
            resolved = _resolve_zenoh_cert(zenoh_base, zenoh_val)
            if resolved:
                user_env["ZENOH_CONFIG"] = str(resolved)
            else:
                logger.warning(
                    f"ZENOH_CONFIG '{zenoh_val}' not found under {zenoh_base} "
                    f"(tried {zenoh_val}/zenoh_peer_config.json5 and {zenoh_val}.dzcfg)"
                )
        elif not zenoh_val:
            # Auto-detect: pick the first available cert (directory or .dzcfg)
            try:
                resolved = _autodetect_zenoh_cert(zenoh_base)
                if resolved:
                    user_env["ZENOH_CONFIG"] = str(resolved)
                    logger.info(f"Auto-detected ZENOH_CONFIG: {resolved}")
                else:
                    logger.warning(f"No zenoh certs found under {zenoh_base}")
            except FileNotFoundError:
                logger.warning(f"Zenoh certs directory not found: {zenoh_base}")

        env.update(user_env)
        self._teleop_env = user_env  # store so we can report it in state
        # Persist the resolved env across stop() so the next launch (teleop or
        # replay) can reuse blank fields.
        self._last_launch_env = dict(user_env)
        return env

    def _spawn_proc(
        self,
        cmd: List[str],
        env: Dict[str, str],
        use_stdin: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Spawn one subprocess with log tailing; returns the Popen or None.

        Derives a friendly name for logs/tabs. Two cases:
          python /path/to/arm_reader.py ...             → "arm_reader"
          python -m omniteleop.record.mcap_recorder ... → "mcap_recorder"
        """
        if cmd[1] == "-m" and len(cmd) > 2:
            name = cmd[2].rsplit(".", 1)[-1]
        else:
            name = Path(cmd[1]).stem
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.PIPE if use_stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError:
            logger.error(f"Command not found: {name} — is omniteleop installed?")
            return None
        self._procs.append(proc)
        self._proc_names.append(name)
        with self._log_lock:
            self._proc_logs[name] = []
        threading.Thread(
            target=self._tail_proc,
            args=(proc, name),
            daemon=True,
            name=f"log-{name}",
        ).start()
        logger.info(f"Started: {' '.join(cmd)}  (pid={proc.pid})")
        return proc

    def _start_teleop_procs(self, user_env: Dict[str, str]):
        """Spawn all teleop subprocesses with the given env overrides."""
        if self._teleop_running:
            return

        # _stop_teleop_procs latches this and never clears it; the clearing
        # lived in a duplicate definition of this method that Python discarded
        # in favour of the later one. Left set, it permanently disables both
        # the crash detection and the clean-exit detection in _tail_proc, so
        # only the first joycon-initiated exit was ever noticed.
        self._stop_in_progress = False
        self._proc_crashed.clear()

        env = self._build_proc_env(user_env)

        # Re-init StateChecker with ROBOT_NAME from env if provided
        robot_name = env.get("ROBOT_NAME", "")
        try:
            if self.state_checker:
                self._state_checker_running = False
                self.state_checker.cleanup()
            self.state_checker = StateChecker(
                robot_name,
                rpi_mode=self.rpi_mode,
                leader_mode=self.leader_mode,
            )
            self._state_checker_running = True
            self._state_checker_thread = threading.Thread(
                target=self._state_checker_loop, daemon=True, name="StateCheckerLoop"
            )
            self._state_checker_thread.start()
            logger.info(f"StateChecker initialised (robot={robot_name!r})")
        except Exception as exc:
            logger.warning(f"StateChecker init failed: {exc}")

        _src = Path(__file__).parents[2]
        ns_args = ["--namespace", self.namespace] if self.namespace else []
        debug_args = ["--debug"] if self.debug else []
        commands = self._build_teleop_commands(_src, ns_args, debug_args)

        spawned_any = False
        for entry in commands:
            # Each entry is (cmd_list, pre_delay_s). The delay lets earlier
            # processes settle before launching a dependent one — used in VR
            # mode so robot_controller publishes robot_joints before
            # paddle_leader's _bootstrap() starts waiting for it.
            cmd, pre_delay = entry
            if pre_delay > 0:
                time.sleep(pre_delay)
            if self._spawn_proc(cmd, env) is not None:
                spawned_any = True

        if not spawned_any:
            raise RuntimeError(
                "Failed to start any teleop process — is omniteleop installed?"
            )
        self._teleop_running = True

        # Retire the previous session's endpoints before re-declaring them.
        # These were created on every start and never released, so each
        # start/stop cycle left another live subscriber on the shared node.
        # (The replay path avoids this with its own _replay_sub_topics guard.)
        self._retire_teleop_endpoints()

        # Subscribe to robot joint feedback published by robot_controller
        joints_topic = self.config.get_topic("robot_joints")
        self._teleop_subs.append(
            self._node.create_subscriber(
                joints_topic,
                callback=self._on_joints_received,
                decoder=DictDataCodec.decode,
            )
        )

        # Subscribe to robot commands — always (estop sync) + record mode (action tracking)
        commands_topic = self.config.get_topic("robot_commands")
        self._teleop_subs.append(
            self._node.create_subscriber(
                commands_topic,
                callback=self._on_command_received,
                decoder=DictDataCodec.decode,
            )
        )

        # Publisher used by /record/start, /record/stop to drive the recorder
        # subprocess via the shared recorder_control topic.
        if self.record_mode and self._recorder_ctrl_pub is None:
            control_topic = self.config.get_topic("recorder_control")
            self._recorder_ctrl_pub = self._node.create_publisher(
                control_topic,
                encoder=DictDataCodec.encode,
            )

        mode_str = " [record mode]" if self.record_mode else ""
        logger.success(f"Teleop stack started (leader={self.leader_mode}){mode_str}")

    def _build_teleop_commands(
        self,
        src: Path,
        ns_args: List[str],
        debug_args: List[str],
    ) -> List[Tuple[List[str], float]]:
        """Return the ordered subprocess commands for the teleop stack.

        Each entry is ``(cmd, pre_delay_s)`` — the caller sleeps for
        ``pre_delay_s`` before spawning ``cmd``. Used to enforce startup
        ordering when a process depends on a topic another spawns.

        Branches on ``self.leader_mode``:
          - ``exoskeleton``: arm_reader + joycon_reader (leaders) → then
            command_processor + robot_controller (+ optional recorder). No
            inter-process delay needed — leaders don't gate on robot state.
          - ``vr``: robot_controller (follower) → 0.5s sleep → paddle_leader.
            paddle_leader's ``_bootstrap()`` waits for ``robot_joints``;
            launching the follower first avoids a startup race that would
            cause the leader to time out and run against MM defaults
            (clutch-in snap on first VR trigger press).
          - ``vr_sim``: paddle_leader --visualize only. No follower, no
            recorder — the SAPIEN viewer is the only sink.
        """
        unlock_torso_args = ["--unlock-torso"] if self.unlock_torso else []

        camera_cmds: List[List[str]] = []
        wrist_cfg = (
            (self.config.get("recorder", {}) or {}).get("wrist_camera_adapter", {})
            or {}
        )
        wrist_requested = any(
            self.record_components.get(f"{side}_wrist_rgb", False)
            for side in ("left", "right")
        )
        if self.record_mode and wrist_requested and wrist_cfg.get("enabled", False):
            camera_cmds.append([
                sys.executable,
                "-m",
                "omniteleop.record.wrist_camera_adapter",
                *ns_args,
                *debug_args,
            ])

        # vr_sim has no robot in the loop; everything else needs the follower.
        if self.leader_mode == "vr_sim":
            return [(
                [
                    sys.executable,
                    str(src / "leader" / "paddle_leader.py"),
                    *ns_args,
                    *debug_args,
                    *unlock_torso_args,
                    "--visualize",
                ],
                0.0,
            )]

        lock_hand_args = (
            ["--lock-hand-collision-avoidance"]
            if self.lock_hand_collision_avoidance
            else []
        )

        # robot_controller is shared across exoskeleton and vr modes.
        # command_processor is only used by exoskeleton — paddle_leader
        # publishes robot_commands directly.
        follower_cmds: List[List[str]] = []
        if self.leader_mode == "exoskeleton":
            follower_cmds.append([
                sys.executable,
                str(src / "follower" / "command_processor.py"),
                *ns_args,
                *debug_args,
                *lock_hand_args,
            ])
        robot_controller_cmd = [
            sys.executable,
            str(src / "follower" / "robot_controller.py"),
            *ns_args,
            *debug_args,
            "--interpolation-method",
            "none",
            "--no-arm-filter",
            # robot_controller homes to init_pos on startup; the joint-feedback
            # publisher starts only after homing finishes, so the ALIGN gate
            # (robot data comes from that topic) naturally waits for the robot
            # to reach home before the operator aligns the leader to it.
        ]
        # Both leader stacks publish ready-to-send joint targets. Keep the
        # follower's real-time path explicit and identical in exoskeleton and
        # VR modes: no interpolation and no extra arm filtering.
        follower_cmds.append(robot_controller_cmd)

        recorder_cmds: List[List[str]] = []
        if self.record_mode:
            # Pick the recorder module based on the user's format choice.
            # Both inherit BaseRecorder, listen on the recorder_control topic,
            # and emit the same stdout markers that _tail_proc parses.
            module = (
                "omniteleop.record.mdp_recorder"
                if self.recorder_format == "mdp"
                else "omniteleop.record.mcap_recorder"
            )
            recorder_cmds.append([
                sys.executable,
                "-m",
                module,
                *ns_args,
                *debug_args,
                "--record-mode",
                "joycon",
            ])

        # No leader scripts in rpi_mode — exo lives on a remote machine.
        if self.rpi_mode:
            logger.info("RPi mode: skipping leader scripts")
            return [(c, 0.0) for c in camera_cmds + follower_cmds + recorder_cmds]

        if self.leader_mode == "vr":
            # Follower first so robot_joints is publishing before
            # paddle_leader's _bootstrap() starts polling. Robot()/Zenoh
            # initialization can take ~10s and concurrent Robot() queries can
            # time out, so start the VR leader and recorder serially.
            rate_args = ["--publish-rate", "40"]
            leader_cmd = [
                sys.executable,
                str(src / "leader" / "paddle_leader.py"),
                *ns_args,
                *debug_args,
                *unlock_torso_args,
                *rate_args,
            ]
            entries: List[Tuple[List[str], float]] = []
            entries.extend((c, 0.0) for c in camera_cmds)
            entries.extend((c, 0.0) for c in follower_cmds)
            entries.append((leader_cmd, 12.0))
            entries.extend((c, 30.0) for c in recorder_cmds)
            return entries

        # exoskeleton (default)
        leader_cmds = [
            [
                sys.executable,
                str(src / "leader" / "arm_reader.py"),
                *ns_args,
                *debug_args,
            ],
            [
                sys.executable,
                str(src / "leader" / "joycon_reader.py"),
                *ns_args,
                *debug_args,
            ],
        ]
        return [
            (c, 0.0)
            for c in camera_cmds + leader_cmds + follower_cmds + recorder_cmds
        ]

    def _is_process_running(self, name: str) -> bool:
        return any(
            proc_name == name and proc.poll() is None
            for proc, proc_name in zip(self._procs, self._proc_names)
        )

    # =========================================================================
    # Replay stack
    # =========================================================================

    def _list_episodes(self) -> List[Dict[str, Any]]:
        """Scan the recorder save_dir for recorded episodes, newest first.

        Episodes live either directly under save_dir or nested in
        MM-DD-YYYY day folders (both recorders group by calendar day; legacy
        flat episodes still show up). Format detection is by marker file
        (transitions.pkl → mdp, episode.mcap → mcap).
        """
        recorder_cfg = self.config.get("recorder", {}) or {}
        save_dir = Path(recorder_cfg.get("save_dir") or "recordings").expanduser()
        if not save_dir.is_absolute():
            # Same resolution rule as BaseRecorder: relative → under $HOME.
            save_dir = Path.home() / save_dir

        episodes: List[Dict[str, Any]] = []
        if not save_dir.is_dir():
            return episodes
        try:
            children = sorted(save_dir.iterdir())
        except OSError:
            return episodes
        for child in children:
            if not child.is_dir():
                continue
            fmt = detect_format(child)
            if fmt:
                candidates = [(child, fmt, "")]
            else:
                # Day folder — one level of nesting.
                try:
                    candidates = [
                        (sub, sub_fmt, child.name)
                        for sub in sorted(child.iterdir())
                        if sub.is_dir() and (sub_fmt := detect_format(sub))
                    ]
                except OSError:
                    continue
            for d, d_fmt, day in candidates:
                try:
                    mtime = d.stat().st_mtime
                except OSError:
                    mtime = 0.0
                episodes.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "format": d_fmt,
                        "day": day,
                        "mtime": mtime,
                    }
                )
        episodes.sort(key=lambda e: e["mtime"], reverse=True)
        return episodes

    def _on_replay_status(self, data: Dict[str, Any]):
        """Store the latest replay/status message from replay_record."""
        if isinstance(data, dict):
            with self._replay_lock:
                self._replay_status = data

    def _on_replay_frame(self, cam: str, data: bytes):
        """Store the latest recorded-camera JPEG published by replay_record."""
        if data:
            with self._replay_lock:
                self._replay_frames[cam] = bytes(data)

    def _start_replay_procs(
        self, episode_info: Dict[str, Any], user_env: Dict[str, str]
    ):
        """Spawn the replay stack: robot_controller → replay_record.

        replay_record is spawned with stdin=PIPE and blocks at its
        "Press Enter" gate until POST /replay/begin writes a newline — the
        robot does not move before that.
        """
        env = self._build_proc_env(user_env)

        _src = Path(__file__).parents[2]
        ns_args = ["--namespace", self.namespace] if self.namespace else []
        debug_args = ["--debug"] if self.debug else []

        with self._replay_lock:
            self._replay_status = {}
            self._replay_frames = {}
            self._replay_episode = dict(episode_info)

        # Follower first — same rationale as VR mode: give robot_controller's
        # Robot()/Zenoh init a head start before commands can arrive.
        self._spawn_proc(
            [
                sys.executable,
                str(_src / "follower" / "robot_controller.py"),
                *ns_args,
                *debug_args,
            ],
            env,
        )
        time.sleep(0.5)
        self._replay_proc = self._spawn_proc(
            [
                sys.executable,
                "-m",
                "omniteleop.record.replay_record",
                "--episode-path",
                str(episode_info["path"]),
                *ns_args,
                *debug_args,
            ],
            env,
            use_stdin=True,
        )
        self._replay_running = self._replay_proc is not None

        # Subscribe to the replay status + per-camera JPEG topics. Subscribers
        # persist on the shared node, so only subscribe to topics we haven't
        # seen before (repeat replays reuse them; callbacks just overwrite the
        # latest-value stores).
        status_topic = self.config.get_topic("replay_status", "replay/status")
        if status_topic not in self._replay_sub_topics:
            self._node.create_subscriber(
                status_topic,
                callback=self._on_replay_status,
                decoder=DictDataCodec.decode,
            )
            self._replay_sub_topics.add(status_topic)
        for cam in episode_info.get("cameras", []):
            video_topic = f"replay/video/{cam}"
            if video_topic in self._replay_sub_topics:
                continue
            self._node.create_subscriber(
                video_topic,
                callback=(lambda data, _cam=cam: self._on_replay_frame(_cam, data)),
            )
            self._replay_sub_topics.add(video_topic)

        logger.success(
            f"Replay stack started: {episode_info['name']} "
            f"({episode_info['format']}, {episode_info['num_frames']} frames "
            f"@ {episode_info['rate_hz']}Hz)"
        )

    def _retire_teleop_endpoints(self) -> None:
        """Release this session's subscribers/publishers on the shared node.

        The node itself outlives every teleop session, so anything declared on
        it per-session has to be shut down explicitly or it stays live for the
        life of the backend.
        """
        for endpoint in self._teleop_subs:
            try:
                endpoint.shutdown()
            except Exception:  # pylint: disable=broad-except
                pass
        self._teleop_subs.clear()
        if self._recorder_ctrl_pub is not None:
            try:
                self._recorder_ctrl_pub.shutdown()
            except Exception:  # pylint: disable=broad-except
                pass
            self._recorder_ctrl_pub = None

    def _stop_teleop_procs(self):
        """Terminate all teleop subprocesses and their entire process groups.

        robot_controller is stopped last and given a longer grace period: it is
        the only process holding the arms, and on SIGTERM it runs a safe-park
        sequence (hold measured pose, then engage brakes) before exiting. Killing
        it first, or killing it early, abandons the arms in position mode and
        lets the wrist fall past its joint limit.
        """
        import signal

        self._stop_in_progress = True

        processes = list(zip(self._procs, self._proc_names))
        others = [pair for pair in processes if pair[1] != self.ARM_HOLDER_PROC]
        arm_holders = [pair for pair in processes if pair[1] == self.ARM_HOLDER_PROC]

        def _signal(proc, sig) -> None:
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, OSError):
                pass  # already dead

        def _stop_group(group, grace: float, *, warn_for_arm: bool = False) -> None:
            for proc, _name in group:
                _signal(proc, signal.SIGTERM)
            deadline = time.monotonic() + grace
            stragglers = []
            for proc, name in group:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    stragglers.append((proc, name))
            for proc, name in stragglers:
                if warn_for_arm:
                    logger.warning(
                        f"{name} did not exit within {grace}s — the arms may "
                        f"not have been parked; verify joint limits before the "
                        f"next session"
                    )
                _signal(proc, signal.SIGKILL)

        # Stop command producers first. robot_controller keeps publishing its
        # last target until it receives SIGTERM, then performs the brake handoff.
        _stop_group(others, self.PROC_STOP_GRACE_S)
        _stop_group(
            arm_holders,
            self.ARM_HOLDER_STOP_GRACE_S,
            warn_for_arm=True,
        )
        self._procs.clear()
        self._proc_names.clear()
        self._state_checker_running = False
        self.robot_state = "BOOT"
        self._teleop_running = False
        self.record_mode = False
        self.recorder_format = "mcap"
        self.leader_mode = "exoskeleton"
        self._teleop_env = {}
        # Reset recording mirror state — subprocess is gone.
        self.is_recording = False
        self.recording_status = "idle"
        self.recorder_ready = False
        self.episode_dir_name = None
        # Drops the recorder control publisher too, so a stopped stack holds
        # nothing on the shared node.
        self._retire_teleop_endpoints()
        # Reset replay mirror state — replay procs die with the rest.
        self._replay_running = False
        self._replay_proc = None
        with self._replay_lock:
            self._replay_status = {}
            self._replay_frames = {}
            self._replay_episode = {}
        # The command producers are gone, so any E-Stop request they left
        # behind is stale by definition. Reset it here rather than in each
        # caller — the callers that forgot are how a joycon-initiated exit
        # ended up latching E-Stop into the next session.
        self._cmd_estop = False
        self._cmd_pending_realign = False
        self._last_command_at = None
        # Reset align latch so the next teleop session runs the alignment
        # check from scratch.
        self._motion_started_signaled = False
        if self.state_checker:
            self.state_checker.reset_motion_started()
        logger.info("Teleop stack stopped")

    def _tail_proc(self, proc: subprocess.Popen, name: str):
        """Read stdout/stderr of a subprocess and fan-out to log subscribers.

        For the mcap_recorder / mdp_recorder subprocess, also parses the
        recorder's own log markers to mirror recording state, and treats
        stream-EOF as a recording terminator (clean exit or crash).
        """
        MAX_LINES = 500
        is_recorder = name in ("mcap_recorder", "mdp_recorder")
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.rstrip("\n")
                entry = {"source": name, "line": line}
                with self._log_lock:
                    buf = self._proc_logs.get(name)
                    if buf is not None:
                        buf.append(entry)
                        if len(buf) > MAX_LINES:
                            buf.pop(0)
                    subs = list(self._log_subscribers)
                loop = self._log_loop
                if loop and loop.is_running():
                    for q in subs:
                        try:
                            loop.call_soon_threadsafe(q.put_nowait, entry)
                        except Exception:
                            pass
                if is_recorder:
                    self._observe_recorder_line(line)
        except Exception:
            pass
        finally:
            rc = proc.poll()
            # Any non-recorder process exiting with a non-zero code while the
            # stack is not being torn down is an unexpected crash. Surface it
            # in error_state immediately — StateChecker can take 20s+ to
            # notice a dead robot_controller, and the frontend needs the
            # signal now (the operator may be mid-pause/resume).
            if (
                not is_recorder
                and rc not in (None, 0)
                and not self._stop_in_progress
                and self._teleop_running
            ):
                with self._log_lock:
                    self._proc_crashed[name] = rc
                logger.error(f"{name} exited unexpectedly (code={rc})")
            elif (
                not is_recorder
                and rc == 0
                and name == self.ARM_HOLDER_PROC
                and not self._stop_in_progress
                and self._teleop_running
            ):
                # The operator ended the session from the joycon: R3+L3 sets
                # exit_requested, which brings the stack down without ever
                # touching the Stop endpoint. Nothing then reset the lifecycle,
                # so reconnecting was refused with "Teleop already running"
                # while the exit command's emergency_stop=True stayed latched
                # in _cmd_estop. Run the same teardown the Stop button does.
                # Keyed on the arm holder alone so the sibling tail threads do
                # not each schedule a teardown.
                logger.info(
                    f"{name} exited cleanly — teleop session ended by operator"
                )
                loop = self._log_loop
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._stop_teleop_background(), loop
                    )
            if is_recorder:
                self.recorder_ready = False
                # Subprocess has exited (clean or crash). If we still think
                # we're recording, flip state so the UI doesn't get stuck.
                if self.is_recording:
                    self.is_recording = False
                    self.recording_status = (
                        "stopped" if rc == 0 else "crashed"
                    )
                    self.episode_dir_name = None
                    logger.warning(
                        f"{name} exited (code={rc}) while recording — "
                        f"marked as {self.recording_status}"
                    )
                elif self.recording_status == "recording":
                    # Defensive — shouldn't normally happen.
                    self.recording_status = (
                        "stopped" if rc == 0 else "crashed"
                    )
                elif rc not in (None, 0):
                    self.recording_status = "crashed"

    def _observe_recorder_line(self, line: str):
        """Update recording_status + episode_dir_name from recorder stdout.

        The recorder prints these unambiguous markers on every state
        transition (see BaseRecorder.start_episode/end_episode/discard_episode):
          "🎬 RECORDING STARTED - Episode N"
          "💾 EPISODE SAVED - Episode N"
          "🗑️ EPISODE DISCARDED - Episode N"
        followed by a "📁 Directory: <name>" line.
        """
        # Keyword-based detection (avoids depending on emoji code points).
        if "RECORDER READY" in line:
            self.recorder_ready = True
            self.recording_status = "ready"
            # If the operator pre-filled web UI metadata before launch, push
            # it to the recorder now so JoyCon L+R starts carry it too.
            if self._recorder_ctrl_pub is not None and self._pending_record_metadata:
                self._recorder_ctrl_pub.publish(
                    {
                        "command": "set_metadata",
                        "metadata": self._pending_record_metadata,
                    }
                )
        elif "RECORDING STARTED" in line:
            self.is_recording = True
            self.recording_status = "recording"
            self.episode_dir_name = None  # will be set by the next Directory: line
        elif "EPISODE SAVED" in line:
            self.is_recording = False
            self.recording_status = "saved"
            self._on_episode_saved()
        elif "EPISODE DISCARDED" in line:
            self.is_recording = False
            self.recording_status = "discarded"

        # Capture the directory name from the "Directory: …" line that follows
        # a "RECORDING STARTED" marker.
        if self.recording_status == "recording":
            idx = line.find("Directory:")
            if idx >= 0:
                candidate = line[idx + len("Directory:") :].strip()
                if candidate and candidate != "N/A":
                    self.episode_dir_name = candidate

    # =========================================================================
    # Saved-episode counts
    # =========================================================================

    def _load_record_counts(self) -> Dict[str, Dict[str, int]]:
        """Load the persisted day→tag→count index, tolerating corruption."""
        try:
            if self._record_counts_path.is_file():
                raw = json.loads(self._record_counts_path.read_text())
                if isinstance(raw, dict):
                    return {
                        str(day): {
                            str(tag): int(n)
                            for tag, n in counts.items()
                            if isinstance(n, (int, float))
                        }
                        for day, counts in raw.items()
                        if isinstance(counts, dict)
                    }
        except (OSError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable record-count index: {exc}")
        return {}

    def _persist_record_counts(self) -> None:
        """Write the day→tag→count index next to save_dir."""
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._record_counts_path.write_text(
                json.dumps(self._record_counts, indent=2)
            )
        except OSError as exc:
            logger.warning(f"Could not persist record counts: {exc}")

    def _on_episode_saved(self) -> None:
        """Count one saved episode under the tag active at /record/start."""
        day = datetime.now().strftime("%m-%d-%Y")
        counts = self._record_counts.setdefault(day, {})
        tag = self._current_task_label or ""
        counts[tag] = counts.get(tag, 0) + 1
        self._persist_record_counts()

    def _set_pending_record_metadata(self, metadata: Dict[str, Any]) -> None:
        """Remember web UI metadata for JoyCon-initiated recording."""
        self._pending_record_metadata = dict(metadata or {})
        self._current_task_label = str(metadata.get("task_label") or "")

    def _record_counts_status(self) -> Dict[str, Any]:
        """Status-card numbers: current-tag saved today + total saved today."""
        day = datetime.now().strftime("%m-%d-%Y")
        today = self._record_counts.get(day, {})
        return {
            "tag_label": self._current_task_label or "",
            "tag_count": today.get(self._current_task_label or "", 0),
            "today_total": sum(today.values()),
        }

    # =========================================================================
    # Robot / StateChecker background loop
    # =========================================================================

    def initialize(self):
        """Initialize robot interface, state-checker loop, and video publisher."""
        logger.info("Initializing robot interface…")
        try:
            robot_configs = get_robot_config()
            robot_configs.sensors["head_camera"].enabled = True
            # Enable the wrist cameras whenever the robot config has them, so the
            # UI can view their streams (camera_3 / camera_4) regardless of
            # whether wrist RGB recording is turned on. Recording is gated
            # separately by the record components in the recorder subprocess.
            if "left_wrist_camera" in robot_configs.sensors:
                robot_configs.sensors["left_wrist_camera"].enabled = True
            if "right_wrist_camera" in robot_configs.sensors:
                robot_configs.sensors["right_wrist_camera"].enabled = True
            self.robot = ReadOnlyRobot(configs=robot_configs)
            self._estop_control = SoftwareEStopControl(self.robot.robot_info)

            if self.save_images:
                if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
                    logger.success("Camera streams active")
                else:
                    logger.warning("Camera streams may not be active")

            logger.success("Robot interface ready")
        except Exception as exc:
            logger.error(f"Robot init failed: {exc}")
            if self._estop_control is not None:
                try:
                    self._estop_control.shutdown()
                except Exception:
                    pass
                self._estop_control = None
            if self.robot is not None:
                try:
                    self.robot.shutdown()
                except Exception:
                    pass
            self.robot = None

        if self.state_checker:
            self._state_checker_running = True
            self._state_checker_thread = threading.Thread(
                target=self._state_checker_loop, daemon=True, name="StateCheckerLoop"
            )
            self._state_checker_thread.start()

        def get_frame(camera_id: str) -> bytes:
            # Replay cameras: "replay_<camera>" ids serve the recorded
            # JPEGs mirrored from the replay_record subprocess — no robot
            # needed.
            if camera_id.startswith("replay_"):
                cam = camera_id[len("replay_"):]
                with self._replay_lock:
                    return self._replay_frames.get(cam, b"")
            if not self.robot:
                return b""
            try:
                camera_map = {
                    "camera_1": ("head_camera", "left_rgb"),
                    "camera_2": ("head_camera", "right_rgb"),
                    # Wrist ZED X One cameras are single-stream: pass a
                    # None obs_key so get_frame calls get_obs() without
                    # obs_keys (that sensor takes include_timestamp, not
                    # obs_keys). Only present when the wrist_camera sensors
                    # are enabled; get_frame returns b"" otherwise.
                    "camera_3": ("left_wrist_camera", None),
                    "camera_4": ("right_wrist_camera", None),
                }
                if camera_id not in camera_map:

                    return b""
                sensor_name, obs_key = camera_map[camera_id]
                sensor = getattr(self.robot.sensors, sensor_name, None)
                if not sensor:
                    return b""
                obs = (
                    sensor.get_obs(obs_keys=[obs_key])
                    if obs_key
                    else sensor.get_obs()
                )
                img = obs.get(obs_key) if isinstance(obs, dict) and obs_key else obs
                if isinstance(img, dict):
                    img = img.get("data", img)
                if not isinstance(img, np.ndarray):
                    return b""
                if (
                    self.image_resolution
                    and img.shape[:2] != self.image_resolution[::-1]
                ):
                    img = cv2.resize(img, self.image_resolution)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buf.tobytes()
            except Exception:
                return b""

        video_publisher.set_frame_callback(get_frame)

    def _state_checker_loop(self):
        prev_state = None
        while self._state_checker_running and self.state_checker:
            try:
                # Robot joints come from robot_controller's robot/joints topic.
                # The publisher only starts after initialize() finishes homing,
                # so ALIGN naturally waits for the robot to reach home before
                # offering a stable alignment reference. When the topic goes
                # stale (robot_controller died), update_robot_joints stops and
                # get_alignment_status surfaces the data age instead.
                joints = self._latest_robot_joints.get("joints", {})
                left = joints.get("left_arm") or []
                right = joints.get("right_arm") or []
                if left and right:
                    self.state_checker.update_robot_joints(left, right)
                robot_state = self.state_checker.get_state()
                if robot_state != prev_state:
                    logger.info(
                        f"[StateChecker] transition: {prev_state} → {robot_state}"
                    )
                    prev_state = robot_state
                with self._robot_state_lock:
                    self.robot_state = robot_state
            except Exception as exc:
                logger.error(f"StateChecker loop error: {exc}")
                with self._robot_state_lock:
                    self.robot_state = "ERROR"
            # Avoid a tight loop that would re-run dextop topic list (and
            # create a fresh Zenoh session) immediately after the previous
            # call returns. Zenoh discovery is expensive; 2 s between ticks
            # is plenty for state transitions to propagate.
            time.sleep(2.0)

    # =========================================================================
    # Robot state observation helpers
    # =========================================================================

    async def _get_robot_state(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"joint_pos": {}, "joint_vel": {}, "wrench": {}}
        rc = self.record_components

        # Use joint feedback published by robot_controller (robot/joints topic).
        # Falls back to direct hardware read only if no feedback has arrived yet.
        joints_data = self._latest_robot_joints
        joint_positions = joints_data.get("joints", {})
        joint_velocities = joints_data.get("velocities", {})

        if joint_positions:
            for comp in [
                "left_arm",
                "right_arm",
                "torso",
                "head",
                "left_hand",
                "right_hand",
            ]:
                if rc.get(comp) and comp in joint_positions:
                    obs["joint_pos"][comp] = joint_positions[comp]
            for comp in ["left_arm", "right_arm"]:
                if rc.get(comp) and comp in joint_velocities:
                    obs["joint_vel"][comp] = joint_velocities[comp]
        if self.robot:
            # Hardware reads (fallback joints + wrench) run in a thread with a
            # timeout — a hung SDK read must not freeze the event loop (this
            # path is shared by GET /state and the 10 Hz state broadcast).
            def _read_hardware(joints: bool) -> Dict[str, Any]:
                hw: Dict[str, Any] = {"joint_pos": {}, "wrench": {}}
                if joints:
                    for comp in ["left_arm", "right_arm"]:
                        if not rc.get(comp):
                            continue
                        arm = getattr(self.robot, comp, None)
                        if arm is None:
                            continue
                        try:
                            hw["joint_pos"][comp] = arm.get_joint_pos().tolist()
                        except Exception:
                            pass
                if rc.get("left_wrist_wrench") and self.robot.left_arm.wrench_sensor:
                    try:
                        w = self.robot.left_arm.wrench_sensor.get_wrench_state()
                        hw["wrench"]["left"] = {
                            "force": w[:3].tolist(),
                            "torque": w[3:].tolist(),
                        }
                    except Exception:
                        pass
                if rc.get("right_wrist_wrench") and self.robot.right_arm.wrench_sensor:
                    try:
                        w = self.robot.right_arm.wrench_sensor.get_wrench_state()
                        hw["wrench"]["right"] = {
                            "force": w[:3].tolist(),
                            "torque": w[3:].tolist(),
                        }
                    except Exception:
                        pass
                return hw

            # Joints read from hardware only when the topic feed is empty
            # (robot_controller not publishing yet); wrench always from hardware.
            try:
                hw = await asyncio.wait_for(
                    asyncio.to_thread(_read_hardware, not bool(joint_positions)),
                    timeout=1.0,
                )
            except (asyncio.TimeoutError, Exception):
                hw = {}
            obs["joint_pos"].update(hw.get("joint_pos", {}))
            obs["wrench"].update(hw.get("wrench", {}))
        return obs

    # =========================================================================
    # Recording (only active when record_mode=True)
    # =========================================================================

    def _on_joints_received(self, data: Dict[str, Any]):
        self._latest_robot_joints = data

    def _on_command_received(self, data: Dict[str, Any]):
        now = time.time()
        with self._action_lock:
            for comp, comp_data in data.get("components", {}).items():
                self.current_action[comp] = comp_data
                self._component_last_seen[comp] = now

        estop_active = data.get("safety_flags", {}).get("emergency_stop", False)

        # GUI estop and command_processor estop are independent sources with
        # their own lifecycles. Only update the cmd flag here — GUI estop is
        # cleared exclusively by DELETE /robots/estop. Touching _gui_estop on
        # every normal command (which carries emergency_stop=False) would
        # immediately undo a GUI-initiated E-Stop and flicker the button back.
        self._cmd_estop = estop_active
        self._cmd_pending_realign = data.get("safety_flags", {}).get(
            "pending_realign", False
        )
        self._last_command_at = now
        loop = self._log_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._sync_effective_estop(), loop)
        # Note: recorder handles its own auto_stop_on_estop via its own
        # robot_commands subscriber (BaseRecorder._on_command_received).

    def _cmd_estop_is_live(self) -> bool:
        """True only when a running producer is still asserting E-Stop.

        The last command published before teleop exits carries
        emergency_stop=True — CommandProcessor forces it alongside
        exit_requested — so _cmd_estop latches True and, with the producer
        gone, no later message ever flips it back. A joycon R3+L3 exit
        therefore left the operator permanently unable to clear E-Stop, since
        the clear endpoint refuses while _cmd_estop is set. An E-Stop request
        from a stream that stopped is not a live objection.
        """
        if not self._cmd_estop:
            return False
        last = self._last_command_at
        return (
            last is not None and (time.time() - last) <= self.CMD_ESTOP_STALE_S
        )

    async def _sync_effective_estop(self):
        """Sync effective estop (gui OR command_processor) to state_manager."""
        if self._gui_estop or self._cmd_estop:
            await state_manager.estop()
        else:
            await state_manager.clear_estop()

    # =========================================================================
    # Run
    # =========================================================================

    def cleanup(self):
        self._state_checker_running = False
        if self._state_checker_thread:
            self._state_checker_thread.join(timeout=2.0)
        if self.state_checker:
            try:
                self.state_checker.cleanup()
            except Exception:
                pass
        # Recorder subprocess (if running) is shut down as part of _stop_teleop_procs.
        self._stop_teleop_procs()
        if self.robot:
            self.robot.shutdown()
        if self._estop_control:
            self._estop_control.shutdown()
            self._estop_control = None
        if hasattr(self._node, "shutdown"):
            self._node.shutdown()

    def run(self):
        import os

        certs_dir = os.path.join(os.path.dirname(__file__), "..", "certs")
        cert_file = os.path.join(certs_dir, "cert.pem")
        key_file = os.path.join(certs_dir, "key.pem")
        ssl_kwargs = {}
        if os.path.isfile(cert_file) and os.path.isfile(key_file):
            ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}
        uvicorn.run(
            self.app, host="0.0.0.0", port=5006, log_level="warning", **ssl_kwargs
        )


# =============================================================================
# Entry point
# =============================================================================


def main(
    namespace: str = "",
    debug: bool = False,
    rpi_mode: bool = False,
    lock_hand_collision_avoidance: bool = False,
    unlock_torso: bool = False,
):
    """OmniTeleop App Backend.

    Launches the FastAPI server. Teleoperation processes are started via
    POST /teleop/start from the frontend, which also passes env vars
    (ROBOT_NAME, ROBOT_CONFIG, ZENOH_CONFIG) and the record_mode flag.

    Args:
        namespace: Optional Zenoh namespace prefix.
        debug: Enable verbose logging.
        rpi_mode: RPi mode — leader scripts (arm_reader, joycon_reader) are
            assumed to already be running on a remote machine and will NOT be
            started locally. Only follower scripts are launched.
        lock_hand_collision_avoidance: Forwarded to command_processor — freezes
            hand joints in the MotionManager collision model at the configured
            "open" pose. Joycon hand commands still drive the real robot.
        unlock_torso: Forwarded to paddle_leader for vr / vr_sim modes — lets
            the IK solver recruit torso joints. No-op on upper-body-only
            embodiments (e.g. vega_1u). Default off so the same launch works
            across full and upper-body Vegas.
    """
    os.environ.setdefault("ROBOT_CONFIG", "vega_1u_gripper")
    if not os.environ.get("ZENOH_CONFIG"):
        zenoh_base = Path.home() / ".dexmate" / "comm" / "zenoh"
        try:
            detected_zenoh = _autodetect_zenoh_cert(zenoh_base)
        except FileNotFoundError:
            detected_zenoh = None
        if detected_zenoh:
            os.environ["ZENOH_CONFIG"] = str(detected_zenoh)
    setup_logging(debug)
    logger.info("OmniTeleop App Backend starting" + (" [rpi-mode]" if rpi_mode else ""))

    app = TeleopApp(
        namespace=namespace,
        debug=debug,
        rpi_mode=rpi_mode,
        lock_hand_collision_avoidance=lock_hand_collision_avoidance,
        unlock_torso=unlock_torso,
    )
    try:
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
    finally:
        app.cleanup()


if __name__ == "__main__":
    tyro.cli(main)
