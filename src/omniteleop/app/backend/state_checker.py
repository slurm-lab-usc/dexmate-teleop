"""State checker for monitoring robot control states.

Determines the current operational state of the robot by checking network
connectivity, topic availability, process status, and joint positions.
"""

import math
import os
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

from omniteleop.common.platform import AUTO, find_serial_port
class StateChecker:
    """Determines robot control state via a priority-based logic tree.

    States are checked in order: DEAD → BOOT → DIAGNOSIS → ALIGN → ACTIVE.

    Attributes:
        robot_name: Robot namespace name for topic resolution.
        jetson_ip: IP address of the Jetson for connectivity checks.

    Example:
        >>> checker = StateChecker("dm/robot1", "192.168.0.10")
        >>> state = checker.get_state()
        >>> print(state)  # "DEAD", "BOOT", "DIAGNOSIS", "ALIGN", or "ACTIVE"
        >>> checker.cleanup()
    """

    # Threshold (radians) for exo-vs-robot proximity check in ALIGN state.
    ESTOP_ALIGN_THRESHOLD: float = 0.75
    REQUIRED_TOPICS_GRACE_SEC: float = 20.0
    JOYCON_DATA_TIMEOUT_SEC: float = 2.0

    # Threshold (radians) between exo and config init_pos. Must stay in sync
    # with CommandProcessor._align_threshold — CommandProcessor gates motion
    # on this criterion, and StateChecker reports ALIGN whenever the gate is
    # engaged so the UI shows "Aligning" while the robot is actually blocked.
    #
    # 0.50 rad (~29°) trades a slightly larger start-up jump for an alignment
    # pose the operator can actually hold by hand; 1.0 rad (the 0.4.3 default)
    # accepted a ~57° mismatch as "aligned" and produced a visible jump at
    # start. A jump beyond the realtime tracking tolerance (0.30 rad) is now
    # absorbed by the stall recovery in RobotController rather than by keeping
    # this gate tight.
    INIT_POS_ALIGN_THRESHOLD: float = 0.50

    # Joint limits for exoskeleton in radians, format: (min, max).
    JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
        "arm_j1": (-3.071, 3.071),
        "arm_j2_left": (-0.453, 1.553),
        "arm_j2_right": (-1.553, 0.453),
        "arm_j3": (-3.071, 3.071),
        "arm_j4": (-3.071, 0.244),
        "arm_j5": (-3.071, 3.071),
        "arm_j6": (-1.396, 1.396),
        "arm_j7_left": (-1.378, 1.117),
        "arm_j7_right": (-1.117, 1.378),
    }

    # Topics required for the BOOT state check (exoskeleton default).
    REQUIRED_TOPIC_SUFFIXES: List[str] = [
        "state/arm/left",
        "state/arm/right",
        "heartbeat",
        "sensors/head_camera/left_rgb",
        "sensors/head_camera/right_rgb",
    ]


    # Sensor topics surfaced to the UI so users can verify each stream is live
    # and whether the active config will record it. Mapping goes from the topic
    # suffix (robot_name prepended at check time) → the matching key under
    # ``recorder.components`` in the robot YAML config. Keep in sync with the
    # record_components dict in ``record/base_recorder.py``.
    SENSOR_TOPIC_TO_RECORD_KEY: Dict[str, str] = {
        "sensors/head_camera/left_rgb": "head_left_rgb",
        "sensors/head_camera/right_rgb": "head_right_rgb",
        "sensors/head_camera/depth": "head_left_depth",
        "sensors/left_wrist_camera/rgb": "left_wrist_rgb",
        "sensors/right_wrist_camera/rgb": "right_wrist_rgb",
        "state/wrench/left": "left_wrist_wrench",
        "state/wrench/right": "right_wrist_wrench",
    }
    SENSOR_TOPIC_SUFFIXES: List[str] = list(SENSOR_TOPIC_TO_RECORD_KEY.keys())

    # Processes required to pass the DIAGNOSIS state.
    REQUIRED_PROCESSES: List[str] = [
        "command_processor.py",
        "robot_controller.py",
        "joycon_reader.py",
        "arm_reader.py",
    ]

    def __init__(
        self,
        robot_name: str,
        rpi_mode: bool = False,
        leader_mode: str = "exoskeleton",
    ) -> None:
        """Initializes the state checker.

        Args:
            robot_name: Robot namespace name used for topic resolution.
            rpi_mode: If True, skip local hardware/process checks and instead
                verify that the RPi is publishing ``exo/joints`` and
                ``exo/joycon`` topics.
            leader_mode: ``"exoskeleton"``, ``"vr"``, or ``"vr_sim"``.
                Determines which BOOT/DIAGNOSIS checks apply. ``"vr"`` skips
                exo/joycon-specific checks (paddle_leader replaces those).
                ``"vr_sim"`` skips all robot-side checks (no follower).
        """
        self.robot_name = robot_name
        self.rpi_mode = rpi_mode
        self.leader_mode = leader_mode if leader_mode in {"exoskeleton", "vr", "vr_sim"} else "exoskeleton"

        # Exo joints subscriber state.
        self._node = None
        self._exo_subscriber = None
        self._joycon_subscriber = None
        self._estop_subscriber = None
        self._exo_data: Dict[str, Optional[dict]] = {"data": None}
        self._exo_data_event = threading.Event()
        self._last_exo_data_at: float = 0.0
        self._last_joycon_data_at: float = 0.0
        self._latest_estop_state: dict = {}

        # Component query interface and robot instance.
        self._query_interface = None
        self._component_errors_cache: dict = {}

        # Latest robot joint positions, updated externally via update_robot_joints().
        self._robot_left_joints: List[float] = []
        self._robot_right_joints: List[float] = []
        self._robot_joints_at: float = 0.0

        # Diagnosis details populated on each call to _check_active_conditions.
        self._diagnosis_details: dict = {}

        # Leader-arm serial port as configured; _check_exo_hardware resolves it
        # by USB id so this agrees with arm_reader on Linux and macOS alike.
        self.exo_port: str = AUTO

        # Init positions loaded from the same config CommandProcessor reads.
        # Used by the init_pos ALIGN criterion — if any exo joint drifts more
        # than INIT_POS_ALIGN_THRESHOLD from init_pos, CommandProcessor will
        # refuse to start motion, so we report ALIGN.
        try:
            from omniteleop.common import get_config
            _cfg = get_config()
            _recorder = (_cfg.get("recorder") or {})
            _rec_components = (_recorder.get("components") or {})
            self.exo_port = (_cfg.get_leader_arms() or {}).get("port", AUTO)
        except Exception:
            _recorder = {}
            _rec_components = {}
        # Per-component recording flags from config — drives the Recorded /
        # Not recorded label in the Sensors UI. ``enabled`` is the global
        # recorder on/off; if False every sensor reports "not recorded"
        # regardless of its component flag.
        self._recorder_enabled: bool = bool(_recorder.get("enabled", False))
        self._recorder_components: Dict[str, bool] = {
            k: bool(v) for k, v in _rec_components.items()
        }

        # Cached `dextop topic list` output. Refreshed by each get_state() call
        # so get_sensor_topics_status() can reuse it without shelling out again.
        self._last_topic_list: Optional[str] = None
        self._last_topic_list_at: float = 0.0
        self._last_required_topics_ok_at: float = 0.0
        # Background topic-list fetch so the 20 s discovery timeout never
        # blocks the StateChecker tick thread.
        self._topic_fetch_lock = threading.Lock()
        self._topic_fetch_pending = False

        # Once motion has started for the first time (operator pressed then
        # released estop after initial alignment), ALIGN is bypassed forever
        # for this teleop session. Reset by reset_motion_started() on stop.
        self._motion_started_latch: bool = False

        self._init_exo_subscriber()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _init_exo_subscriber(self) -> None:
        """Initializes reusable exoskeleton and Joy-Con Zenoh subscribers."""
        try:
            from dexcomm import Node
            from dexcomm.codecs import DictDataCodec, EStopStateCodec
            from dexbot_utils import RobotInfo

            def on_exo_joints(data: dict) -> None:
                self._exo_data["data"] = data
                self._last_exo_data_at = time.monotonic()
                self._exo_data_event.set()

            def on_joycon(data: dict) -> None:
                # A fresh message is the most direct indication that the same
                # evdev-based Joy-Con path used by teleoperation is healthy.
                if isinstance(data, dict) and data:
                    self._last_joycon_data_at = time.monotonic()

            self._node = Node(
                name="state_checker_exo_reader",
                namespace=self.robot_name,
            )
            self._exo_subscriber = self._node.create_subscriber(
                "exo/joints",
                callback=on_exo_joints,
                decoder=DictDataCodec.decode,
            )
            self._joycon_subscriber = self._node.create_subscriber(
                "exo/joycon",
                callback=on_joycon,
                decoder=DictDataCodec.decode,
            )
            robot_info = RobotInfo()
            estop_config = robot_info.get_component_config("estop")
            self._estop_subscriber = self._node.create_subscriber(
                estop_config.state_sub_topic,
                callback=lambda data: setattr(
                    self, "_latest_estop_state", dict(data)
                ),
                decoder=EStopStateCodec.decode,
            )
        except Exception:
            self._node = None
            self._exo_subscriber = None
            self._joycon_subscriber = None

    def _init_query_interface(self) -> None:
        """Initializes the reusable robot query interface."""
        try:
            from dexcontrol.core.robot_query_interface import RobotQueryInterface

            self._query_interface = RobotQueryInterface.create()
        except Exception:
            self._query_interface = None

    def cleanup(self) -> None:
        """Releases network connections and other resources.

        Should be called when done using the StateChecker.
        """
        if self._query_interface:
            try:
                self._query_interface.close()
            except Exception:
                pass
            self._query_interface = None

        if self._node:
            try:
                self._node.shutdown()
            except Exception:
                pass
            self._node = None

    # -------------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------------

    def get_state(self) -> str:
        """Returns the current robot state.

        Follows a priority-based logic tree:

            1. BOOT      - Required topics not yet published.
            2. DIAGNOSIS - Processes, JoyCons, or hardware not ready.
            3. ALIGN     - Exo joints outside safe limits.
            4. ACTIVE    - All checks passed; ready for operation.

        Returns:
            One of: ``"BOOT"``, ``"DIAGNOSIS"``, ``"ALIGN"``, ``"ACTIVE"``.
        """
        now = time.time()
        # Refresh the topic list in a background thread so the 20 s
        # discovery timeout never blocks the StateChecker tick. While a
        # fetch is in-flight, keep using the last known list (or None).
        needs_refresh = (
            self._last_topic_list is None
            or now - self._last_topic_list_at > self._TOPIC_LIST_REFRESH_INTERVAL_S
        )
        if needs_refresh and not self._topic_fetch_pending:
            with self._topic_fetch_lock:
                if not self._topic_fetch_pending:
                    self._topic_fetch_pending = True
                    threading.Thread(
                        target=self._fetch_topic_list_background,
                        daemon=True,
                        name="StateCheckerTopicFetch",
                    ).start()
        topic_list = self._last_topic_list
        topics_ok = topic_list is not None and self._check_required_topics(topic_list)
        if topics_ok:
            self._last_required_topics_ok_at = now
        elif now - self._last_required_topics_ok_at > self.REQUIRED_TOPICS_GRACE_SEC:
            return "BOOT"

        if not self._check_active_conditions(topic_list or ""):
            return "DIAGNOSIS"

        # ALIGN compares the worn exoskeleton against the robot's init_pos —
        # only meaningful in exoskeleton mode. VR / VR sim have no exo input,
        # so skip straight to ACTIVE once DIAGNOSIS passes.
        if self.leader_mode == "exoskeleton" and not self._check_exo_joints_within_limits():
            return "ALIGN"

        return "ACTIVE"

    # -------------------------------------------------------------------------
    # BOOT — Topic availability
    # -------------------------------------------------------------------------

    _TOPIC_LIST_DISCOVERY_TIMEOUT_S: float = 20.0
    _TOPIC_LIST_RETRIES: int = 2
    _TOPIC_LIST_REFRESH_INTERVAL_S: float = 12.0

    def _fetch_topic_list_background(self) -> None:
        """Run dextop topic list in a daemon thread so the tick is never blocked."""
        try:
            result = self._get_topic_list()
            now = time.time()
            self._last_topic_list = result
            self._last_topic_list_at = now
        except Exception:
            pass
        finally:
            self._topic_fetch_pending = False

    def _get_topic_list(self) -> Optional[str]:
        """Retrieves the list of active topics from dextop.

        Returns:
            Topic list string, or None if the command fails.
        """
        try:
            cmd = [
                "dextop",
                "topic",
                "list",
                "--timeout",
                str(self._TOPIC_LIST_DISCOVERY_TIMEOUT_S),
                "--retries",
                str(self._TOPIC_LIST_RETRIES),
            ]
            # Explicitly pass the Zenoh config when available so dextop
            # doesn't have to fall back to slow multicast scouting.
            zenoh_config = os.environ.get("ZENOH_CONFIG", "")
            if zenoh_config:
                cmd.extend(["--config", zenoh_config])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._TOPIC_LIST_DISCOVERY_TIMEOUT_S + 10.0,
            )
            return result.stdout if result.stdout else None
        except Exception:
            return None

    def _build_required_topics(self) -> List[str]:
        """Builds the fully-qualified required topic names for this robot.

        Branches on ``self.leader_mode``:
          - ``vr_sim``: no required topics — BOOT is trivially satisfied.
            Process checks still apply in DIAGNOSIS.
          - ``vr``: head cameras only. Robot readiness is checked in
            DIAGNOSIS via the follower/leader processes and component health;
            robot state topics can be transient while the VR stack owns the
            direct Robot() connections.
          - ``exoskeleton``: full robot-side requirements (arm state,
            heartbeat, head cameras).

        Returns:
            List of fully-qualified topic name strings.
        """
        if self.leader_mode == "vr_sim":
            return []
        if self.leader_mode == "vr":
            return [
                f"{self.robot_name}/sensors/head_camera/left_rgb",
                f"{self.robot_name}/sensors/head_camera/right_rgb",
            ]
        suffixes = list(self.REQUIRED_TOPIC_SUFFIXES)
        return [f"{self.robot_name}/{s}" for s in suffixes]

    def _check_required_topics(self, topic_list: str) -> bool:
        """Checks whether all required topics are present.

        Args:
            topic_list: Raw string output from ``dextop topic list``.

        Returns:
            True if all required topics appear in the topic list.
        """
        return all(t in topic_list for t in self._build_required_topics())

    def get_missing_topics(self) -> dict:
        """Returns which required topics are missing and which are found.

        Returns:
            Dictionary with keys:
                missing: List of topic names not yet published.
                found:   List of topic names that are active.
        """
        required = self._build_required_topics()
        topic_list = self._last_topic_list

        if topic_list is None:
            return {"missing": required, "found": []}

        missing = [t for t in required if t not in topic_list]
        found = [t for t in required if t in topic_list]
        return {"missing": missing, "found": found}

    def get_sensor_topics_status(self) -> Dict[str, Dict[str, bool]]:
        """Returns publication + recording status for each sensor topic.

        Reuses the topic list captured during the most recent
        :meth:`get_state` call, so calling this does not shell out again.
        If ``get_state`` has not run yet (or the last call failed to get a
        topic list), every sensor reports ``live=False``.

        The ``recorded`` flag reflects the active robot YAML config
        (selected via the ``ROBOT_CONFIG`` env var by ``get_config``). A
        sensor is "recorded" iff ``recorder.enabled`` is true AND its
        entry in ``recorder.components`` is truthy.

        Returns:
            Dict mapping topic suffix → ``{"live": bool, "recorded": bool}``.
            Empty when leader_mode is ``"vr_sim"`` — no robot, no sensors.
        """
        if self.leader_mode == "vr_sim":
            return {}
        topic_list = self._last_topic_list
        result: Dict[str, Dict[str, bool]] = {}
        for suffix, record_key in self.SENSOR_TOPIC_TO_RECORD_KEY.items():
            live = (
                topic_list is not None
                and f"{self.robot_name}/{suffix}" in topic_list
            )
            recorded = (
                self._recorder_enabled
                and bool(self._recorder_components.get(record_key, False))
            )
            result[suffix] = {"live": live, "recorded": recorded}
        return result

    # -------------------------------------------------------------------------
    # DIAGNOSIS — Process and hardware readiness
    # -------------------------------------------------------------------------

    def _check_active_conditions(self, topic_list: str) -> bool:
        """Checks all conditions required to pass the DIAGNOSIS state.

        Branches on ``self.leader_mode``:
          - ``vr_sim``: only the paddle_leader process is required. No
            joycon/exo/robot checks (no robot in the loop).
          - ``vr``: paddle_leader + robot_controller processes plus robot
            component health. Skips joycon, exo serial port, and exo motors
            checks (paddle_leader replaces those leader-side responsibilities).
          - ``exoskeleton`` (default): the historical full set.

        Results are stored in ``_diagnosis_details`` for later retrieval
        via :meth:`get_diagnosis_details`.

        Args:
            topic_list: Raw string output from ``dextop topic list``.

        Returns:
            True if all active conditions are met.
        """
        if self.leader_mode == "vr_sim":
            self._diagnosis_details = {
                "processes_running": self._check_processes_running(),
            }
            return all(self._diagnosis_details.values())

        if self.leader_mode == "vr":
            self._diagnosis_details = {
                "processes_running": self._check_processes_running(),
                "no_component_errors": not self._check_component_errors(),
            }
            return all(self._diagnosis_details.values())

        if self.rpi_mode:
            # In RPi mode the leader hardware (JoyCon, exo serial port, arm_reader,
            # joycon_reader) lives on a remote machine. Replace those checks with a
            # single liveness check: both exo topics must be visible on the network.
            rpi_ready = self._check_rpi_topics(topic_list)
            self._diagnosis_details = {
                "joycon_connected": rpi_ready,
                "processes_running": rpi_ready,
                "exo_hardware_ok": rpi_ready,
                "exo_motors_connected": self._check_exo_motors(topic_list),
                "no_component_errors": not self._check_component_errors(),
            }
        else:
            self._diagnosis_details = {
                "joycon_connected": self._check_joycons_connected(),
                "processes_running": self._check_processes_running(),
                "exo_hardware_ok": self._check_exo_hardware(),
                "exo_motors_connected": self._check_exo_motors(topic_list),
                "no_component_errors": not self._check_component_errors(),
            }
        return all(self._diagnosis_details.values())

    def _check_joycons_connected(self) -> bool:
        """Checks whether the teleoperation Joy-Con stream is live.

        Returns:
            True if a decoded ``exo/joycon`` message arrived recently.
        """
        if self._last_joycon_data_at <= 0.0:
            return False
        return (
            time.monotonic() - self._last_joycon_data_at
            <= self.JOYCON_DATA_TIMEOUT_SEC
        )

    def _required_processes(self) -> List[str]:
        """Process names that must be running for DIAGNOSIS to pass.

        Branches on ``self.leader_mode`` since each leader stack spawns a
        different set of subprocesses.
        """
        if self.leader_mode == "vr_sim":
            return ["paddle_leader.py"]
        if self.leader_mode == "vr":
            return ["paddle_leader.py", "robot_controller.py"]
        return self.REQUIRED_PROCESSES

    def _check_processes_running(self) -> bool:
        """Checks if all required processes are running.

        Returns:
            True if all entries in the mode-specific process list appear in
            ``ps aux`` output.
        """
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            process_list = result.stdout
            return all(proc in process_list for proc in self._required_processes())
        except Exception:
            return False

    def _check_exo_hardware(self) -> bool:
        """Checks if the exoskeleton serial port is accessible.

        The port is resolved the same way ``arm_reader`` resolves it (by USB
        id) rather than assumed to be ``/dev/ttyUSB0``, so this check agrees
        with the reader on both Ubuntu (``/dev/ttyUSB*``) and macOS
        (``/dev/cu.usbserial-*``).

        Returns:
            True if the adapter is present and readable/writable.
        """
        port = find_serial_port(self.exo_port)
        if not port:
            return False
        return os.access(port, os.R_OK | os.W_OK)

    def _check_exo_motors(self, topic_list: str) -> bool:
        """Checks if the exo arm reader is publishing joint data.

        Args:
            topic_list: Raw string output from ``dextop topic list``.

        Returns:
            True if the ``exo/joints`` topic is present.
        """
        return f"{self.robot_name}/exo/joints" in topic_list

    def _check_rpi_topics(self, topic_list: str) -> bool:
        """Checks if the RPi leader machine is publishing its required topics.

        Used in rpi_mode as a combined proxy for joycon_connected,
        processes_running, and exo_hardware_ok.

        Returns:
            True if both ``exo/joints`` and ``exo/joycon`` are present.
        """
        return (
            f"{self.robot_name}/exo/joints" in topic_list
            and f"{self.robot_name}/exo/joycon" in topic_list
        )

    def _check_component_errors(self) -> bool:
        """Checks if any robot components are reporting errors.

        Returns:
            True if any component has a non-zero error code, non-empty
            error message, or an operation value of 0.
        """
        if not self._query_interface:
            self._init_query_interface()
        if not self._query_interface:
            return False

        try:
            status = self._query_interface.get_component_status(show=False)
            errors: dict = {}
            for component_status in status.get("states", {}).values():
                if not isinstance(component_status, dict):
                    continue

                error = component_status.get("error")
                if isinstance(error, dict):
                    if error.get("error_code", 0) != 0:
                        self._component_errors_cache = status.get("states", {})
                        return True
                    if error.get("error_message", ""):
                        self._component_errors_cache = status.get("states", {})
                        return True

                operation = component_status.get("operation")
                if operation is not None and operation == 0:
                    self._component_errors_cache = status.get("states", {})
                    return True

            self._component_errors_cache = errors
            return False
        except Exception:
            return False

    def get_diagnosis_details(self) -> dict:
        """Returns diagnostic check results from the last state evaluation.

        Useful for surfacing which specific check failed when state is
        ``DIAGNOSIS``.

        Returns:
            Dictionary mapping check names to bool results. Empty dict if
            :meth:`get_state` has not yet been called. Keys:
                joycon_connected, processes_running, exo_hardware_ok,
                exo_motors_connected, no_component_errors.
        """
        return self._diagnosis_details

    def get_component_errors(self) -> Tuple[dict, bool]:
        """Returns detailed component errors and software estop status.

        Returns:
            Tuple of (errors, estop_enabled):
                errors:        Dict mapping component name to error detail dict.
                               Contains a ``"_query_error"`` key if the query
                               itself fails.
                estop_enabled: True if the software estop is currently active.

        Example:
            errors, estop_on = checker.get_component_errors()
            # errors = {"left_arm": {"error_code": 5, "operation": 0}}
            # estop_on = True
        """
        errors: dict = {}
        for component_name, component_status in self._component_errors_cache.items():
            if not isinstance(component_status, dict):
                continue
            error = component_status.get("error")
            operation = component_status.get("operation")
            if (
                isinstance(error, dict)
                and (
                    error.get("error_code", 0) != 0
                    or bool(error.get("error_message"))
                )
            ) or operation == 0:
                errors[component_name] = {
                    "error_code": error.get("error_code", 0)
                    if isinstance(error, dict)
                    else 0,
                    "error_message": error.get("error_message", "")
                    if isinstance(error, dict)
                    else "Component operation status is 0 (error)",
                    "operation": operation,
                }
        return errors, bool(
            self._latest_estop_state.get("software_estop_enabled", False)
        )

    # -------------------------------------------------------------------------
    # ALIGN — Exo joint limits
    # -------------------------------------------------------------------------

    def _check_exo_joints_within_limits(self) -> bool:
        """Pre-motion alignment check.

        Compares exo joints against the robot's CURRENT joint positions
        (not config init_pos). Once motion has started for the first time
        (``_motion_started_latch`` is set via :meth:`mark_motion_started`),
        this always returns True — ALIGN is never shown again.

        Returns False (ALIGN) if:
          - Latch not set AND exo data unavailable / short, OR
          - Any exo joint outside hardware limits, OR
          - Any exo joint more than INIT_POS_ALIGN_THRESHOLD from the robot's
            current joint position.
        """
        if self._motion_started_latch:
            return True

        # Keep the state transition and the UI guide on one source of truth.
        # If the guide shows 14/14 ready, this check will pass on the same
        # sample instead of duplicating subtly different criteria here.
        return bool(self.get_alignment_status()["aligned"])

    def _arm_within_threshold(
        self, exo: List[float], ref: List[float]
    ) -> bool:
        """Returns True iff every exo joint is within INIT_POS_ALIGN_THRESHOLD
        of the corresponding reference joint (robot current position).

        **Strict**: missing/empty/short data returns False (stays in ALIGN).
        """
        if not exo or len(exo) < 7 or not ref or len(ref) < 7:
            return False
        for e, r in zip(exo, ref):
            if abs(e - r) > self.INIT_POS_ALIGN_THRESHOLD:
                return False
        return True

    def mark_motion_started(self) -> None:
        """Permanently bypass ALIGN for this teleop session.

        Call once the operator has successfully started motion (first estop
        release after initial alignment). After this, get_state() will never
        return "ALIGN" regardless of exo or robot joint positions.
        """
        self._motion_started_latch = True

    def reset_motion_started(self) -> None:
        """Reset the motion-started latch when teleop stops.

        Allows ALIGN to be shown again at the start of the next session.
        """
        self._motion_started_latch = False

    def update_robot_joints(self, left: List[float], right: List[float]) -> None:
        """Updates the cached robot joint positions used for proximity checks.

        Should be called by the owner (e.g. recorder_backend) before each
        call to :meth:`get_state` to ensure the proximity check uses live
        robot joint positions that match what is displayed as observations.

        Args:
            left:  Current left arm joint positions in radians.
            right: Current right arm joint positions in radians.
        """
        self._robot_left_joints = left
        self._robot_right_joints = right
        self._robot_joints_at = time.monotonic()

    def _check_exo_proximity_to_robot(self) -> bool:
        """Checks if exo joints are within ESTOP_ALIGN_THRESHOLD of robot joints.

        Uses robot joint positions last provided via :meth:`update_robot_joints`.

        Returns:
            True if all joints are within threshold or if no robot joint data
            is available. False if any joint deviates beyond the threshold.
        """
        robot_left = self._robot_left_joints
        robot_right = self._robot_right_joints

        if not robot_left and not robot_right:
            return True

        exo_joints = self.get_latest_exo_joints()
        if not exo_joints:
            return False

        sides = (
            ("left", exo_joints.get("left", []), robot_left),
            ("right", exo_joints.get("right", []), robot_right),
        )
        for side, exo, robot in sides:
            if not exo or not robot:
                continue
            for e, r in zip(exo, robot):
                if abs(e - r) > self.ESTOP_ALIGN_THRESHOLD:
                    return False

        return True

    def get_out_of_limit_joints(self) -> dict:
        """Returns which joints are currently outside their limits.

        Returns:
            Dictionary with keys:
                message:   Human-readable summary.
                left_arm:  List of out-of-limit joint descriptions (if any).
                right_arm: List of out-of-limit joint descriptions (if any).
        """
        # UI state generation must never wait for a fresh sample. The
        # subscriber callback continuously refreshes this cache.
        joint_angles = self.get_latest_exo_joints()

        if joint_angles is None:
            return {"message": "Cannot read exoskeleton joint angles"}

        result: dict = {"message": "Exoskeleton joints not within limits"}

        for side, is_left in (("left", True), ("right", False)):
            joints = joint_angles.get(side, [])
            if joints and len(joints) >= 7:
                out_of_limit = self._get_out_of_limit_details(joints, is_left)
                if out_of_limit:
                    result[f"{side}_arm"] = out_of_limit

        return result

    def get_alignment_status(self) -> dict:
        """Return non-blocking, per-joint guidance for the ALIGN UI.

        The values and pass/fail criteria intentionally match
        :meth:`_check_exo_joints_within_limits`: every exoskeleton joint must
        be inside its configured hardware range and within
        ``INIT_POS_ALIGN_THRESHOLD`` of the robot's current joint position.
        ``target_delta`` is ``robot - exo``; a positive value tells the
        operator to increase the displayed exoskeleton angle.
        """
        exo = self.get_latest_exo_joints()
        robot = {
            "left": self._robot_left_joints,
            "right": self._robot_right_joints,
        }
        now = time.monotonic()
        exo_at = getattr(self, "_last_exo_data_at", 0.0)
        robot_at = getattr(self, "_robot_joints_at", 0.0)

        result = {
            "aligned": False,
            "threshold_rad": self.INIT_POS_ALIGN_THRESHOLD,
            "aligned_joints": 0,
            "total_joints": 14,
            "reason": "not_aligned",
            "exo_data_age_sec": round(now - exo_at, 2) if exo_at else None,
            "robot_data_age_sec": round(now - robot_at, 2) if robot_at else None,
            "arms": {},
        }

        if exo is None:
            result["reason"] = "exo_data_missing"
        elif not robot["left"] or not robot["right"]:
            result["reason"] = "robot_data_missing"

        for side, is_left in (("left", True), ("right", False)):
            exo_joints = (exo or {}).get(side) or []
            robot_joints = robot[side] or []
            joint_rows = []

            for index, limit_key in enumerate(self._get_limit_keys(is_left)):
                lo, hi = self.JOINT_LIMITS[limit_key]
                exo_value = exo_joints[index] if index < len(exo_joints) else None
                robot_value = (
                    robot_joints[index] if index < len(robot_joints) else None
                )
                values_valid = (
                    isinstance(exo_value, (int, float))
                    and isinstance(robot_value, (int, float))
                    and math.isfinite(float(exo_value))
                    and math.isfinite(float(robot_value))
                )

                in_limits = (
                    values_valid and lo <= float(exo_value) <= hi
                )
                target_delta = (
                    float(robot_value) - float(exo_value)
                    if values_valid
                    else None
                )
                error = abs(target_delta) if target_delta is not None else None
                aligned = bool(
                    in_limits
                    and error is not None
                    and error <= self.INIT_POS_ALIGN_THRESHOLD
                )
                if aligned:
                    result["aligned_joints"] += 1

                if not values_valid:
                    direction = "missing"
                elif not in_limits:
                    if float(exo_value) < lo:
                        direction = "increase_to_limit"
                    else:
                        direction = "decrease_to_limit"
                elif aligned:
                    direction = "hold"
                elif target_delta > 0:
                    direction = "increase"
                else:
                    direction = "decrease"

                joint_rows.append(
                    {
                        "joint": index + 1,
                        "name": limit_key,
                        "exo": round(float(exo_value), 4)
                        if isinstance(exo_value, (int, float))
                        and math.isfinite(float(exo_value))
                        else None,
                        "robot": round(float(robot_value), 4)
                        if isinstance(robot_value, (int, float))
                        and math.isfinite(float(robot_value))
                        else None,
                        "target_delta": round(target_delta, 4)
                        if target_delta is not None
                        else None,
                        "error": round(error, 4) if error is not None else None,
                        "limit_min": lo,
                        "limit_max": hi,
                        "in_limits": bool(in_limits),
                        "aligned": aligned,
                        "direction": direction,
                    }
                )

            result["arms"][side] = {
                "exo_count": len(exo_joints),
                "robot_count": len(robot_joints),
                "joints": joint_rows,
            }

        if result["reason"] not in {"exo_data_missing", "robot_data_missing"} and any(
            arm["exo_count"] < 7 or arm["robot_count"] < 7
            for arm in result["arms"].values()
        ):
            result["reason"] = (
                "exo_data_incomplete"
                if any(arm["exo_count"] < 7 for arm in result["arms"].values())
                else "robot_data_incomplete"
            )
        elif result["aligned_joints"] == result["total_joints"]:
            result["aligned"] = True
            result["reason"] = "aligned"

        return result

    def _get_out_of_limit_details(
        self, joints: List[float], is_left: bool
    ) -> List[str]:
        """Human-readable descriptions of joints that either fall outside
        hardware limits OR deviate from the robot's current joints by more
        than ``INIT_POS_ALIGN_THRESHOLD`` (both drive ALIGN).

        Args:
            joints: List of 7 exo joint angles in radians.
            is_left: True for left arm, False for right arm.
        """
        out: List[str] = []
        robot_pos = self._robot_left_joints if is_left else self._robot_right_joints
        for i, key in enumerate(self._get_limit_keys(is_left)):
            lo, hi = self.JOINT_LIMITS[key]
            if not (lo <= joints[i] <= hi):
                out.append(
                    f"{key}: {joints[i]:.3f} rad (limit: {lo:.3f} to {hi:.3f})"
                )
                continue
            if robot_pos and i < len(robot_pos):
                diff = abs(joints[i] - robot_pos[i])
                if diff > self.INIT_POS_ALIGN_THRESHOLD:
                    out.append(
                        f"{key}: {joints[i]:.3f} rad — off robot pos "
                        f"({robot_pos[i]:.3f}) by {diff:.3f} rad "
                        f"(threshold {self.INIT_POS_ALIGN_THRESHOLD:.3f})"
                    )
        return out

    def _check_arm_joints_within_limits(
        self, joints: List[float], is_left: bool
    ) -> bool:
        """Checks whether all joints of one arm are within hardware limits.

        **Strict**: missing/short data returns False (drives ALIGN). This
        matches the validity check in ``_check_exo_joints_within_limits``
        so a short exo payload can never latch alignment prematurely.
        """
        if not joints or len(joints) < 7:
            return False

        for i, key in enumerate(self._get_limit_keys(is_left)):
            lo, hi = self.JOINT_LIMITS[key]
            if not (lo <= joints[i] <= hi):
                return False

        return True

    def _get_limit_keys(self, is_left: bool) -> List[str]:
        """Returns the ordered ``JOINT_LIMITS`` keys for one arm.

        Args:
            is_left: True for left arm, False for right arm.

        Returns:
            List of 7 joint limit key strings.
        """
        side = "left" if is_left else "right"
        return [
            "arm_j1",
            f"arm_j2_{side}",
            "arm_j3",
            "arm_j4",
            "arm_j5",
            "arm_j6",
            f"arm_j7_{side}",
        ]

    def get_latest_exo_joints(self) -> Optional[Dict]:
        """Returns the most recently received exo joint angles without blocking.

        Unlike ``_get_exo_joint_angles``, this does not wait for a new message
        and does not touch the threading event, so it never interferes with the
        ALIGN state check.

        Returns:
            Dictionary with keys ``"left"`` and ``"right"`` containing joint
            angle lists, or None if no data has been received yet.
        """
        data = self._exo_data.get("data")
        if not data:
            return None
        return {
            "left": data.get("left_arm_pos", []),
            "right": data.get("right_arm_pos", []),
        }

    def _get_exo_joint_angles(self, timeout: float = 0.5) -> Optional[Dict]:
        """Reads the latest exoskeleton joint angles from the subscriber.

        Args:
            timeout: Seconds to wait for a new message.

        Returns:
            Dictionary with keys ``"left"`` and ``"right"`` containing joint
            angle lists, or None if data is unavailable.
        """
        if not self._exo_subscriber:
            return None

        try:
            self._exo_data_event.clear()

            got_data = self._exo_data_event.wait(timeout=timeout)
            if got_data:
                data = self._exo_data["data"]
                if data:
                    return {
                        "left": data.get("left_arm_pos", []),
                        "right": data.get("right_arm_pos", []),
                    }
                return None
            return None
        except Exception:
            return None


def main() -> None:
    """CLI entry point for manual state checker diagnostics."""
    robot_name = os.environ.get("ROBOT_NAME", "")
    jetson_ip = os.environ.get("JETSON_IP", "192.168.50.20")

    print(f"ROBOT_NAME: {robot_name if robot_name else 'not set'}")
    print(f"JETSON_IP:  {jetson_ip}")

    checker = StateChecker(robot_name, jetson_ip)

    try:
        while True:
            t0 = time.time()
            state = checker.get_state()
            elapsed = time.time() - t0
            print(f"Robot State: {state} (took {elapsed:.3f}s)")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        checker.cleanup()


if __name__ == "__main__":
    main()
