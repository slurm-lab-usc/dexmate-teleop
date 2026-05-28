"""State checker for monitoring robot control states.

Determines the current operational state of the robot by checking network
connectivity, topic availability, process status, and joint positions.
"""

import os
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

from loguru import logger


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

    # Threshold (radians) between exo and config init_pos. Must stay in sync
    # with CommandProcessor._align_threshold — CommandProcessor gates motion
    # on this criterion, and StateChecker reports ALIGN whenever the gate is
    # engaged so the UI shows "Aligning" while the robot is actually blocked.
    INIT_POS_ALIGN_THRESHOLD: float = 1.0

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
        self._exo_data: Dict[str, Optional[dict]] = {"data": None}
        self._exo_data_event = threading.Event()

        # Component query interface and robot instance.
        self._bot = None
        self._query_interface = None

        # Latest robot joint positions, updated externally via update_robot_joints().
        self._robot_left_joints: List[float] = []
        self._robot_right_joints: List[float] = []

        # Diagnosis details populated on each call to _check_active_conditions.
        self._diagnosis_details: dict = {}

        # Init positions loaded from the same config CommandProcessor reads.
        # Used by the init_pos ALIGN criterion — if any exo joint drifts more
        # than INIT_POS_ALIGN_THRESHOLD from init_pos, CommandProcessor will
        # refuse to start motion, so we report ALIGN.
        try:
            from omniteleop.common import get_config
            _cfg = get_config()
            _recorder = (_cfg.get("recorder") or {})
            _rec_components = (_recorder.get("components") or {})
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

        # Once motion has started for the first time (operator pressed then
        # released estop after initial alignment), ALIGN is bypassed forever
        # for this teleop session. Reset by reset_motion_started() on stop.
        self._motion_started_latch: bool = False

        self._init_exo_subscriber()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _init_exo_subscriber(self) -> None:
        """Initializes the reusable exo joints Zenoh subscriber."""
        try:
            from dexcomm import Node
            from dexcomm.codecs import DictDataCodec

            def on_exo_joints(data: dict) -> None:
                self._exo_data["data"] = data
                self._exo_data_event.set()

            self._node = Node(
                name="state_checker_exo_reader",
                namespace=self.robot_name,
            )
            self._exo_subscriber = self._node.create_subscriber(
                "exo/joints",
                callback=on_exo_joints,
                decoder=DictDataCodec.decode,
            )
        except Exception:
            self._node = None
            self._exo_subscriber = None

    def _init_query_interface(self) -> None:
        """Initializes the reusable robot query interface."""
        try:
            from dexcontrol.core.robot_query_interface import RobotQueryInterface
            from dexcontrol.robot import Robot

            self._bot = Robot()
            self._query_interface = RobotQueryInterface.create()
        except Exception:
            self._bot = None
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
        topic_list = self._get_topic_list()
        self._last_topic_list = topic_list
        if topic_list is None or not self._check_required_topics(topic_list):
            return "BOOT"

        if not self._check_active_conditions(topic_list):
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

    def _get_topic_list(self) -> Optional[str]:
        """Retrieves the list of active topics from dextop.

        Returns:
            Topic list string, or None if the command fails.
        """
        try:
            result = subprocess.run(
                ["dextop", "topic", "list"],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            return result.stdout if result.stdout else None
        except Exception:
            return None

    def _build_required_topics(self) -> List[str]:
        """Builds the fully-qualified required topic names for this robot.

        Branches on ``self.leader_mode``:
          - ``vr_sim``: no required topics — BOOT is trivially satisfied.
            Process checks still apply in DIAGNOSIS.
          - ``exoskeleton`` / ``vr``: full robot-side requirements (arm state,
            heartbeat, head cameras).

        Returns:
            List of fully-qualified topic name strings.
        """
        if self.leader_mode == "vr_sim":
            return []
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
        topic_list = self._get_topic_list()

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
        """Checks if both JoyCon controllers are connected via Bluetooth.

        Returns:
            True if both Joy-Con (L) and Joy-Con (R) are connected.
        """
        try:
            result = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            output = result.stdout
            return "Joy-Con (L)" in output and "Joy-Con (R)" in output
        except Exception:
            return False

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

        Returns:
            True if ``/dev/ttyUSB0`` is readable and writable.
        """
        return os.access("/dev/ttyUSB0", os.R_OK | os.W_OK)

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
            for component_status in status.get("states", {}).values():
                if not isinstance(component_status, dict):
                    continue

                error = component_status.get("error")
                if isinstance(error, dict):
                    if error.get("error_code", 0) != 0:
                        return True
                    if error.get("error_message", ""):
                        return True

                operation = component_status.get("operation")
                if operation is not None and operation == 0:
                    return True

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
        if not self._query_interface:
            return {}, False

        errors: dict = {}
        estop_status = False

        try:
            status = self._query_interface.get_component_status(show=False)
            for component_name, component_status in status.get("states", {}).items():
                if not isinstance(component_status, dict):
                    continue

                component_errors: dict = {}
                has_error = False

                error = component_status.get("error")
                if isinstance(error, dict):
                    error_code = error.get("error_code", 0)
                    error_message = error.get("error_message", "")
                    if error_code != 0 or error_message:
                        component_errors["error_code"] = error_code
                        component_errors["error_message"] = error_message
                        has_error = True

                operation = component_status.get("operation")
                if operation is not None:
                    component_errors["operation"] = operation
                    if operation == 0:
                        has_error = True
                        component_errors.setdefault(
                            "error_message",
                            "Component operation status is 0 (error)",
                        )

                if has_error:
                    errors[component_name] = component_errors

            if self._bot:
                estop_status = self._bot.estop._get_state().get(
                    "software_estop_enabled", False
                )

        except Exception as e:
            return {
                "_query_error": {
                    "error_message": f"Failed to query component status: {e}"
                }
            }, False

        return errors, estop_status

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

        joint_angles = self._get_exo_joint_angles()
        if joint_angles is None:
            return False

        left = joint_angles.get("left") or []
        right = joint_angles.get("right") or []

        if len(left) < 7 or len(right) < 7:
            return False

        left_ok = self._check_arm_joints_within_limits(left, is_left=True)
        right_ok = self._check_arm_joints_within_limits(right, is_left=False)
        if not (left_ok and right_ok):
            return False

        # Compare against robot's current joints, not config init_pos.
        # If robot joints aren't available yet, stay in ALIGN (safe default).
        robot_left = self._robot_left_joints
        robot_right = self._robot_right_joints
        if not robot_left or not robot_right:
            return False

        left_aligned = self._arm_within_threshold(left, robot_left)
        right_aligned = self._arm_within_threshold(right, robot_right)

        return left_aligned and right_aligned

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
        joint_angles = self._get_exo_joint_angles()

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
