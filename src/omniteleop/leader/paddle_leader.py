"""VR paddle leader — subscribes to dexstream vr/controllers, runs delta-based
IK and gripper control via RealRobotInterventionController, and publishes
robot_commands directly for robot_controller.py to execute.

Structurally mirrors `internal/vr/examples/relative_viz_controller.py`: a
thin loop around an `InterventionController` that pumps IK per tick. The
only substantive differences from that viz script are:

  1. Uses `RealRobotInterventionController` (drives a live `Robot` for
     FK/IK + joint feedback) instead of a viz-only `BaseIKController`.
  2. Publishes the per-tick joint output on the `robot_commands` Zenoh
     topic so `robot_controller.py` can execute it on real hardware.

Software e-stop gesture is implemented locally (see `_update_estop_gesture`):
holding both thumbstick_click buttons simultaneously for 1.0s toggles
`self._estop_active`, which is published on every tick via
`safety_flags.emergency_stop`. No dependency on the follower-side
`VRController` / `ButtonManager` / component routing — those are
deliberately absent from this leader because they belong to the old
VRHandler architecture that was retired alongside vr_reader.py.

See docs/superpowers/specs/2026-04-10-vr-paddle-leader-design.md.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter

from omniteleop.common import get_config

# Paddle-mode IK stack (lives under internal/vr).
from omniteleop.leader.vr.controllers import get_hand_poses
from omniteleop.leader.vr.controllers.intervention import RealRobotInterventionController
from omniteleop.leader.vr.trackers.pose import install_reverse_parity

# Tunables.
_NODE_NAME = "paddle_leader"
_BOOTSTRAP_POLL_INTERVAL_S = 0.05
_BOOTSTRAP_TIMEOUT_S = 10.0

# Software-estop gesture timings. Match the shape of the combo that used
# to live inside VRController's ButtonManager (both thumbstick_click held
# for 1.0s with a 0.1s grace period for finger wobble).
_ESTOP_HOLD_DURATION_S = 1.0
_ESTOP_GRACE_PERIOD_S = 0.1


def build_paddle_payload(
    intervention: tuple[
        bool,
        np.ndarray | None,
        np.ndarray | None,
        list | None,
        list | None,
    ],
    hold_components: dict[str, dict],
    estop_active: bool,
    timestamp_ns: int,
    torso_qpos: np.ndarray | None = None,
) -> dict:
    """Produce the dict published on robot_commands for a single tick.

    Two sources for the components dict, gated by the tracking bool:
      - Tracking (any side active): publish the intervention tuple's
        non-None arm/hand entries plus ``torso_qpos`` (if provided) as
        the joint targets. The solver returns held qpos for the inactive
        side based on the motion manager's joint state. Correctness here
        depends on the caller syncing the MM from live joint feedback
        every non-tracking tick so it is fresh when tracking starts —
        without that discipline the inactive arm's qpos is stale and the
        robot drifts.
      - Not tracking: publish `hold_components`, which is a snapshot
        of the current joint feedback (including torso if present).

    Torso is emitted with ``"mode": "absolute"`` because the follower's
    ``TorsoProcessor`` defaults to RELATIVE (delta IK), which is the wrong
    interpretation for already-solved joint positions.

    `intervention` is the tuple returned by
    `RealRobotInterventionController._get_intervention_qpos()`:
    `(tracking, left_arm, right_arm, left_hand, right_hand)`. Torso is
    passed separately via ``torso_qpos`` (read from
    :meth:`InterventionController.get_torso_qpos`) so the tuple stays
    backward-compatible with the other call sites that share this entry
    point. None when the TORSO region is locked.

    `hold_components` is a dict like
    `{"left_arm": {"pos": [...]}, "right_arm": {"pos": [...]}, "torso": {...}}`
    built from the most recent `robot_joints` feedback. May be empty if
    feedback hasn't arrived yet.
    """
    safety_flags = {
        "emergency_stop": bool(estop_active),
        "exit_requested": False,
    }

    if estop_active:
        return {
            "timestamp_ns": timestamp_ns,
            "components": {},
            "safety_flags": safety_flags,
        }

    components: dict[str, Any] = {}
    tracking, l_arm, r_arm, l_hand, r_hand = intervention

    if tracking:
        if l_arm is not None:
            components["left_arm"] = {"pos": l_arm.tolist()}
        if r_arm is not None:
            components["right_arm"] = {"pos": r_arm.tolist()}
        if l_hand is not None:
            components["left_hand"] = {"pos": l_hand}
        if r_hand is not None:
            components["right_hand"] = {"pos": r_hand}
        if torso_qpos is not None:
            components["torso"] = {
                "pos": torso_qpos.tolist(),
                "mode": "absolute",
            }
    else:
        components.update(hold_components)

    return {
        "timestamp_ns": timestamp_ns,
        "components": components,
        "safety_flags": safety_flags,
    }


class PaddleLeader:
    """Single-process VR paddle leader.

    Owns a `RealRobotInterventionController` (which internally owns an
    `InterventionTracker` subscribed to dexstream's `vr/controllers` and a
    `RealRobotIKController` with its own FK/IK `Robot` instance). Each
    tick, pulls `(tracking, arms, hands)` from `_get_intervention_qpos()`,
    updates the local software-estop gesture state from the tracker's VR
    subscriber, and publishes `robot_commands` for `robot_controller.py`
    to execute.

    The leader also subscribes directly to `robot_joints` so it can seed
    the intervention controller's motion manager with live joint state on
    startup and periodically re-sync during the loop. Without that seed,
    IK deltas are computed against a stale origin and the first
    clutch-in snaps the arms.
    """

    def __init__(
        self,
        namespace: str = "",
        publish_rate: int | None = None,
        debug: bool = False,
        visualize: bool = False,
        config_name: str | None = None,
        reverse_parity: bool = False,
        custom_urdf: str | None = None,
        unlock_torso: bool = False,
        torso_damping: float | None = None,
    ) -> None:
        self.debug = debug
        self.namespace = namespace
        self.reverse_parity = reverse_parity
        self.custom_urdf = custom_urdf
        self.unlock_torso = unlock_torso
        # `visualize` doubles as "viz-only" mode: no real Robot, no command
        # publishing, no joint-feedback subscription. The SAPIEN visualizer
        # reflects IK output against MM's config-default initial state.
        self.viz_only = visualize

        # Only external call in __init__ — matches the other leaders in this repo.
        self.node = Node(name=_NODE_NAME, namespace=namespace)

        self.config = (
            get_config() if config_name is None else get_config(config_name)
        )

        topics = self.config.get("topics", {})
        self.vr_topic = topics.get("vr_controllers", "vr/controllers")
        self.cmd_topic = topics.get("robot_commands", "robot_commands")
        self.joints_topic = topics.get("robot_joints", "robot_joints")

        rates = self.config.get("rates", {})
        resolved_rate = (
            int(publish_rate)
            if publish_rate is not None
            else int(rates.get("input_rate", 40))
        )
        self.publish_rate = resolved_rate
        self.sync_interval = float(rates.get("sync_interval", 1.0))
        self.rate_limiter = RateLimiter(resolved_rate)

        # Paddle-mode IK stack. Internally owns a VRSubscriber on the same
        # vr/controllers topic and a Robot instance used for FK/IK only.
        hand_open, hand_close = self._load_hand_poses()
        mm_kwargs: dict = {}
        if custom_urdf:
            mm_kwargs["custom_urdf_path"] = Path(custom_urdf)
        if torso_damping is not None:
            mm_kwargs["custom_local_ik_config"] = self._build_ik_config(torso_damping)
        self.intervention_controller = RealRobotInterventionController(
            topic=self.vr_topic,
            rate=resolved_rate,
            visualize=visualize,
            hand_open_poses=hand_open,
            hand_close_poses=hand_close,
            mm_kwargs=mm_kwargs or None,
            unlock_torso=unlock_torso,
            viz_only=self.viz_only,
        )

        if self.reverse_parity:
            install_reverse_parity(self.intervention_controller.tracker)

        # Software e-stop gesture state. Toggled by holding both
        # thumbstick_click buttons simultaneously for _ESTOP_HOLD_DURATION_S.
        # Independent of `dexcontrol.Robot.estop` (which is hardware); this
        # flag just drives `safety_flags.emergency_stop` on the outgoing
        # robot_commands payload.
        self._estop_active: bool = False
        self._estop_combo_start_time: float | None = None
        self._estop_combo_last_both_time: float | None = None
        self._estop_combo_fired: bool = False

        # Joint feedback, guarded by _data_lock.
        self._data_lock = threading.RLock()
        self._latest_joint_feedback: dict | None = None

        # Publisher + subscriber are created in initialize(), so __init__
        # is side-effect-free on its comms surface beyond Node().
        self.command_pub = None
        self._joints_sub = None
        self._last_sync_time: float = 0.0

        logger.info(
            f"PaddleLeader constructed (vr_topic={self.vr_topic}, "
            f"cmd_topic={self.cmd_topic}, rate={resolved_rate})"
        )

    @staticmethod
    def _build_ik_config(torso_damping: float):
        """Build a LocalPinkIKConfig with overridden torso damping weights.

        Defaults from `IKDampingWeightsConfig` use ``3e4`` for torso joints,
        which heavily penalizes torso motion. Lowering this lets the IK solver
        recruit torso joints — useful in tandem with ``unlock_torso=True``.
        Arm-j2 overrides (``L_arm_j2``/``R_arm_j2`` at 100) are preserved.
        """
        from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkIKConfig

        damping = IKDampingWeightsConfig(
            default=1e-12,
            override={
                "torso_j1": float(torso_damping),
                "torso_j2": float(torso_damping),
                "torso_j3": float(torso_damping),
            },
        )
        return LocalPinkIKConfig(damping_weights=damping)

    def _load_hand_poses(self) -> tuple[dict | None, dict | None]:
        """Return (open_poses, close_poses) or (None, None) if unavailable.

        `get_hand_poses()` returns either a (open, close) tuple or None;
        this wrapper normalizes to a 2-tuple so callers can always unpack.
        """
        try:
            result = get_hand_poses(self.config)
        except Exception as e:  # noqa: BLE001 — hand control is optional
            logger.warning(f"Hand poses unavailable ({e}); gripper control disabled")
            return None, None
        if result is None:
            logger.info("Hand poses not configured; gripper control disabled")
            return None, None
        return result

    def initialize(self) -> bool:
        """Create the joint feedback subscriber and the command publisher.

        Non-blocking. Returns True on success, False on any initialization
        failure (with logging). Called once before `run()`.

        In ``viz_only`` mode, no Zenoh I/O is set up — the loop runs purely
        against MotionManager's internal state.
        """
        if self.viz_only:
            logger.info("viz_only mode: skipping joint-feedback subscription and command publisher")
            return True
        try:
            self._joints_sub = self.node.create_subscriber(
                self.joints_topic,
                self._on_joint_feedback,
                decoder=DictDataCodec.decode,
            )
            self.command_pub = self.node.create_publisher(
                self.cmd_topic,
                encoder=DictDataCodec.encode,
            )
            logger.info(
                f"Subscribed to {self.joints_topic}; "
                f"publishing to {self.cmd_topic}"
            )
            return True
        except Exception as e:  # noqa: BLE001 — initialize is a supervisor boundary
            logger.error(f"PaddleLeader initialize failed: {e}")
            return False

    def _on_joint_feedback(self, data: dict) -> None:
        """Zenoh callback: store latest joint feedback under lock."""
        with self._data_lock:
            self._latest_joint_feedback = data

    def _build_hold_components(self) -> dict[str, dict]:
        """Snapshot current joint feedback as per-side hold components.

        Returns a dict shaped like
        `{"left_arm": {"pos": [...]}, "right_arm": {"pos": [...]}, "torso": {...}}`
        that can be folded into a `robot_commands` payload as the idle hold
        target. Any component missing from the feedback dict is omitted.

        Torso is emitted with ``"mode": "absolute"`` because the follower's
        TorsoProcessor defaults to RELATIVE; the feedback values are absolute
        joint positions, not deltas.

        Continuous publishing of these hold targets (every tick, for
        idle sides) matches `arm_reader`'s always-publish pattern and
        stops the robot from coasting on a stale IK trajectory after
        the operator releases the trigger.
        """
        with self._data_lock:
            jf = self._latest_joint_feedback
        if jf is None:
            return {}
        joints = jf.get("joints", {})
        hold: dict[str, dict] = {}
        for key in ("left_arm", "right_arm"):
            pos = joints.get(key)
            if pos:
                hold[key] = {"pos": list(pos)}
        torso_pos = joints.get("torso")
        if torso_pos:
            hold["torso"] = {"pos": list(torso_pos), "mode": "absolute"}
        return hold

    def _bootstrap_ready(self) -> bool:
        """True once at least one joint-feedback frame has been received.

        VR data is not awaited here because the intervention tracker owns
        its own VRSubscriber and the hot loop gracefully handles "no VR
        yet" (the tracker returns `tracking=False`, and we publish an
        empty-components hold). Joint feedback is different — without it
        the first IK solve would be against a stale motion-manager origin.
        """
        with self._data_lock:
            return self._latest_joint_feedback is not None

    def _sync_intervention_from_feedback(self) -> None:
        """Push latest joint positions into the intervention controller's
        motion manager. Caller is responsible for ensuring feedback exists.

        If either arm key is absent or empty in the feedback, skip the
        sync entirely — passing a zero-length array would corrupt the IK
        seed. Hand arrays are independently optional.
        """
        with self._data_lock:
            jf = self._latest_joint_feedback
            if jf is None:
                return
            joints = jf.get("joints", {})
            left_arm_raw = joints.get("left_arm")
            right_arm_raw = joints.get("right_arm")
            left_hand_list = joints.get("left_hand")
            right_hand_list = joints.get("right_hand")
            torso_list = joints.get("torso")

        if not left_arm_raw or not right_arm_raw:
            logger.debug(
                "Skipping intervention sync: joint feedback missing left_arm "
                "or right_arm (got left=%s right=%s)",
                left_arm_raw,
                right_arm_raw,
            )
            return

        left_arm = np.asarray(left_arm_raw, dtype=float)
        right_arm = np.asarray(right_arm_raw, dtype=float)
        left_hand = (
            np.asarray(left_hand_list, dtype=float)
            if left_hand_list is not None
            else None
        )
        right_hand = (
            np.asarray(right_hand_list, dtype=float)
            if right_hand_list is not None
            else None
        )
        torso = (
            np.asarray(torso_list, dtype=float)
            if torso_list
            else None
        )

        self.intervention_controller.set_joint_state(
            left_arm_qpos=left_arm,
            right_arm_qpos=right_arm,
            left_hand_qpos=left_hand,
            right_hand_qpos=right_hand,
            torso_qpos=torso,
        )
        self._last_sync_time = time.monotonic()

    def _bootstrap(self, timeout_s: float = _BOOTSTRAP_TIMEOUT_S) -> None:
        """Block until joint feedback has arrived, then seed the
        intervention controller. On timeout, log a warning and return
        early — the hot loop will treat missing feedback as "hold" via
        `build_paddle_payload`'s empty-components path.

        In ``viz_only`` mode, there is no joint feedback to await — the MM is
        already seeded from its config defaults inside MotionManager.
        """
        if self.viz_only:
            return
        deadline = time.monotonic() + float(timeout_s)
        while not self._bootstrap_ready():
            if time.monotonic() >= deadline:
                logger.warning(
                    f"PaddleLeader bootstrap timeout after {timeout_s}s "
                    "— entering hot loop without initial joint sync"
                )
                return
            time.sleep(_BOOTSTRAP_POLL_INTERVAL_S)

        logger.info("PaddleLeader bootstrap ready — syncing intervention controller")
        self._sync_intervention_from_feedback()

    def _update_estop_gesture(
        self,
        left_click: bool,
        right_click: bool,
        now: float,
    ) -> None:
        """Tick the software-estop combo state machine.

        Rule: both `thumbstick_click` held simultaneously for
        `_ESTOP_HOLD_DURATION_S` toggles `self._estop_active`. Briefly
        releasing within `_ESTOP_GRACE_PERIOD_S` does not abort a hold in
        progress — allows for small finger wobble. Once a hold has fired
        (`_estop_combo_fired`), it does not re-fire until both clicks are
        released past the grace period and re-pressed.

        Takes an explicit `now` so tests can inject time deterministically.

        Args:
            left_click: Current debounced state of the left thumbstick click.
            right_click: Current debounced state of the right thumbstick click.
            now: Monotonic timestamp in seconds (e.g., `time.monotonic()`).
        """
        both_clicked = bool(left_click) and bool(right_click)

        if both_clicked:
            self._estop_combo_last_both_time = now
            if self._estop_combo_start_time is None:
                self._estop_combo_start_time = now
                self._estop_combo_fired = False

            if not self._estop_combo_fired:
                held_for = now - self._estop_combo_start_time
                if held_for >= _ESTOP_HOLD_DURATION_S:
                    self._estop_combo_fired = True
                    self._estop_active = not self._estop_active
                    state = "ACTIVATED" if self._estop_active else "DEACTIVATED"
                    logger.warning(
                        f"Software e-stop {state} (double-thumbstick hold)"
                    )
            return

        # Not both clicked — grace period before aborting a hold in progress.
        if self._estop_combo_start_time is None:
            return

        if self._estop_combo_last_both_time is None:
            # Defensive: should be unreachable given the `both_clicked`
            # path always sets last_both_time first.
            self._estop_combo_start_time = None
            self._estop_combo_fired = False
            return

        grace_elapsed = now - self._estop_combo_last_both_time
        if grace_elapsed > _ESTOP_GRACE_PERIOD_S:
            self._estop_combo_start_time = None
            self._estop_combo_last_both_time = None
            self._estop_combo_fired = False

    def _tick(self) -> None:
        """One iteration of the hot loop.

        Exceptions propagate to `run()` by design — do NOT wrap this in a
        broad try/except. A silent failure here would keep publishing
        stale commands indefinitely; a crash propagates to the operator.

        The ordering matches `internal/vr/examples/intervention_deployer.py`
        exactly:

          1. Pull the intervention tuple (uses current MM state; on the
             first tracking frame this captures the reference pose via
             FK from the MM).
          2. Update the software-estop gesture state from the VR
             thumbstick clicks.
          3. Sync the MM from live joint feedback:
             - Every tick when NOT tracking, so the MM is fresh when
               tracking starts next. This is the critical invariant.
               The reference-pose capture uses MM FK; a stale MM means
               a wrong delta origin, which manifests as clutch-in
               jumps and post-release "momentum".
             - Every `sync_interval` during tracking to correct IK
               seed drift without per-frame round-trip latency.
          4. Publish: while tracking, forward the intervention tuple
             as-is (the solver returns held qpos for the inactive
             side, which is correct because the MM was fresh up until
             the moment tracking began). While not tracking, publish
             the latest joint feedback as a hold target — same as
             intervention_deployer's fake policy pattern, minus the
             oscillation.
        """
        now_ns = time.time_ns()
        now_mono = time.monotonic()

        # NOTE: Calling a convention-private method on
        # RealRobotInterventionController. hub_recorder.py and
        # intervention_deployer.py use the same entry point; upstream
        # should promote it to a public name in a follow-up PR. If it
        # moves, update all three call sites.
        intervention_result = (
            self.intervention_controller._get_intervention_qpos()  # noqa: SLF001
        )
        tracking = intervention_result[0]

        # Read thumbstick click from the same VRSubscriber the tracker
        # uses. Reaches past one private access layer to avoid opening a
        # second subscriber on `vr/controllers`; same pattern as
        # internal/vr/examples/relative_viz_controller.py.
        #
        # ControllerState (dexstream.subscribers.vr) nests the click
        # inside a Thumbstick dataclass — the flat `thumbstick_click`
        # key only exists in the raw wire dict, not the typed view.
        tracker = self.intervention_controller.tracker
        vr_sub = tracker._vr_sub  # noqa: SLF001
        self._update_estop_gesture(
            left_click=bool(vr_sub.left.thumbstick.click),
            right_click=bool(vr_sub.right.thumbstick.click),
            now=now_mono,
        )

        # In viz_only mode there is no joint feedback to sync against and no
        # downstream consumer to publish to — IK output is reflected directly
        # in the SAPIEN visualizer via MM.
        if self.viz_only:
            return

        # Sync discipline (see docstring above).
        if not tracking:
            self._sync_intervention_from_feedback()
        elif (now_mono - self._last_sync_time) >= self.sync_interval:
            self._sync_intervention_from_feedback()

        # Hold target for the not-tracking branch.
        hold_components = self._build_hold_components()

        # Read torso qpos out of MM after the IK solve. Returns None when the
        # TORSO region is locked, which leaves the payload's `torso` slot
        # empty — robot_controller then holds the last commanded torso pose.
        torso_qpos = self.intervention_controller.get_torso_qpos()

        payload = build_paddle_payload(
            intervention=intervention_result,
            hold_components=hold_components,
            estop_active=self._estop_active,
            timestamp_ns=now_ns,
            torso_qpos=torso_qpos,
        )
        self.command_pub.publish(payload)

    def run(self) -> None:
        """Main blocking loop. Returns on KeyboardInterrupt."""
        logger.info(f"PaddleLeader starting hot loop at {self.publish_rate} Hz")
        self._bootstrap()

        try:
            while True:
                self._tick()
                self.rate_limiter.sleep()
        except KeyboardInterrupt:
            logger.info("PaddleLeader interrupted by user")

    def cleanup(self) -> None:
        """Release resources cleanly."""
        try:
            self.intervention_controller.cleanup()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"intervention_controller cleanup raised: {e}")
        try:
            self.node.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"node shutdown raised: {e}")
        logger.info("PaddleLeader cleaned up")


def main(
    namespace: str = "",
    publish_rate: int | None = None,
    debug: bool = False,
    visualize: bool = False,
    config_name: str | None = None,
    reverse_parity: bool = False,
    custom_urdf: str | None = None,
    unlock_torso: bool = False,
    torso_damping: float | None = None,
) -> int:
    """Console entry point for `omni-paddle`.

    Args:
        namespace: Optional namespace prefix for Zenoh topics.
        publish_rate: Override the command publish rate (Hz). Defaults to
            config.rates.input_rate.
        debug: Enable debug logging and display.
        visualize: Visualization-only mode. Spawns the SAPIEN MotionManager
            viewer and runs the IK pipeline against virtual MM state — no
            dexcontrol ``Robot`` is constructed, no ``robot_joints``
            subscription, and no ``robot_commands`` publishing. Off by default.
        config_name: Override the robot config name (otherwise uses
            `$ROBOT_CONFIG`).
        reverse_parity: Swap L/R controllers and apply a 180° Z rotation to
            position and orientation deltas so a backward-facing robot can
            be teleoperated as if front-facing. Off by default.
        custom_urdf: Path to a custom URDF forwarded to MotionManager as
            ``custom_urdf_path``. If None, MotionManager uses its default.
        unlock_torso: Leave the TORSO joint region unlocked so the IK solver
            can recruit torso joints to reach the target pose. HEAD and BASE
            remain locked. Off by default.
        torso_damping: Override the per-joint damping weight applied to
            ``torso_j1/j2/j3`` in the local Pink IK solver. The default
            (``3e4``) heavily resists torso motion; lower values let IK use
            the torso more freely. Only meaningful with ``unlock_torso=True``.

    Returns:
        Process exit code. 0 on normal shutdown, 1 if initialization failed.
    """
    from omniteleop.common.logging import setup_logging

    setup_logging(debug)
    logger.info(
        f"Starting PaddleLeader"
        f"{f' (namespace: {namespace})' if namespace else ''}"
    )

    leader = PaddleLeader(
        namespace=namespace,
        publish_rate=publish_rate,
        debug=debug,
        visualize=visualize,
        config_name=config_name,
        reverse_parity=reverse_parity,
        custom_urdf=custom_urdf,
        unlock_torso=unlock_torso,
        torso_damping=torso_damping,
    )

    if not leader.initialize():
        return 1

    try:
        leader.run()
    finally:
        leader.cleanup()

    return 0


if __name__ == "__main__":
    import sys

    import tyro

    sys.exit(tyro.cli(main))
