"""MCAPRecorder: standalone recorder writing dexdata-composable episodes.

Sibling to MDPRecorder — inherits control modes and lifecycle from
BaseRecorder. Storage is via dexdata's spec-driven ``Writer`` (one MCAP
channel per ``Spec`` topic, one ``write(topic, value, publish_ts, recv_ts)``
call per signal per tick).

The recorder owns three short pieces of work:

1. **Init** — load the embodiment Spec, resolve which topics the YAML
   wants, drop topics whose hardware is missing on the live robot.
2. **Collect** — for each :data:`STATE_SENSORS` row whose hardware is
   present, fetch one Reading per enabled topic.
3. **Write** — pump each Reading (state from the sensors, action from
   :data:`ACTION_PRODUCERS`) into the Writer.

All robot-API ↔ spec-topic translation lives in :mod:`omniteleop.record.sensors`.
This file no longer contains hardcoded body-part lists.
"""

from __future__ import annotations

import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tyro
from loguru import logger

from dexcontrol.core.config import get_robot_config

from dexdata.handlers.containers import CompressedVideoSpec, DepthImageSpec
from dexdata.mcap_utils import Writer
from dexdata.metadata import (
    SCHEMA_VERSION,
    Collection,
    Data,
    DataClass,
    EpisodeMetadata,
    Robot as EpisodeRobot,
    Scene,
    Source,
    Task,
    Version,
    new_episode_id,
    now_iso,
)
from dexdata.spec import Spec, load_spec, restrict
from omniteleop.common.logging import setup_logging
from omniteleop.read_only_robot import ReadOnlyRobot
from omniteleop.record.base_recorder import BaseRecorder
from omniteleop.record.component_map import prune_to_spec, resolve_topics
from omniteleop.record.sensors import (
    ACTION_PRODUCERS,
    Reading,
    StateSensor,
    state_sensors,
)


class MCAPRecorder(BaseRecorder):
    """Standalone recorder writing dexdata-composable episodes."""

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        record_mode: str = "joycon",
        show_rerun: bool = False,
    ):
        super().__init__(
            namespace=namespace, debug=debug,
            record_mode=record_mode, show_rerun=show_rerun,
        )
        self.robot: Optional[ReadOnlyRobot] = None
        self._writer: Optional[Writer] = None
        self._spec: Optional[Spec] = None
        self._episode_metadata: Optional[EpisodeMetadata] = None
        self._state_sensors: Tuple[StateSensor, ...] = ()

        recorder_cfg = self.config.get("recorder", {}) or {}
        depth_cfg = recorder_cfg.get("depth", {}) or {}
        self._depth_min_range = float(depth_cfg.get("min_range", 0.1))
        self._depth_max_range = float(depth_cfg.get("max_range", 8.0))
        wrist_cfg = recorder_cfg.get("wrist_camera_adapter", {}) or {}
        self._wrist_startup_timeout_s = float(wrist_cfg.get("startup_timeout_s", 5.0))
        self._wrist_stale_timeout_s = float(
            wrist_cfg.get("stale_frame_timeout_s", 0.25)
        )
        # image_resolution is (width, height); cv2.resize takes (w, h).
        self._video_width, self._video_height = self.image_resolution

    def _node_name(self) -> str:
        return "mcap_recorder"

    # ─── Initialization ────────────────────────────────────────────────────

    def initialize(self) -> None:
        super().initialize()

        # 1) Resolve which topics the YAML wants. This is just the toggle
        #    layer — hardware presence is checked after the Robot is built.
        spec_path = self.config.get_data_spec_path()
        if not spec_path.exists():
            raise FileNotFoundError(
                f"dexdata Spec not found: {spec_path}\n"
                f"Set recorder.data_spec_id or recorder.data_spec_path in the YAML, "
                f"or extend dexdata/src/dexdata/specs/embodiment/ with this spec."
            )
        full_spec = load_spec(spec_path)
        #    A toggle the embodiment can't satisfy (e.g. left_hand on a
        #    vega_1u, which has no end effector) is switched off and reported
        #    rather than aborting startup.
        adjusted, dropped = prune_to_spec(full_spec, self.record_components)
        if dropped:
            logger.warning(
                f"spec {full_spec.spec_id!r} declares no topics for "
                f"{', '.join(dropped)} — this robot has no such hardware, "
                f"not recording them"
            )
            self.record_components.update(adjusted)
        wanted_topics = resolve_topics(full_spec, self.record_components)

        # 2) Build sensor descriptors, then flip dexcontrol's per-sensor
        #    `enabled` flags for every sensor producing a wanted topic.
        self._state_sensors = state_sensors(
            self.image_resolution,
            wrist_stale_timeout_s=self._wrist_stale_timeout_s,
        )
        robot_configs = get_robot_config()
        for sensor in self._state_sensors:
            if sensor.topics & wanted_topics:
                sensor.enable_in_config(robot_configs)

        self.robot = ReadOnlyRobot(configs=robot_configs)

        # 3) Reconcile the toggles against the live robot, then drop topics
        #    whose hardware is missing on this specific robot.
        self._drop_unavailable_components(self.robot)
        missing: set[str] = set()
        for sensor in self._state_sensors:
            relevant = sensor.topics & wanted_topics
            if relevant and not sensor.is_present(self.robot):
                logger.warning(
                    f"{type(sensor).__name__} not present on robot — "
                    f"dropping {sorted(relevant)}"
                )
                missing |= relevant
        self._spec = restrict(full_spec, wanted_topics - missing)

        logger.info(
            f"Loaded spec {self._spec.spec_id}@{self._spec.spec_version} "
            f"({len(self._spec.signals)}/{len(full_spec.signals)} signals after filter)"
        )

        # 4) Wait for any head-camera stream we plan to read.
        head_topics = {
            "/camera/head_left/rgb/video",
            "/camera/head_right/rgb/video",
            "/camera/head_left/depth",
        }
        if head_topics & set(self._spec.topics()):
            logger.info("Waiting for camera streams...")
            if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
                logger.info("Camera streams active")
            else:
                logger.warning("Some camera streams may not be active")

        for side in ("left", "right"):
            component = f"{side}_wrist_rgb"
            if not self.record_components.get(component, False):
                continue
            camera = getattr(self.robot.sensors, f"{side}_wrist_camera")
            logger.info(f"Waiting for {side} wrist camera stream...")
            if not camera.wait_for_active(timeout=self._wrist_startup_timeout_s):
                raise RuntimeError(
                    f"{side} wrist camera produced no frame within "
                    f"{self._wrist_startup_timeout_s:.1f}s"
                )
            value = camera.get_obs(include_timestamp=True)
            if not isinstance(value, dict) or not value.get("timestamp_ns"):
                raise RuntimeError(f"{side} wrist camera returned no source timestamp")
            logger.info(f"{side.capitalize()} wrist camera stream active")

    # ─── Storage hooks ─────────────────────────────────────────────────────

    def _build_episode_metadata(self, metadata: Dict[str, Any]) -> EpisodeMetadata:
        """Build the EpisodeMetadata sidecar from request kwargs + config.

        Derivable fields (duration_s, length, size_bytes, checksum, uri,
        topics) are filled by Writer at close(); we leave placeholders.
        """
        return EpisodeMetadata(
            version=Version(
                episode_id=new_episode_id(),
                schema_version=SCHEMA_VERSION,
                created_at=now_iso(),
                source=Source.ROBOT_TELEOP,
            ),
            collection=Collection(
                duration_s=0.0,
                record_hz=int(self.record_rate),
                length=0,
                operator_id=metadata.get("operator", "unknown"),
                data_class=DataClass.EXPERT_DEMO,
                success=True,
            ),
            robot=EpisodeRobot(
                embodiment=self._spec.spec_id if self._spec else None,
            ),
            task=Task(
                task_id=metadata.get("task_label") or None,
                language=metadata.get("task_instruction", ""),
            ),
            scene=Scene(),
            data=Data(uri="episode.mcap"),
        )

    def _setup_storage(self, metadata: Dict[str, Any]) -> None:
        if self._spec is None:
            raise RuntimeError("MCAPRecorder.initialize() must run before recording")
        exchange_dir = Path(
            self.config.get("recorder", {}).get("save_dir") or "~/recordings"
        ).expanduser()
        if not exchange_dir.exists():
            exchange_dir = self.save_dir
        # Group episodes by calendar day: <base>/MM-DD-YYYY/<prefix>_<ts>_<suffix>.
        # One datetime.now() drives both the day folder and the timestamp so they
        # can't disagree across a midnight rollover.
        now = datetime.now()
        day_dir = exchange_dir / now.strftime("%m-%d-%Y")
        day_dir.mkdir(parents=True, exist_ok=True)
        ts = now.strftime("%Y%m%d_%H%M%S")
        suffix = random.randint(100000, 999999)
        self.episode_dir = day_dir / f"{self.episode_prefix}_{ts}_{suffix}"

        self._episode_metadata = self._build_episode_metadata(metadata)
        length_topic = (
            "/robot/state/left_arm/qpos"
            if "/robot/state/left_arm/qpos" in self._spec.by_topic()
            else None
        )
        self._writer = Writer(
            self.episode_dir,
            self._spec,
            metadata=self._episode_metadata,
            length_topic=length_topic,
        )

        for signal in self._spec.signals:
            if isinstance(signal.container, CompressedVideoSpec):
                self._writer.set_runtime_metadata(
                    signal.topic, width=self._video_width, height=self._video_height,
                )
            elif isinstance(signal.container, DepthImageSpec):
                self._writer.set_runtime_metadata(
                    signal.topic, min_range=self._depth_min_range, max_range=self._depth_max_range,
                )

    def _finalize_storage(self, success: bool) -> None:
        if self._writer is None:
            return
        if not success and self._episode_metadata is not None:
            self._episode_metadata.collection.success = False
        try:
            self._writer.close()
        except Exception as e:
            logger.error(f"Failed to close Writer: {e}")
        finally:
            self._writer = None
            self._episode_metadata = None
        if not success and self.episode_dir and self.episode_dir.exists():
            try:
                shutil.rmtree(self.episode_dir)
                logger.info(f"Deleted episode directory: {self.episode_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to delete episode directory: {e}")

    # ─── Per-tick collect + write ──────────────────────────────────────────

    def _collect_observation(self) -> Dict[str, Any]:
        """Fetch one Reading per enabled state topic from the live robot.

        BaseRecorder's action-fallback path expects ``obs["joint_pos"]`` to
        be a component-name-keyed dict, so we derive that view from the
        Readings rather than maintaining a parallel structure.
        """
        readings: list[Reading] = []
        if self.robot is not None and self._spec is not None:
            keep = set(self._spec.topics())
            for sensor in self._state_sensors:
                if not sensor.is_present(self.robot):
                    continue
                readings.extend(sensor.collect(self.robot, keep))

        joint_pos: Dict[str, list[float]] = {}
        for r in readings:
            if r.topic.startswith("/robot/state/") and r.topic.endswith("/qpos"):
                comp = r.topic[len("/robot/state/") : -len("/qpos")]
                joint_pos[comp] = r.value.tolist()
        return {"joint_pos": joint_pos, "readings": readings}

    def _write_frame(
        self,
        timestamp_ns: int,
        resolved_action: Dict[str, Any],
        observation: Dict[str, Any],
        safety_flags: Dict[str, Any],
    ) -> None:
        if self._writer is None or self._spec is None:
            return
        keep = set(self._spec.topics())

        for r in observation.get("readings", ()):
            self._writer.write(r.topic, r.value, r.publish_ts_ns, timestamp_ns)

        for producer in ACTION_PRODUCERS:
            for r in producer.collect(resolved_action, keep):
                self._writer.write(r.topic, r.value, timestamp_ns, timestamp_ns)

    def cleanup(self) -> None:
        super().cleanup()
        if self.robot:
            self.robot.shutdown()
            self.robot = None


def main(
    namespace: str = "",
    debug: bool = False,
    record_mode: str = "joycon",
    show_rerun: bool = False,
) -> int:
    """Entry point for MCAPRecorder CLI."""
    log = setup_logging(debug)
    log.info(f"Starting MCAPRecorder (mode={record_mode})")

    recorder = MCAPRecorder(
        namespace=namespace, debug=debug,
        record_mode=record_mode, show_rerun=show_rerun,
    )
    try:
        recorder.initialize()
        log.info("RECORDER READY")
        recorder.run()
    except Exception as e:
        log.error(f"Recorder error: {e}")
        return 1
    finally:
        recorder.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
