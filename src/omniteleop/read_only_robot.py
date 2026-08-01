"""Read-only robot observations for non-control OmniTeleop processes.

The main ``dexcontrol.Robot`` interface intentionally owns command publishers and
mode services.  Recorders, the web backend, and VR IK initialization only need
observations, so constructing a full Robot in those processes creates unnecessary
command-capable owners.  This module exposes the small observation surface those
callers use and never creates a component command publisher.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from dexbot_utils import RobotInfo
from dexbot_utils.configs import BaseRobotConfig
from dexcomm import Node
from dexcomm.codecs import (
    EStopStateCodec,
    JointStateCodec,
    SoftwareEstopCodec,
    WrenchStateCodec,
)
from dexcontrol.core.component import RobotComponent
from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.sensors import Sensors


class ReadOnlyJointComponent(RobotComponent):
    """Joint-state subscriber with no command publisher or mode service."""

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
        state_sub_topic: str,
    ) -> None:
        super().__init__(
            name=f"{name}_observer",
            state_sub_topic=state_sub_topic,
            state_decoder=JointStateCodec.decode,
        )
        self._joint_name = list(robot_info.get_component_joints(name))
        try:
            self._joint_pos_limit = np.asarray(
                robot_info.get_joint_pos_limits(self._joint_name),
                dtype=np.float32,
            )
        except Exception:
            self._joint_pos_limit = None

    @property
    def joint_name(self) -> list[str]:
        return self._joint_name.copy()

    @property
    def joint_pos_limit(self) -> np.ndarray | None:
        if self._joint_pos_limit is None:
            return None
        return self._joint_pos_limit.copy()

    def get_joint_pos(self, joint_id: list[int] | int | None = None) -> np.ndarray:
        values = np.asarray(self._get_state()["pos"], dtype=np.float32)
        return self._select(values, joint_id)

    def get_joint_vel(self, joint_id: list[int] | int | None = None) -> np.ndarray:
        state = self._get_state()
        if "vel" not in state:
            raise ValueError("Joint velocities are not available for this component")
        values = np.asarray(state["vel"], dtype=np.float32)
        return self._select(values, joint_id)

    def get_joint_pos_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, float]:
        values = self.get_joint_pos(joint_id)
        if joint_id is None:
            indices = list(range(len(self._joint_name)))
        elif isinstance(joint_id, int):
            indices = [joint_id]
            values = np.atleast_1d(values)
        else:
            indices = joint_id
        return {
            self._joint_name[index]: float(values[offset])
            for offset, index in enumerate(indices)
        }

    @staticmethod
    def _select(
        values: np.ndarray, joint_id: list[int] | int | None
    ) -> np.ndarray:
        if joint_id is None:
            return values.copy()
        return np.asarray(values[joint_id], dtype=np.float32)


class ReadOnlyWrenchSensor(RobotComponent):
    """Wrist wrench subscriber with no end-effector command interfaces."""

    def __init__(self, name: str, state_sub_topic: str) -> None:
        super().__init__(
            name=f"{name}_observer",
            state_sub_topic=state_sub_topic,
            state_decoder=WrenchStateCodec.decode,
        )

    def get_wrench_state(self) -> np.ndarray:
        return np.asarray(self._get_state()["wrench"], dtype=np.float32)


class ReadOnlyEStop(RobotComponent):
    """Emergency-stop state subscriber without activate/deactivate methods."""

    def __init__(self, name: str, state_sub_topic: str) -> None:
        super().__init__(
            name=f"{name}_observer",
            state_sub_topic=state_sub_topic,
            state_decoder=EStopStateCodec.decode,
        )

    def is_software_estop_enabled(self) -> bool:
        state = self._subscriber.get_latest()
        return bool(state and state.get("software_estop_enabled", False))

    def is_button_pressed(self) -> bool:
        state = self._subscriber.get_latest()
        if not state:
            return False
        return any(
            bool(state.get(key, False))
            for key in (
                "left_base_estop_enabled",
                "right_base_estop_enabled",
                "torso_estop_enabled",
                "remote_estop_enabled",
            )
        )


class SoftwareEStopControl:
    """Narrow, explicit software E-stop command client.

    This is deliberately separate from :class:`ReadOnlyRobot`: callers cannot
    accidentally gain arm, hand, head, or chassis command publishers.
    """

    def __init__(self, robot_info: RobotInfo) -> None:
        config = robot_info.get_component_config("estop")
        self._node = Node(name="omniteleop_estop_control")
        self._client = self._node.create_service_client(
            service_name=config.estop_query_name,
            request_encoder=SoftwareEstopCodec.encode,
            response_decoder=None,
            timeout=0.2,
        )

    def set_enabled(
        self,
        enabled: bool,
        observer: ReadOnlyEStop,
        timeout: float = 2.0,
    ) -> None:
        if not self._client.wait_for_service(timeout=1.0):
            raise ServiceUnavailableError("Software E-stop service is unavailable")
        self._client.call({"enabled": enabled})

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if observer.is_software_estop_enabled() is enabled:
                return
            time.sleep(0.02)
        raise ServiceUnavailableError(
            f"Software E-stop did not reach enabled={enabled} within {timeout:.1f}s"
        )

    def shutdown(self) -> None:
        self._node.shutdown()


class ReadOnlyRobot:
    """Observation-only subset of ``dexcontrol.Robot`` used by OmniTeleop."""

    _JOINT_COMPONENTS = (
        "left_arm",
        "right_arm",
        "head",
        "torso",
        "left_hand",
        "right_hand",
    )

    def __init__(self, configs: BaseRobotConfig | None = None) -> None:
        self._shutdown_called = False
        self._components: list[RobotComponent] = []
        self.sensors: Sensors | None = None
        self._robot_info = RobotInfo(configs=configs)
        self._configs = self._robot_info.config

        try:
            for name in self._JOINT_COMPONENTS:
                config = self._configs.components.get(name)
                if config is None or not config.enabled:
                    continue
                component = ReadOnlyJointComponent(
                    name=name,
                    robot_info=self._robot_info,
                    state_sub_topic=config.state_sub_topic,
                )
                if name in ("left_arm", "right_arm"):
                    wrench_topic = getattr(config, "wrench_sub_topic", "")
                    component.wrench_sensor = (
                        ReadOnlyWrenchSensor(f"{name}_wrench", wrench_topic)
                        if wrench_topic
                        else None
                    )
                    if component.wrench_sensor is not None:
                        self._components.append(component.wrench_sensor)
                setattr(self, name, component)
                self._components.append(component)

            estop_config = self._configs.components.get("estop")
            if estop_config is not None and estop_config.enabled:
                self.estop = ReadOnlyEStop("estop", estop_config.state_sub_topic)
                self._components.append(self.estop)

            self.sensors = Sensors(self._configs.sensors)
        except Exception:
            self.shutdown()
            raise

    @property
    def robot_model(self) -> str:
        return self._robot_info.robot_model

    @property
    def robot_info(self) -> RobotInfo:
        """Configuration metadata for constructing narrow safety clients."""
        return self._robot_info

    def __enter__(self) -> "ReadOnlyRobot":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def has_component(self, component: str) -> bool:
        return hasattr(self, component)

    def get_joint_pos_dict(
        self, component: str | list[str]
    ) -> dict[str, float]:
        names = [component] if isinstance(component, str) else component
        result: dict[str, float] = {}
        for name in names:
            part = getattr(self, name, None)
            if part is None:
                raise KeyError(f"Unavailable component: {name}")
            result.update(part.get_joint_pos_dict())
        return result

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self.sensors is not None:
            self.sensors.shutdown()
        for component in reversed(self._components):
            component.shutdown()
