from __future__ import annotations

from types import SimpleNamespace

from dexcontrol.core import component as component_module

import omniteleop.read_only_robot as read_only_module


class FakeSubscriber:
    is_active = True

    def get_latest(self):
        return {
            "pos": [0.0] * 7,
            "vel": [0.0] * 7,
            "timestamp_ns": 1,
        }


class FakeNode:
    instances: list["FakeNode"] = []

    def __init__(self, **_kwargs) -> None:
        self.subscriptions: list[str] = []
        self.publisher_calls = 0
        self.instances.append(self)

    def create_subscriber(self, topic, **_kwargs):
        self.subscriptions.append(topic)
        return FakeSubscriber()

    def create_publisher(self, *_args, **_kwargs):
        self.publisher_calls += 1
        raise AssertionError("ReadOnlyRobot must never create command publishers")

    def shutdown(self) -> None:
        pass


class FakeRobotInfo:
    def __init__(self, configs=None) -> None:
        del configs
        self.config = SimpleNamespace(
            components={
                "left_arm": SimpleNamespace(
                    enabled=True,
                    state_sub_topic="state/left_arm",
                    wrench_sub_topic="",
                ),
                "right_arm": SimpleNamespace(
                    enabled=True,
                    state_sub_topic="state/right_arm",
                    wrench_sub_topic="",
                ),
                "head": SimpleNamespace(
                    enabled=False,
                    state_sub_topic="state/head",
                ),
                "torso": SimpleNamespace(
                    enabled=False,
                    state_sub_topic="state/torso",
                ),
                "left_hand": SimpleNamespace(
                    enabled=False,
                    state_sub_topic="state/left_hand",
                ),
                "right_hand": SimpleNamespace(
                    enabled=False,
                    state_sub_topic="state/right_hand",
                ),
                "estop": SimpleNamespace(
                    enabled=False,
                    state_sub_topic="state/estop",
                ),
            },
            sensors={},
        )
        self.robot_model = "vega_1u"

    def get_component_joints(self, name: str) -> list[str]:
        prefix = "L" if name == "left_arm" else "R"
        return [f"{prefix}_arm_j{i}" for i in range(1, 8)]

    def get_joint_pos_limits(self, names: list[str]) -> list[list[float]]:
        return [[-1.0, 1.0] for _ in names]


class FakeSensors:
    def __init__(self, _configs) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_read_only_robot_creates_subscribers_but_no_publishers(monkeypatch) -> None:
    FakeNode.instances.clear()
    monkeypatch.setattr(component_module, "Node", FakeNode)
    monkeypatch.setattr(read_only_module, "RobotInfo", FakeRobotInfo)
    monkeypatch.setattr(read_only_module, "Sensors", FakeSensors)

    robot = read_only_module.ReadOnlyRobot()
    try:
        assert robot.has_component("left_arm")
        assert robot.has_component("right_arm")
        assert len(FakeNode.instances) == 2
        assert all(node.subscriptions for node in FakeNode.instances)
        assert all(node.publisher_calls == 0 for node in FakeNode.instances)
        assert len(robot.get_joint_pos_dict(["left_arm", "right_arm"])) == 14
    finally:
        robot.shutdown()

