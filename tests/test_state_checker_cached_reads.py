from __future__ import annotations

from omniteleop.app.backend.state_checker import StateChecker


def cached_checker() -> StateChecker:
    checker = StateChecker.__new__(StateChecker)
    checker.robot_name = "dm/test"
    checker.leader_mode = "exoskeleton"
    checker._last_topic_list = "dm/test/state/arm/left"
    checker._recorder_enabled = False
    checker._recorder_components = {}
    checker._component_errors_cache = {
        "right_arm": {
            "error": {"error_code": 5, "error_message": "Position limit exceeded"},
            "operation": 0,
        }
    }
    checker._latest_estop_state = {"software_estop_enabled": True}
    checker._exo_data = {
        "data": {
            "left_arm_pos": [0.0] * 7,
            "right_arm_pos": [0.0] * 7,
        }
    }
    checker._robot_left_joints = [0.0] * 7
    checker._robot_right_joints = [0.0] * 7
    return checker


def test_ui_diagnostics_only_read_caches(monkeypatch) -> None:
    checker = cached_checker()

    def fail(*_args, **_kwargs):
        raise AssertionError("UI state generation attempted blocking I/O")

    monkeypatch.setattr(checker, "_get_topic_list", fail)
    monkeypatch.setattr(checker, "_get_exo_joint_angles", fail)
    checker._query_interface = type(
        "FailingQuery",
        (),
        {"get_component_status": fail},
    )()

    topics = checker.get_missing_topics()
    errors, estop = checker.get_component_errors()
    limits = checker.get_out_of_limit_joints()
    sensors = checker.get_sensor_topics_status()

    assert topics["found"] == ["dm/test/state/arm/left"]
    assert errors["right_arm"]["error_code"] == 5
    assert estop is True
    assert limits == {"message": "Exoskeleton joints not within limits"}
    assert isinstance(sensors, dict)
