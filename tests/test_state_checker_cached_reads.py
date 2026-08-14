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
    checker._motion_started_latch = False
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


def test_alignment_status_reports_live_per_joint_directions() -> None:
    checker = cached_checker()
    checker._exo_data["data"]["left_arm_pos"][0] = -1.2
    checker._robot_left_joints[0] = 0.2
    checker._exo_data["data"]["right_arm_pos"][1] = 0.4
    checker._robot_right_joints[1] = -0.8

    status = checker.get_alignment_status()

    assert status["aligned"] is False
    assert status["aligned_joints"] == 12
    assert status["reason"] == "not_aligned"
    assert status["arms"]["left"]["joints"][0]["direction"] == "increase"
    assert status["arms"]["left"]["joints"][0]["target_delta"] == 1.4
    assert status["arms"]["right"]["joints"][1]["direction"] == "decrease"
    assert status["arms"]["right"]["joints"][1]["target_delta"] == -1.2
    assert checker._check_exo_joints_within_limits() is False


def test_alignment_status_distinguishes_limit_violation_from_pose_error() -> None:
    checker = cached_checker()
    checker._exo_data["data"]["left_arm_pos"][1] = -0.8

    status = checker.get_alignment_status()
    joint = status["arms"]["left"]["joints"][1]

    assert joint["in_limits"] is False
    assert joint["aligned"] is False
    assert joint["direction"] == "increase_to_limit"
    assert joint["limit_min"] == -0.453


def test_alignment_status_explains_missing_exoskeleton_data() -> None:
    checker = cached_checker()
    checker._exo_data = {"data": None}

    status = checker.get_alignment_status()

    assert status["aligned"] is False
    assert status["aligned_joints"] == 0
    assert status["reason"] == "exo_data_missing"
    assert status["arms"]["left"]["joints"][0]["direction"] == "missing"


def test_alignment_guide_and_state_gate_pass_together() -> None:
    checker = cached_checker()
    checker._motion_started_latch = False

    status = checker.get_alignment_status()

    assert status["aligned"] is True
    assert status["aligned_joints"] == 14
    assert status["reason"] == "aligned"
    assert checker._check_exo_joints_within_limits() is True
