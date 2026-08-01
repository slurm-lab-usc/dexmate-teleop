from omniteleop.common.joint_state_safety import has_active_joint_error


def test_zero_filled_joint_error_list_is_healthy() -> None:
    assert not has_active_joint_error([0, 0, 0, 0, 0, 0, 0])


def test_nonzero_joint_error_code_is_active() -> None:
    assert has_active_joint_error([0, 0, 0, 0, 0, 5, 0])


def test_diagnostic_error_mapping_is_supported() -> None:
    assert has_active_joint_error(
        {"joint_6": {"error_code": 5, "error_message": "Position limit exceeded"}}
    )
    assert not has_active_joint_error(
        {"joint_6": {"error_code": 0, "error_message": ""}}
    )

