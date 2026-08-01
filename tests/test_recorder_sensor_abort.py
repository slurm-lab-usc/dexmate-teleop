from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path


class _FakeNode:
    def __init__(self, **_kwargs) -> None:
        pass


class _FakeRateLimiter:
    def __init__(self, _rate: float) -> None:
        pass

    def sleep(self) -> None:
        time.sleep(0.001)


class _FakeCodec:
    decode = staticmethod(lambda value: value)


dexcomm_stub = sys.modules.setdefault("dexcomm", types.ModuleType("dexcomm"))
dexcomm_stub.Node = _FakeNode
codecs_stub = sys.modules.setdefault(
    "dexcomm.codecs", types.ModuleType("dexcomm.codecs")
)
codecs_stub.DictDataCodec = _FakeCodec
utils_stub = sys.modules.setdefault("dexcomm.utils", types.ModuleType("dexcomm.utils"))
utils_stub.RateLimiter = _FakeRateLimiter

common_stub = types.ModuleType("omniteleop.common")
common_stub.get_config = lambda: {
    "recorder": {
        "save_dir": "/tmp/omniteleop-recorder-test",
        "record_rate": 1000.0,
        "wrist_camera_adapter": {"abort_after_s": 0.0},
        "components": {
            "left_arm": False,
            "right_arm": False,
            "torso": False,
            "head": False,
            "left_hand": False,
            "right_hand": False,
        },
    }
}
previous_common = sys.modules.get("omniteleop.common")
sys.modules["omniteleop.common"] = common_stub

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "omniteleop"
    / "record"
    / "base_recorder.py"
)
_spec = importlib.util.spec_from_file_location("base_recorder_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
if previous_common is None:
    sys.modules.pop("omniteleop.common", None)
else:
    sys.modules["omniteleop.common"] = previous_common

BaseRecorder = _base.BaseRecorder
RequiredSensorError = _base.RequiredSensorError


class _FailingRecorder(BaseRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.finalized: list[bool] = []

    def _setup_storage(self, _metadata) -> None:
        self.episode_dir = Path("/tmp/fake-required-sensor-episode")

    def _collect_observation(self):
        raise RequiredSensorError("left wrist camera stale")

    def _write_frame(self, *_args) -> None:
        raise AssertionError("a stale frame must never be written")

    def _finalize_storage(self, success: bool) -> None:
        self.finalized.append(success)


def test_required_sensor_failure_auto_discards_without_self_join_deadlock() -> None:
    recorder = _FailingRecorder()
    recorder.start_episode()

    deadline = time.monotonic() + 2.0
    while recorder.is_recording and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not recorder.is_recording
    assert recorder.finalized == [False]
    assert recorder.episode_num == 1


def test_toggle_command_uses_recorder_as_authoritative_state() -> None:
    recorder = BaseRecorder.__new__(BaseRecorder)
    calls: list[tuple[str, dict]] = []
    recorder.is_recording = False
    recorder.start_episode = lambda metadata: calls.append(("start", metadata))
    recorder.end_episode = lambda: calls.append(("stop", {}))

    recorder._on_control_received({"command": "toggle", "metadata": {"run": 1}})
    assert calls == [("start", {"run": 1})]

    recorder.is_recording = True
    recorder._on_control_received({"command": "toggle", "metadata": {"run": 2}})
    assert calls[-1] == ("stop", {})
