from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from omniteleop.record.errors import RequiredSensorError
from omniteleop.record.sensors import CameraFreshnessGuard

if "dexcomm" not in sys.modules and importlib.util.find_spec("dexcomm") is None:
    dexcomm_stub = types.ModuleType("dexcomm")
    dexcomm_stub.Node = object
    codecs_stub = types.ModuleType("dexcomm.codecs")
    codecs_stub.RGBImageCodec = object
    sys.modules["dexcomm"] = dexcomm_stub
    sys.modules["dexcomm.codecs"] = codecs_stub
else:
    sys.modules["dexcomm"].Node = getattr(sys.modules["dexcomm"], "Node", object)
    codecs_stub = sys.modules.setdefault(
        "dexcomm.codecs", types.ModuleType("dexcomm.codecs")
    )
    codecs_stub.RGBImageCodec = getattr(codecs_stub, "RGBImageCodec", object)

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "omniteleop"
    / "record"
    / "wrist_camera_adapter.py"
)
_spec = importlib.util.spec_from_file_location("wrist_camera_adapter_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _adapter
_spec.loader.exec_module(_adapter)

AdapterConfig = _adapter.AdapterConfig
center_crop_and_resize = _adapter.center_crop_and_resize


def test_adapter_config_defaults_and_validation() -> None:
    config = AdapterConfig.from_mapping({"enabled": True})
    assert config.left_device == "/dev/dexmate-wrist-left"
    assert config.right_device == "/dev/dexmate-wrist-right"
    assert config.capture_resolution == (1280, 720)
    assert config.output_resolution == (960, 600)
    assert config.fourcc == "MJPG"

    with pytest.raises(ValueError, match="fourcc"):
        AdapterConfig.from_mapping({"fourcc": "MJPEG"})
    with pytest.raises(ValueError, match="output_resolution"):
        AdapterConfig.from_mapping({"output_resolution": [0, 600]})


def test_center_crop_resize_preserves_center_and_converts_bgr_to_rgb() -> None:
    # For 1280x720 -> 16:10, 64 columns are removed from each side.
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :64] = [1, 2, 3]
    image[:, 64:1216] = [10, 20, 30]
    image[:, 1216:] = [4, 5, 6]

    output = center_crop_and_resize(image, (960, 600))

    assert output.shape == (600, 960, 3)
    assert output.flags.c_contiguous
    assert np.all(output[300, 480] == [30, 20, 10])
    assert np.all(output[:, 0] == [30, 20, 10])
    assert np.all(output[:, -1] == [30, 20, 10])


def test_center_crop_resize_rejects_non_color_input() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        center_crop_and_resize(np.zeros((10, 10), dtype=np.uint8), (8, 8))


def test_camera_freshness_guard_rejects_missing_and_stale_frames(monkeypatch) -> None:
    clock = {"wall_ns": 1_000_000_000, "mono": 1.0}
    monkeypatch.setattr(
        "omniteleop.record.sensors.time.time_ns",
        lambda: clock["wall_ns"],
    )
    monkeypatch.setattr(
        "omniteleop.record.sensors.time.monotonic",
        lambda: clock["mono"],
    )
    guard = CameraFreshnessGuard("left", stale_timeout_s=0.25)
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    payload, timestamp_ns = guard.validate(
        {"data": image, "timestamp_ns": clock["wall_ns"]}
    )
    assert payload is image
    assert timestamp_ns == 1_000_000_000

    clock["wall_ns"] += 300_000_000
    clock["mono"] += 0.3
    with pytest.raises(RequiredSensorError, match="left wrist camera stale"):
        guard.validate({"data": image, "timestamp_ns": timestamp_ns})

    with pytest.raises(RequiredSensorError, match="frame unavailable"):
        guard.validate(None)
