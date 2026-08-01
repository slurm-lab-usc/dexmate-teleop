#!/usr/bin/env python3
"""Publish local UVC wrist cameras on the standard Dexmate camera topics."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from multiprocessing import get_context
from queue import Empty, Full
from typing import Any

import cv2  # type: ignore
import numpy as np
import tyro
from dexcomm import Node
from dexcomm.codecs import RGBImageCodec
from loguru import logger


@dataclass(frozen=True)
class AdapterConfig:
    enabled: bool
    left_device: str
    right_device: str
    capture_resolution: tuple[int, int]
    output_resolution: tuple[int, int]
    fps: float
    fourcc: str
    startup_timeout_s: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "AdapterConfig":
        cfg = value or {}
        capture = cfg.get("capture_resolution", [1280, 720])
        output = cfg.get("output_resolution", [960, 600])
        fourcc = str(cfg.get("fourcc", "MJPG")).upper()
        if len(capture) != 2 or min(int(v) for v in capture) <= 0:
            raise ValueError("capture_resolution must be [width, height] with positive values")
        if len(output) != 2 or min(int(v) for v in output) <= 0:
            raise ValueError("output_resolution must be [width, height] with positive values")
        if len(fourcc) != 4:
            raise ValueError("fourcc must contain exactly four characters")
        fps = float(cfg.get("fps", 30.0))
        timeout = float(cfg.get("startup_timeout_s", 5.0))
        if fps <= 0 or timeout <= 0:
            raise ValueError("fps and startup_timeout_s must be positive")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            left_device=str(cfg.get("left_device", "/dev/dexmate-wrist-left")),
            right_device=str(cfg.get("right_device", "/dev/dexmate-wrist-right")),
            capture_resolution=(int(capture[0]), int(capture[1])),
            output_resolution=(int(output[0]), int(output[1])),
            fps=fps,
            fourcc=fourcc,
            startup_timeout_s=timeout,
        )


def center_crop_and_resize(
    image_bgr: np.ndarray,
    output_resolution: tuple[int, int],
) -> np.ndarray:
    """Center-crop without distortion, resize, and convert BGR to contiguous RGB."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {image_bgr.shape}")
    src_h, src_w = image_bgr.shape[:2]
    out_w, out_h = output_resolution
    src_ratio = src_w / src_h
    out_ratio = out_w / out_h
    if src_ratio > out_ratio:
        crop_w = max(1, round(src_h * out_ratio))
        x0 = (src_w - crop_w) // 2
        cropped = image_bgr[:, x0 : x0 + crop_w]
    else:
        crop_h = max(1, round(src_w / out_ratio))
        y0 = (src_h - crop_h) // 2
        cropped = image_bgr[y0 : y0 + crop_h, :]
    resized = cv2.resize(cropped, output_resolution, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def _open_capture(side: str, device: str, config: AdapterConfig) -> Any:
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {device}")
    width, height = config.capture_resolution
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (actual_width, actual_height) != (width, height):
        capture.release()
        raise RuntimeError(
            f"{device} negotiated {actual_width}x{actual_height}, "
            f"expected {width}x{height}"
        )
    if actual_fps and abs(actual_fps - config.fps) > 1.0:
        capture.release()
        raise RuntimeError(
            f"{device} negotiated {actual_fps:.1f} FPS, expected {config.fps:.1f}"
        )
    logger.info(
        f"{side} wrist camera opened: {device}, "
        f"{actual_width}x{actual_height}@{actual_fps or config.fps:.1f}"
    )
    return capture


def _capture_process_entry(
    side: str,
    device: str,
    config: AdapterConfig,
    frame_queue: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """Capture continuously in a separate interpreter and keep one latest frame."""
    capture = None
    sequence = 0
    try:
        capture = _open_capture(side, device, config)
        frames_since_log = 0
        log_started = time.monotonic()
        while not stop_event.is_set():
            ok, image_bgr = capture.read()
            capture_timestamp_ns = time.time_ns()
            if not ok or not isinstance(image_bgr, np.ndarray):
                raise RuntimeError(f"read failed for {device}")
            image_rgb = center_crop_and_resize(image_bgr, config.output_resolution)
            frame = (image_rgb, capture_timestamp_ns, sequence)
            sequence += 1
            try:
                frame_queue.put_nowait(frame)
            except Full:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
                try:
                    frame_queue.put_nowait(frame)
                except Full:
                    pass
            frames_since_log += 1

            now = time.monotonic()
            if now - log_started >= 5.0:
                logger.info(
                    f"{side} wrist capture: "
                    f"{frames_since_log / (now - log_started):.1f} FPS, "
                    f"sequence={sequence}"
                )
                frames_since_log = 0
                log_started = now
    except Exception as exc:
        try:
            error_queue.put_nowait(str(exc))
        except Full:
            pass
        stop_event.set()
        raise
    finally:
        if capture is not None:
            capture.release()


class UVCCameraPublisher:
    """Capture in a child process and publish the newest frame in this process."""

    def __init__(
        self,
        *,
        side: str,
        device: str,
        config: AdapterConfig,
        publisher: Any,
    ) -> None:
        self.side = side
        self.device = device
        self.config = config
        self.publisher = publisher
        self.ready = threading.Event()
        self.running = threading.Event()
        self.running.set()
        self.error: Exception | None = None
        self.publish_thread: threading.Thread | None = None
        self.published_frames = 0
        self._context = get_context("spawn")
        self._frame_queue = self._context.Queue(maxsize=1)
        self._error_queue = self._context.Queue(maxsize=1)
        self._stop_event = self._context.Event()
        self.capture_process: Any | None = None

    def start(self) -> None:
        self.capture_process = self._context.Process(
            target=_capture_process_entry,
            args=(
                self.side,
                self.device,
                self.config,
                self._frame_queue,
                self._error_queue,
                self._stop_event,
            ),
            name=f"{self.side}-wrist-capture",
        )
        self.publish_thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name=f"{self.side}-wrist-publish",
        )
        self.capture_process.start()
        self.publish_thread.start()

    def wait_until_ready(self, timeout: float) -> None:
        if not self.ready.wait(timeout):
            raise TimeoutError(
                f"{self.side} wrist camera produced no frame within {timeout:.1f}s"
            )
        if self.error is not None:
            raise RuntimeError(f"{self.side} wrist camera failed: {self.error}") from self.error

    def stop(self) -> None:
        self.running.clear()
        self._stop_event.set()
        if self.publish_thread and self.publish_thread is not threading.current_thread():
            self.publish_thread.join(timeout=2.0)
        if self.capture_process:
            self.capture_process.join(timeout=2.0)
            if self.capture_process.is_alive():
                self.capture_process.terminate()
                self.capture_process.join(timeout=1.0)
        self._frame_queue.close()
        self._error_queue.close()

    def _fail(self, exc: Exception) -> None:
        if self.error is None:
            self.error = exc
            logger.error(f"{self.side} wrist camera stopped: {exc}")
        self.running.clear()
        self._stop_event.set()
        self.ready.set()

    def _publish_loop(self) -> None:
        frames_since_log = 0
        log_started = time.monotonic()
        try:
            while self.running.is_set():
                try:
                    image_rgb, timestamp_ns, sequence = self._frame_queue.get(timeout=0.5)
                except Empty:
                    if not self.running.is_set():
                        return
                    if (
                        self.capture_process is not None
                        and not self.capture_process.is_alive()
                    ):
                        try:
                            message = self._error_queue.get_nowait()
                        except Empty:
                            message = (
                                f"capture process exited with "
                                f"code {self.capture_process.exitcode}"
                            )
                        raise RuntimeError(message)
                    continue
                height, width = image_rgb.shape[:2]
                self.publisher.publish(
                    {
                        "data": image_rgb,
                        "height": height,
                        "width": width,
                        "channels": 3,
                        "color_format": "rgb",
                        "timestamp_ns": timestamp_ns,
                        "sequence": sequence,
                    }
                )
                self.published_frames += 1
                frames_since_log += 1
                self.ready.set()

                now = time.monotonic()
                if now - log_started >= 5.0:
                    logger.info(
                        f"{self.side} wrist publish: "
                        f"{frames_since_log / (now - log_started):.1f} FPS, "
                        f"sequence={sequence}"
                    )
                    frames_since_log = 0
                    log_started = now
        except Exception as exc:
            self._fail(exc)


class WristCameraAdapter:
    """Compose enabled UVC sources with the standard Zenoh publishers."""

    def __init__(self, namespace: str = "", side: str | None = None) -> None:
        from omniteleop.common import get_config

        if side not in (None, "left", "right"):
            raise ValueError("side must be 'left', 'right', or None")
        root_config = get_config()
        recorder_cfg = root_config.get("recorder", {}) or {}
        self.config = AdapterConfig.from_mapping(
            recorder_cfg.get("wrist_camera_adapter")
        )
        components = recorder_cfg.get("components", {}) or {}
        self.enabled_sides = [
            candidate
            for candidate in ("left", "right")
            if bool(components.get(f"{candidate}_wrist_rgb", False))
            and (side is None or side == candidate)
        ]
        self.node = Node(name="wrist_camera_adapter", namespace=namespace)
        self.sources: list[UVCCameraPublisher] = []

    def start(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("recorder.wrist_camera_adapter.enabled is false")
        if not self.enabled_sides:
            raise RuntimeError("no wrist RGB components are enabled")
        for side in self.enabled_sides:
            topic = f"sensors/{side}_wrist_camera/rgb"
            publisher = self.node.create_publisher(
                topic,
                encoder=RGBImageCodec.encode,
            )
            source = UVCCameraPublisher(
                side=side,
                device=getattr(self.config, f"{side}_device"),
                config=self.config,
                publisher=publisher,
            )
            self.sources.append(source)
            source.start()
        try:
            for source in self.sources:
                source.wait_until_ready(self.config.startup_timeout_s)
        except Exception:
            self.shutdown()
            raise
        logger.info(
            "WRIST CAMERAS READY: "
            + ", ".join(f"{s.side}={s.device}" for s in self.sources)
        )

    def run(self) -> None:
        while all(source.running.is_set() for source in self.sources):
            time.sleep(0.2)
        failures = [f"{s.side}: {s.error}" for s in self.sources if s.error]
        if failures:
            raise RuntimeError("; ".join(failures))

    def shutdown(self) -> None:
        for source in self.sources:
            source.stop()
        self.sources.clear()
        self.node.shutdown()


def _run_single(namespace: str, debug: bool, side: str) -> int:
    from omniteleop.common.logging import setup_logging

    setup_logging(debug)
    adapter = WristCameraAdapter(namespace=namespace, side=side)
    stop_requested = threading.Event()

    def _request_stop(_signum, _frame) -> None:
        stop_requested.set()
        for source in adapter.sources:
            source.running.clear()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        adapter.start()
        adapter.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        if not stop_requested.is_set():
            logger.error(f"Wrist camera adapter error: {exc}")
            return 1
        return 0
    finally:
        adapter.shutdown()


def _child_entry(namespace: str, debug: bool, side: str) -> None:
    raise SystemExit(_run_single(namespace, debug, side))


def main(namespace: str = "", debug: bool = False, side: str = "both") -> int:
    """Run one or both local wrist-camera adapters until interrupted."""
    if side in ("left", "right"):
        return _run_single(namespace, debug, side)
    if side != "both":
        logger.error("side must be one of: both, left, right")
        return 2

    # OpenCV's V4L2 path and Zenoh publishing contend in one interpreter on this
    # host, halving two streams to ~15 FPS each. Separate spawned processes keep
    # the streams independent and sustain the recorder's 20 Hz latest-frame flow.
    context = get_context("spawn")
    children = [
        context.Process(
            target=_child_entry,
            args=(namespace, debug, child_side),
            name=f"{child_side}-wrist-camera-adapter",
        )
        for child_side in ("left", "right")
    ]
    stop_requested = threading.Event()

    def _request_stop(_signum, _frame) -> None:
        stop_requested.set()
        for child in children:
            if child.is_alive():
                child.terminate()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    for child in children:
        child.start()
    try:
        while not stop_requested.is_set():
            exited = [child for child in children if child.exitcode is not None]
            if exited:
                failed = [child for child in exited if child.exitcode != 0]
                if failed:
                    logger.error(
                        "Wrist camera child failed: "
                        + ", ".join(
                            f"{child.name}={child.exitcode}" for child in failed
                        )
                    )
                    return 1
                if len(exited) == len(children):
                    return 0
            time.sleep(0.2)
        return 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=3.0)


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
