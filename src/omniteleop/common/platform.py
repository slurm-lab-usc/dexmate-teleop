"""Platform-specific device discovery for the leader-side scripts.

Ubuntu and macOS disagree about how the same USB/Bluetooth hardware shows up:

* A U2D2 is ``/dev/ttyUSB0`` on Linux but ``/dev/cu.usbserial-<serial>`` on
  macOS, and the trailing index moves with enumeration order on both.
* Joy-Cons reach userspace through the ``hid-nintendo`` kernel driver (evdev)
  on Linux, and as raw HID devices on macOS.
* Bluetooth pairing is inspectable via ``bluetoothctl`` (BlueZ) on Linux; macOS
  has no equivalent CLI worth shelling out to.

Every one of those branches lives in this module so the readers, the GUI
backend and the state checker share one answer instead of hardcoding paths.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = os.name == "nt"

# Config sentinel meaning "work it out from the connected hardware".
AUTO = "auto"

# USB-serial bridges known to carry a Dynamixel bus, most likely first. The
# U2D2 (FT232H, 0403:6014) is the shipping adapter; the rest cover older
# USB2Dynamixel units and third-party cables.
KNOWN_SERIAL_ADAPTERS: Tuple[Tuple[int, int, str], ...] = (
    (0x0403, 0x6014, "FTDI FT232H (U2D2)"),
    (0x0403, 0x6001, "FTDI FT232R (USB2Dynamixel)"),
    (0x0403, 0x6010, "FTDI FT2232"),
    (0x0403, 0x6015, "FTDI FT-X"),
    (0x10C4, 0xEA60, "Silicon Labs CP210x"),
    (0x1A86, 0x7523, "QinHeng CH340"),
    (0x1A86, 0x55D4, "QinHeng CH9102"),
)

# Device paths that are serial ports but never a robot: Linux legacy UARTs and
# the macOS built-ins that appear in every enumeration.
_IGNORED_PORT_PREFIXES = (
    "/dev/ttyS",
    "/dev/cu.Bluetooth",
    "/dev/tty.Bluetooth",
    "/dev/cu.debug-console",
    "/dev/tty.debug-console",
    "/dev/cu.wlan-debug",
    "/dev/tty.wlan-debug",
)

# Nintendo USB vendor id and the Joy-Con product ids (left, right, Pro
# Controller, charging grip). Used by the HID path and the presence check.
NINTENDO_VID = 0x057E
JOYCON_PIDS: Dict[int, str] = {
    0x2006: "left",
    0x2007: "right",
    0x2009: "pro",
    0x200E: "grip",
}


# ─── Joy-Con backend ───────────────────────────────────────────────────────


def default_joycon_backend() -> str:
    """Return the joycon_lib backend that can work on this machine.

    Linux prefers ``evdev``: the ``hid-nintendo`` kernel driver already does
    the report parsing and stick calibration. Everywhere else there is no such
    driver, so raw ``hid`` is the only option.

    joycon_lib's own detection is consulted first, because it also knows which
    backends are actually *installed* — a Linux box without the evdev package
    should fall back to raw HID rather than fail at connect time.
    """
    try:
        from joycon_lib.joycon import backend as detected
        if detected:
            return detected
    except Exception as e:  # noqa: BLE001 — fall back to the static answer
        logger.debug(f"joycon_lib backend detection unavailable: {e}")
    return "evdev" if IS_LINUX else "hid"


def joycons_present() -> Dict[str, bool]:
    """Best-effort check for a paired left and right Joy-Con.

    Returns a ``{"left": bool, "right": bool}`` dict. Used for pre-flight
    checks in the GUI, so a failure to probe reports False rather than
    raising — the reader itself is the authority on whether it can connect.
    """
    result = {"left": False, "right": False}
    if IS_LINUX:
        # BlueZ knows about the pairing before hid-nintendo binds the device,
        # which makes it the earlier (and therefore more useful) signal.
        try:
            proc = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            out = proc.stdout or ""
            result["left"] = "Joy-Con (L)" in out
            result["right"] = "Joy-Con (R)" in out
            return result
        except Exception as e:  # noqa: BLE001 — probe only, never fatal
            logger.debug(f"bluetoothctl probe failed: {e}")
            return result

    # macOS (and any other non-Linux): a paired Joy-Con is enumerable over HID.
    try:
        import hid  # type: ignore
    except ImportError:
        logger.debug("hid module not installed; cannot probe Joy-Cons")
        return result
    try:
        for info in hid.enumerate(NINTENDO_VID, 0):
            side = JOYCON_PIDS.get(_info_get(info, "product_id"))
            if side in result:
                result[side] = True
    except Exception as e:  # noqa: BLE001 — probe only, never fatal
        logger.debug(f"hid.enumerate probe failed: {e}")
    return result


def _info_get(info: Any, key: str) -> Any:
    """Read a field from a hid.enumerate() entry.

    The two PyPI packages that both import as ``hid`` disagree: ``hidapi``
    (Cython) yields dicts, ``hid`` (ctypes) yields objects with attributes.
    """
    if isinstance(info, dict):
        return info.get(key)
    return getattr(info, key, None)


# ─── Serial ports ──────────────────────────────────────────────────────────


def parse_vid_pid(spec: Any) -> Optional[Tuple[int, int]]:
    """Parse a ``"0403:6014"`` style config value into ``(vid, pid)``.

    Accepts an already-parsed 2-sequence too. Returns None on anything
    unparseable, so a typo in the YAML degrades to "no filter" rather than a
    crash at startup.
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        try:
            return int(spec[0]), int(spec[1])
        except (TypeError, ValueError):
            return None
    if isinstance(spec, str) and ":" in spec:
        vid_str, _, pid_str = spec.partition(":")
        try:
            return int(vid_str, 16), int(pid_str, 16)
        except ValueError:
            return None
    logger.warning(f"Could not parse vid:pid {spec!r} — ignoring the filter")
    return None


def list_serial_ports() -> List[Any]:
    """Return pyserial ``ListPortInfo`` entries, minus the never-useful ones.

    On macOS pyserial reports the callout node (``/dev/cu.*``); the dial-in
    node (``/dev/tty.*``) blocks on open waiting for carrier detect, so it is
    filtered out if a backend ever surfaces one.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        logger.warning("pyserial is not installed — cannot enumerate serial ports")
        return []

    ports = []
    for port in list_ports.comports():
        device = port.device or ""
        if device.startswith(_IGNORED_PORT_PREFIXES):
            continue
        if IS_MAC and device.startswith("/dev/tty."):
            continue
        ports.append(port)
    return ports


def describe_serial_ports() -> str:
    """One-line, human-readable inventory of candidate ports for error text."""
    ports = list_serial_ports()
    if not ports:
        return "none"
    parts = []
    for p in ports:
        ids = f"{p.vid:04x}:{p.pid:04x}" if p.vid is not None else "no-usb-id"
        parts.append(f"{p.device} ({ids}{f', sn={p.serial_number}' if p.serial_number else ''})")
    return "; ".join(parts)


def _rank(port: Any, wanted: Optional[Tuple[int, int]]) -> Optional[int]:
    """Sort key for a candidate port; None means "not a candidate".

    Lower is better: known adapters score by their position in
    :data:`KNOWN_SERIAL_ADAPTERS`, any other USB-serial device sorts after
    them, and non-USB ports are rejected outright.
    """
    if port.vid is None:
        return None
    if wanted is not None:
        return 0 if (port.vid, port.pid) == wanted else None
    for index, (vid, pid, _name) in enumerate(KNOWN_SERIAL_ADAPTERS):
        if (port.vid, port.pid) == (vid, pid):
            return index
    return len(KNOWN_SERIAL_ADAPTERS)


def find_serial_port(
    configured: Optional[str] = None,
    *,
    serial_number: Optional[str] = None,
    vid_pid: Any = None,
) -> Optional[str]:
    """Resolve the serial port to open, working on both Ubuntu and macOS.

    Args:
        configured: The ``port`` value from config (or a CLI override). An
            existing path — or a glob that matches one — is honoured as-is.
            ``"auto"``, empty or None triggers a scan. A path that does *not*
            exist also falls back to a scan (with a warning), which is what
            makes a Linux-authored config work unchanged on a Mac.
        serial_number: Pin to one adapter by its USB serial (e.g. the U2D2's
            ``FTC04DXY``). Survives replugging and reboots, unlike the index.
        vid_pid: Restrict to one adapter type, as ``"0403:6014"`` or a
            ``(vid, pid)`` pair.

    Returns:
        A device path, or None when nothing plausible is attached.
    """
    wanted = parse_vid_pid(vid_pid)

    if configured and str(configured).strip().lower() != AUTO:
        configured = str(configured).strip()
        if os.path.exists(configured):
            return configured
        matches = sorted(glob.glob(configured))
        if matches:
            if len(matches) > 1:
                logger.info(f"Port pattern {configured!r} matched {matches}; using {matches[0]}")
            return matches[0]
        logger.warning(
            f"Configured port {configured!r} does not exist on this machine "
            f"({sys.platform}) — falling back to auto-detection"
        )

    candidates = []
    for port in list_serial_ports():
        rank = _rank(port, wanted)
        if rank is None:
            continue
        if serial_number and (port.serial_number or "").lower() != serial_number.lower():
            continue
        candidates.append((rank, port.device, port))

    if not candidates:
        logger.error(
            "No USB-serial adapter found. Ports visible: " + describe_serial_ports()
        )
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen = candidates[0][2]
    ids = f"{chosen.vid:04x}:{chosen.pid:04x}"
    logger.info(
        f"Auto-detected serial port {chosen.device} "
        f"({ids}{f', sn={chosen.serial_number}' if chosen.serial_number else ''})"
    )
    if len(candidates) > 1:
        others = ", ".join(item[1] for item in candidates[1:])
        logger.warning(
            f"Multiple adapters attached ({others} also matched). Pin one with "
            f"leader_arms.port (explicit path) or leader_arms.port_serial_number."
        )
    return chosen.device
