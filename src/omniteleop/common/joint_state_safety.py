"""Helpers for interpreting decoded joint-state error payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any


def has_active_joint_error(value: Any) -> bool:
    """Return whether a JointState ``error`` payload contains a real error.

    DexComm joint states normally expose a list of numeric per-joint codes, so
    ``[0, 0, ...]`` is healthy even though the list itself is truthy. Some
    diagnostic paths expose nested dictionaries instead; those are handled too.
    """
    if value is None:
        return False
    if isinstance(value, Mapping):
        error_code = value.get("error_code")
        error_message = value.get("error_message")
        if isinstance(error_code, Number) and error_code != 0:
            return True
        if isinstance(error_message, str) and bool(error_message.strip()):
            return True
        return any(
            has_active_joint_error(item)
            for key, item in value.items()
            if key not in {"error_code", "error_message"}
        )
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(has_active_joint_error(item) for item in value)
    if isinstance(value, Number):
        return value != 0
    return bool(value)

