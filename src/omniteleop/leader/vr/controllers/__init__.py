"""VR teleoperation controllers."""

import numpy as np
from omniteleop.common.config import RobotConfig


def get_hand_poses(
    config: RobotConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Extract hand open/close poses from robot config.

    Looks in input_handlers.vr.hands first, then exo_joycon.hands.

    Args:
        config: RobotConfig instance.

    Returns:
        Tuple of (open_poses, close_poses) dicts keyed by "left"/"right",
        or None if hand poses are not configured.
    """
    for handler_key in ("vr", "exo_joycon"):
        hands_config = (
            config.get("input_handlers", {}).get(handler_key, {}).get("hands", {})
        )
        if hands_config:
            break
    else:
        return None

    open_poses = {}
    close_poses = {}
    for side in ("left", "right"):
        side_config = hands_config.get(side, {})
        poses = side_config.get("poses", {})
        if "open" not in poses or "close" not in poses:
            return None
        open_poses[side] = np.array(poses["open"])
        close_poses[side] = np.array(poses["close"])

    return open_poses, close_poses
