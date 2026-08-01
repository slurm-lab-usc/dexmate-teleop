"""Leader modules for teleoperation input devices.

Keep imports lazy so optional hardware dependencies for one leader, such as
``dynamixel_sdk`` for the exoskeleton arm reader, do not prevent unrelated
leaders from starting.
"""

__all__ = [
    "LeaderArmReader",
    "leader_arm_main",
    "JoyConReader",
    "joycon_main",
    "PaddleLeader",
    "paddle_main",
]


def __getattr__(name: str):
    if name in {"LeaderArmReader", "leader_arm_main"}:
        from .arm_reader import LeaderArmReader, main

        return LeaderArmReader if name == "LeaderArmReader" else main
    if name in {"JoyConReader", "joycon_main"}:
        from .joycon_reader import JoyConReader, main

        return JoyConReader if name == "JoyConReader" else main
    if name in {"PaddleLeader", "paddle_main"}:
        from .paddle_leader import PaddleLeader, main

        return PaddleLeader if name == "PaddleLeader" else main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
