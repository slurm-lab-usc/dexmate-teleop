"""Hand component processor."""

import os
from typing import Dict, Any

from dexmotion.utils import robot_utils
from loguru import logger

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import RobotCommand, CommandMode


class HandProcessor(BaseComponentProcessor):
    """Processes hand commands for left or right hand."""

    def __init__(
        self,
        side: str,
        config,
        motion_manager,
        robot_info,
        teleop_mode,
        lock_collision_avoidance: bool = False,
    ):
        """Initialize hand processor.

        Args:
            side: Hand side ('left' or 'right')
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
            lock_collision_avoidance: If True, hand joints are locked in the
                MotionManager (excluded from the reduced pinocchio model). The
                processor still publishes joycon commands to the robot but
                bypasses MotionManager state updates and clipping.
        """
        assert side in ["left", "right"], "Hand side must be 'left' or 'right'"
        super().__init__(side, config, motion_manager, robot_info, teleop_mode)
        self.lock_collision_avoidance = lock_collision_avoidance
        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints(f"{side}_hand")
        # Determine hand type from ROBOT_CONFIG env var
        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        if "gripper" in robot_config:
            self.hand_type = "gripper"
        elif "f5d6" in robot_config:
            self.hand_type = "f5d6"
        else:
            self.hand_type = None  # No hands
        self.joint_pos_limits = self.robot_info.get_joint_pos_limits(self.joint_names)
        # Local last-published positions, used as the relative-mode anchor
        # when MotionManager doesn't carry hand state.
        self._last_pos: list = []
        self._missing_anchor_warned = False

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "hand"

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process hand command based on hand type.

        Routes to hand processing based on configuration.

        Args:
            input_data: Hand input data with 'pos' and 'mode' fields
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        return self._process_hand_command(input_data, command)

    def _process_hand_command(
        self, input_data: Dict[str, Any], command: RobotCommand
    ) -> bool:
        """Process hand_f5d6 joint position commands.

        Updates motion manager state and applies joint limits. When
        lock_collision_avoidance is True, MotionManager state is not touched
        for hand joints (they are locked in the reduced model) and clipping
        falls back to cached URDF limits.

        Args:
            input_data: Hand input data
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        mode = input_data.get("mode", CommandMode.ABSOLUTE)
        positions = input_data.get("pos", [])

        if not positions:
            return False

        # Resolve relative-mode anchor.
        if mode in (CommandMode.RELATIVE, CommandMode.RELATIVE.value):
            if self.lock_collision_avoidance:
                anchor = self._last_pos
            else:
                current_positions = self.motion_manager.get_joint_pos_dict()
                anchor = [
                    current_positions.get(name) for name in self.joint_names
                ]
            if len(anchor) < len(positions) or any(value is None for value in anchor):
                if not self._missing_anchor_warned:
                    logger.warning(
                        f"{self.component_name} fine control ignored: "
                        "current gripper position is not synchronized"
                    )
                    self._missing_anchor_warned = True
                return False
            self._missing_anchor_warned = False
            positions = [a + d for a, d in zip(anchor, positions)]

        if self.lock_collision_avoidance:
            # Clip directly against URDF limits — pin_robot no longer has
            # these joints. Then publish without touching MotionManager.
            clipped = []
            for i, pos in enumerate(positions):
                if i < len(self.joint_names):
                    lo, hi = self.joint_pos_limits[i]
                    clipped.append(float(max(lo, min(hi, pos))))
                else:
                    clipped.append(pos)
            self._last_pos = list(clipped)
            command.output_components[self.component_name] = {
                "pos": clipped,
                "mode": CommandMode.ABSOLUTE.value,
            }
            return True

        # Update motion manager state with clipping
        updated_positions = self.motion_manager.get_joint_pos_dict()
        for i, pos in enumerate(positions):
            if i < len(self.joint_names):
                updated_positions[self.joint_names[i]] = pos

        # Clip to joint limits
        updated_positions = robot_utils.clip_joint_positions_to_limits(
            self.motion_manager.pin_robot, updated_positions
        )

        if updated_positions:
            self.motion_manager.set_joint_pos(updated_positions)
            clipped = [
                float(updated_positions[name])
                for name in self.joint_names[: len(positions)]
            ]
            self._last_pos = list(clipped)
            command.output_components[self.component_name] = {
                "pos": clipped,
                "mode": CommandMode.ABSOLUTE.value,
            }
            return True

        return False

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync hand to robot state.

        Syncs motion manager. When lock_collision_avoidance is True, the
        MotionManager doesn't carry hand joints, so we only cache the current
        robot pos as the relative-mode anchor.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        hand_pos = robot_joints.get(self.component_name, [])
        if not hand_pos:
            return
        if self.lock_collision_avoidance:
            self._last_pos = [
                hand_pos[i] for i in range(min(len(hand_pos), len(self.joint_names)))
            ]
            self._missing_anchor_warned = False
            return
        motion_manager_joints = {}
        for i, pos in enumerate(hand_pos):
            if i < len(self.joint_names):
                motion_manager_joints[self.joint_names[i]] = pos
        if motion_manager_joints:
            self.motion_manager.set_joint_pos(motion_manager_joints)
            self._last_pos = [
                hand_pos[i] for i in range(min(len(hand_pos), len(self.joint_names)))
            ]
            self._missing_anchor_warned = False
