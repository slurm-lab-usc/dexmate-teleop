#!/usr/bin/env python3
"""Modular end-effector controllers for different hand/gripper types.

This module provides a unified interface for controlling different
types of end-effectors (5-finger hands, grippers, etc) from JoyCon input.
"""

from __future__ import annotations

from typing import Dict, Any
from dataclasses import dataclass

import numpy as np
from loguru import logger

from omniteleop.follower.input_handlers.control.end_effector import (
    AbstractEndEffectorController,
)


@dataclass
class VREndEffectorInput:
    """Parsed VR input for end-effector control."""

    stick_x: float  # -1 to 1
    stick_y: float  # -1 to 1
    buttons: Dict[str, bool]  # Button states
    index_trigger: float = 0.0  # 0.0 to 1.0


class HandF5D6Controller(AbstractEndEffectorController):
    """Controller for 5-finger/6-DoF dexterous hands using predefined poses.

    VR Control Mapping:
    - Index Trigger: Interpolate between open (0) and close (1) poses
    - Simple and reliable pose-based control
    """

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize 5-finger hand controller.

        Args:
            side: "left" or "right"
            config: Hand configuration with poses
        """
        super().__init__(side, config)

        self.predefined_poses = config.get("poses", {})

        # Get open and close poses for interpolation
        self.open_pose = np.array(self.predefined_poses.get("open", [0.0] * 6))
        self.close_pose = np.array(self.predefined_poses.get("close", [0.0] * 6))
        self.dof = len(self.open_pose)

        logger.info(
            f"HandF5D6Controller ({side}): Using pose interpolation - open/close"
        )

    def process_input(self, vr_input: VREndEffectorInput) -> Dict[str, float]:
        """Process VR input for 5-finger hand control.

        Simple control: Index trigger interpolates between open and close poses.
        - Trigger = 0: Open pose
        - Trigger = 1: Close pose

        Args:
            vr_input: Parsed VR input data

        Returns:
            Tuple of (joint_positions, mode) - absolute positions
        """
        # Interpolate between open and close poses based on index trigger
        # trigger 0 = open, trigger 1 = close
        t = vr_input.index_trigger

        # Linear interpolation: open * (1-t) + close * t
        positions = self.open_pose * (1.0 - t) + self.close_pose * t

        return (positions.tolist(), "absolute")

    def get_predefined_poses(self, pose_name: str) -> Dict[str, float]:
        """Get predefined pose by name."""
        return self.predefined_poses.get(pose_name, [])


class GripperController(AbstractEndEffectorController):
    """Controller for simple grippers using predefined poses.

    VR Control Mapping:
    - Index Trigger: Interpolate between open (0) and close (1) poses
    - Simple and reliable pose-based control (same as HandF5D6Controller)
    """

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize gripper controller.

        Args:
            side: "left" or "right"
            config: Gripper configuration with poses
        """
        super().__init__(side, config)

        self.predefined_poses = config.get("poses", {})

        # Get open and close poses for interpolation
        self.open_pose = np.array(self.predefined_poses.get("open", [0.78]))
        self.close_pose = np.array(self.predefined_poses.get("close", [0.0]))
        self.dof = len(self.open_pose)

        logger.info(
            f"GripperController ({side}): Using pose interpolation - open/close"
        )

    def process_input(self, vr_input: VREndEffectorInput) -> Dict[str, float]:
        """Process VR input for gripper control.

        Simple control: Index trigger interpolates between open and close poses.
        - Trigger = 0: Open pose
        - Trigger = 1: Close pose

        Args:
            vr_input: Parsed VR input data

        Returns:
            Tuple of (joint_positions, mode) - absolute positions
        """
        # Interpolate between open and close poses based on index trigger
        # trigger 0 = open, trigger 1 = close
        t = vr_input.index_trigger

        # Linear interpolation: open * (1-t) + close * t
        positions = self.open_pose * (1.0 - t) + self.close_pose * t

        return (positions.tolist(), "absolute")


class NoEndEffector(AbstractEndEffectorController):
    """Placeholder for arms without end-effectors."""

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize no-op end-effector."""
        super().__init__(side, config)

    def process_input(self, vr_input: VREndEffectorInput) -> Dict[str, float]:
        """No-op - returns empty dict."""
        return self.joint_positions


def create_end_effector(
    side: str, config: Dict[str, Any]
) -> AbstractEndEffectorController:
    """Factory function to create appropriate end-effector controller.

    Args:
        side: "left" or "right"
        config: End-effector configuration from robot config

    Returns:
        Appropriate EndEffectorController instance
    """
    effector_type = config.get("type", "none")

    if effector_type == "hand_f5d6":
        logger.info(f"Created 5-finger hand controller for {side} side")
        return HandF5D6Controller(side, config)
    elif effector_type == "gripper":
        logger.info(f"Created gripper controller for {side} side")
        return GripperController(side, config)
    else:
        logger.info(f"No end-effector configured for {side} side")
        return NoEndEffector(side, config)
