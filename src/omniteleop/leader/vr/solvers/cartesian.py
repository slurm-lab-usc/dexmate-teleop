# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Cartesian-space IK solvers for VR teleoperation.

Provides controllers that convert Cartesian-space targets (absolute or delta)
into joint-space positions via inverse kinematics using MotionManager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dexmotion.motion_manager import MotionManager
from loguru import logger
from scipy.spatial.transform import Rotation

try:
    import sapien
    import sapien.render
except ImportError:
    sapien = None

if TYPE_CHECKING:
    from dexcontrol.robot import Robot


class BaseIKController:
    """Controller for arm inverse kinematics operations.

    Robot-agnostic IK controller using MotionManager for FK/IK solving.
    Callers provide initial joint configuration and can update state
    externally via set_joint_state().

    Attributes:
        motion_manager (MotionManager): MotionManager instance for IK solving.
    """

    _ARM_SIDE_TO_EE_KEY = {"left": "L_ee", "right": "R_ee"}

    def __init__(
        self,
        initial_joint_configuration_dict: dict | None,
        visualize: bool = True,
        joint_regions_to_lock: list[str] | None = None,
        joints_to_lock: list[str] | None = None,
        target_frames: list[str] | None = None,
        init_global_ik: bool = False,
        mm_kwargs: dict | None = None,
    ):
        """Initialize the arm IK controller.

        Args:
            initial_joint_configuration_dict: Dictionary mapping joint names to
                initial positions, used to seed the MotionManager. If None,
                MotionManager falls back to the robot config's default initial
                joint configuration.
            visualize: Whether to enable visualization. Defaults to True.
            joint_regions_to_lock: Joint regions to lock (e.g. ["HEAD", "TORSO",
                "BASE"]). If None, uses MotionManager defaults.
            joints_to_lock: Explicit list of joint names to lock. Overrides
                joint_regions_to_lock and the config's additional_joints.
            target_frames: IK target frame names (e.g. ["L_ee", "R_ee"]).
                If None, uses MotionManager defaults.
            init_global_ik: Whether to initialize the global IK solver.
                Required for GlobalIKCartesianController. Defaults to False.
            mm_kwargs: Extra keyword arguments forwarded verbatim to
                MotionManager (caller overrides win). Use for specialised
                options (e.g. custom_urdf_path) without polluting named params.

        Raises:
            RuntimeError: If Motion Manager initialization fails.
        """
        self.visualize = visualize

        _mm_kwargs: dict = {
            "init_visualizer": self.visualize,
            "initial_joint_configuration_dict": initial_joint_configuration_dict,
            "visualizer_type": "sapien",
            "init_global_ik": init_global_ik,
        }
        if joints_to_lock is not None:
            _mm_kwargs["joints_to_lock"] = joints_to_lock
        elif joint_regions_to_lock is not None:
            _mm_kwargs["joint_regions_to_lock"] = joint_regions_to_lock
        if target_frames is not None:
            _mm_kwargs["target_frames"] = target_frames
        if mm_kwargs:
            _mm_kwargs.update(mm_kwargs)

        try:
            self.motion_manager = MotionManager(**_mm_kwargs)
            logger.success("Motion Manager setup complete")
        except Exception as e:
            logger.error(f"Error initializing Motion Manager: {e}")
            raise

        # IK target coordinate frames in SAPIEN visualizer
        self._target_actors: dict | None = None
        if self.visualize and sapien is not None:
            vis = self.motion_manager.visualizer
            if vis is not None and hasattr(vis, "scene") and vis.scene is not None:
                self._target_actors = {
                    "L_ee": self._create_axes_actor(vis.scene, "L_ee_target"),
                    "R_ee": self._create_axes_actor(vis.scene, "R_ee_target"),
                }

    def _validate_arm_side(self, arm_side: str) -> str:
        """Validate arm_side and return the corresponding EE key.

        Args:
            arm_side: "left" or "right".

        Returns:
            EE key string ("L_ee" or "R_ee").

        Raises:
            ValueError: If arm_side is invalid.
        """
        if arm_side not in self._ARM_SIDE_TO_EE_KEY:
            raise ValueError(
                f"Invalid arm side: {arm_side}. Must be 'left' or 'right'."
            )
        return self._ARM_SIDE_TO_EE_KEY[arm_side]

    def _ensure_ik_ready(self) -> None:
        """Raise RuntimeError if robot or IK solver is not initialized."""
        if self.motion_manager.pin_robot is None:
            raise RuntimeError("Robot is not initialized")
        if self.motion_manager.local_ik_solver is None:
            raise RuntimeError("Local IK solver is not initialized")

    def _get_current_ee_poses(self) -> dict:
        """Run FK and return a dict of EE poses keyed by EE name.

        Returns:
            Dict mapping "L_ee" and "R_ee" to their current FK poses.
        """
        current_qpos = self.motion_manager.get_joint_pos()
        assert isinstance(current_qpos, np.ndarray)
        return self.motion_manager.fk(
            frame_names=self.motion_manager.target_frames,
            qpos=current_qpos,
            update_robot_state=False,
        )

    @staticmethod
    def _create_axes_actor(
        scene: "sapien.Scene",
        name: str,
        length: float = 0.1,
        radius: float = 0.003,
    ) -> "sapien.Entity":
        """Create a coordinate frame actor (RGB XYZ capsule axes) in a SAPIEN scene.

        Capsules are aligned along X by default (PhysX convention).
        """
        builder = scene.create_actor_builder()
        half = length / 2.0

        axes = [
            # (local_position, local_quaternion_wxyz, color)
            # X red: identity rotation, offset along X
            ([half, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]),
            # Y green: 90° about Z, offset along Y
            ([0, half, 0], [0.7071068, 0, 0, 0.7071068], [0, 1, 0, 1]),
            # Z blue: -90° about Y, offset along Z
            ([0, 0, half], [0.7071068, 0, -0.7071068, 0], [0, 0, 1, 1]),
        ]
        for pos, quat, color in axes:
            mat = sapien.render.RenderMaterial()
            mat.base_color = color
            builder.add_capsule_visual(
                sapien.Pose(pos, quat),
                radius=radius,
                half_length=half,
                material=mat,
            )

        return builder.build_kinematic(name=name)

    def _update_target_visualization(self, target_poses_dict: dict) -> None:
        """Update SAPIEN coordinate frame actors to show IK targets."""
        if self._target_actors is None:
            return
        for ee_key, actor in self._target_actors.items():
            if ee_key not in target_poses_dict:
                continue
            pose_mat = target_poses_dict[ee_key]
            quat_xyzw = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
            # SAPIEN uses wxyz quaternion convention
            quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            actor.set_pose(sapien.Pose(pose_mat[:3, 3].tolist(), quat_wxyz))

    def _solve_and_apply(
        self, target_poses_dict: dict, arm_side: str
    ) -> tuple[np.ndarray, bool]:
        """Solve IK for target poses, apply to MotionManager, return arm joints.

        Args:
            target_poses_dict: Dict mapping EE keys to 4x4 target poses.
            arm_side: "left" or "right".

        Returns:
            Tuple of (joint positions for the requested arm (7,), is_in_collision).
        """
        self._update_target_visualization(target_poses_dict)
        qpos_dict, is_in_collision, _is_within_joint_limits = (
            self.motion_manager.local_ik_solver.solve_ik(target_poses_dict)
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        arm_prefix = arm_side[0].upper()
        arm_joint_names = [f"{arm_prefix}_arm_j{i + 1}" for i in range(7)]
        return np.array([qpos_dict[name] for name in arm_joint_names]), is_in_collision

    def move_delta_cartesian_both(
        self,
        left_delta: tuple[np.ndarray, np.ndarray] | None,
        right_delta: tuple[np.ndarray, np.ndarray] | None,
        left_reference: np.ndarray | None = None,
        right_reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """Solve IK for both arms simultaneously in a single IK call.

        Avoids sequential solving artifacts: both EE targets are included in
        one solve_ik call, so neither arm's result depends on the other's.

        Args:
            left_delta: (delta_xyz, delta_R) for left arm, or None to hold at FK.
            right_delta: (delta_xyz, delta_R) for right arm, or None to hold at FK.
            left_reference: Optional 4x4 reference pose for left arm delta.
            right_reference: Optional 4x4 reference pose for right arm delta.

        Returns:
            Tuple of (left_arm_qpos(7)|None, right_arm_qpos(7)|None, is_in_collision).
            A side is None if its delta was None.
        """
        self._ensure_ik_ready()

        ee_poses = self._get_current_ee_poses()
        deltas = {"left": left_delta, "right": right_delta}
        references = {"left": left_reference, "right": right_reference}

        target_poses_dict = {}
        for side, ee_key in self._ARM_SIDE_TO_EE_KEY.items():
            delta = deltas[side]
            if delta is not None:
                delta_xyz, delta_R = delta
                ref = references[side]
                ee_pose_np = (
                    ref.copy() if ref is not None else ee_poses[ee_key].np.copy()
                )
                ee_pose_np[:3, :3] = delta_R @ ee_pose_np[:3, :3]
                ee_pose_np[:3, 3] += delta_xyz
                target_poses_dict[ee_key] = ee_pose_np
            else:
                target_poses_dict[ee_key] = ee_poses[ee_key].np

        self._update_target_visualization(target_poses_dict)
        qpos_dict, is_in_collision, _ = self.motion_manager.local_ik_solver.solve_ik(
            target_poses_dict
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        results: dict[str, np.ndarray | None] = {}
        for side, delta in deltas.items():
            if delta is not None:
                arm_prefix = side[0].upper()
                arm_joint_names = [f"{arm_prefix}_arm_j{i + 1}" for i in range(7)]
                results[side] = np.array([qpos_dict[name] for name in arm_joint_names])
            else:
                results[side] = None

        return results["left"], results["right"], is_in_collision

    def move_absolute_cartesian(
        self,
        target_pose: np.ndarray,
        arm_side: str,
    ) -> tuple[np.ndarray, bool]:
        """Move the specified arm to an absolute Cartesian target pose.

        Args:
            target_pose: Absolute target as a 4x4 homogeneous
                transformation matrix for the end effector.
            arm_side: Which arm to move ("left" or "right").

        Returns:
            Tuple of (target joint positions for the specified arm (7,), is_in_collision).

        Raises:
            ValueError: If an invalid arm side is specified.
            RuntimeError: If robot or IK solver is not initialized.
        """
        self._ensure_ik_ready()
        ee_key = self._validate_arm_side(arm_side)

        ee_poses = self._get_current_ee_poses()

        target_poses_dict = {
            k: target_pose if k == ee_key else ee_poses[k].np
            for k in self._ARM_SIDE_TO_EE_KEY.values()
        }

        return self._solve_and_apply(target_poses_dict, arm_side)

    def set_joint_state(
        self, left_arm_qpos: np.ndarray, right_arm_qpos: np.ndarray
    ) -> None:
        """Update MotionManager joint state from external data.

        Args:
            left_arm_qpos: Joint positions for the left arm.
            right_arm_qpos: Joint positions for the right arm.
        """
        self.motion_manager.left_arm.set_joint_pos(left_arm_qpos.tolist())
        self.motion_manager.right_arm.set_joint_pos(right_arm_qpos.tolist())

    def set_hand_positions(
        self, left_positions: list[float] | None, right_positions: list[float] | None
    ) -> None:
        """Set hand joint positions via MotionManager hand components.

        Works with any hand DOF count (gripper: 1, f5d6: 6, etc.).
        Requires hand joints to be unlocked (not in joint_regions_to_lock).

        Args:
            left_positions: Left hand joint positions, or None to skip.
            right_positions: Right hand joint positions, or None to skip.
        """
        if left_positions is not None:
            self.motion_manager.left_hand.set_joint_pos(left_positions)
        if right_positions is not None:
            self.motion_manager.right_hand.set_joint_pos(right_positions)


class GlobalCartesianController(BaseIKController):
    """IK controller that solves for absolute wrist poses directly.

    Takes absolute 4x4 EE target poses (in robot frame) for both arms
    simultaneously and solves IK. No delta computation, no reference poses —
    just direct global positioning from VR wrist poses.
    """

    def solve_global(
        self,
        target_poses: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """Solve IK for absolute EE target poses on both arms.

        Args:
            target_poses: Dict mapping side ("left", "right") to 4x4 target
                poses in robot frame. Missing sides are held at current FK.

        Returns:
            Tuple of (left_arm_qpos(7), right_arm_qpos(7), is_in_collision).
            A side is None if it was not in target_poses.
        """
        self._ensure_ik_ready()

        ee_poses = self._get_current_ee_poses()

        target_poses_dict = {}
        for side, ee_key in self._ARM_SIDE_TO_EE_KEY.items():
            if side in target_poses:
                target_poses_dict[ee_key] = target_poses[side]
            else:
                target_poses_dict[ee_key] = ee_poses[ee_key].np

        self._update_target_visualization(target_poses_dict)
        qpos_dict, is_in_collision, _is_within_joint_limits = (
            self.motion_manager.local_ik_solver.solve_ik(target_poses_dict)
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        results = {}
        for side in self._ARM_SIDE_TO_EE_KEY:
            if side in target_poses:
                arm_prefix = side[0].upper()
                arm_joint_names = [f"{arm_prefix}_arm_j{i + 1}" for i in range(7)]
                results[side] = np.array([qpos_dict[name] for name in arm_joint_names])
            else:
                results[side] = None

        return results.get("left"), results.get("right"), is_in_collision


class GlobalIKCartesianController(GlobalCartesianController):
    """IK controller that uses the global (optimization-based) solver.

    Same interface as GlobalCartesianController but routes through
    MotionManager.global_ik() instead of the local Pink QP solver.
    The global solver can escape local minima at the cost of higher
    solve times (~50-200ms vs ~2ms).
    """

    def __init__(
        self,
        initial_joint_configuration_dict: dict,
        visualize: bool = True,
        joint_regions_to_lock: list[str] | None = None,
        joints_to_lock: list[str] | None = None,
        target_frames: list[str] | None = None,
        mm_kwargs: dict | None = None,
    ):
        super().__init__(
            initial_joint_configuration_dict=initial_joint_configuration_dict,
            visualize=visualize,
            joint_regions_to_lock=joint_regions_to_lock,
            joints_to_lock=joints_to_lock,
            target_frames=target_frames,
            init_global_ik=True,
            mm_kwargs=mm_kwargs,
        )

    def solve_global(
        self,
        target_poses: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """Solve IK for absolute EE target poses using global optimization.

        Args:
            target_poses: Dict mapping side ("left", "right") to 4x4 target
                poses in robot frame. Missing sides are held at current FK.

        Returns:
            Tuple of (left_arm_qpos(7), right_arm_qpos(7), is_in_collision).
            A side is None if it was not in target_poses.
        """
        self._ensure_ik_ready()

        ee_poses = self._get_current_ee_poses()

        target_poses_dict = {}
        for side, ee_key in self._ARM_SIDE_TO_EE_KEY.items():
            if side in target_poses:
                target_poses_dict[ee_key] = target_poses[side]
            else:
                target_poses_dict[ee_key] = ee_poses[ee_key].np

        self._update_target_visualization(target_poses_dict)
        qpos_dict, is_in_collision, _is_within_joint_limits = (
            self.motion_manager.global_ik(target_pose=target_poses_dict)
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        results = {}
        for side in self._ARM_SIDE_TO_EE_KEY:
            if side in target_poses:
                arm_prefix = side[0].upper()
                arm_joint_names = [f"{arm_prefix}_arm_j{i + 1}" for i in range(7)]
                results[side] = np.array([qpos_dict[name] for name in arm_joint_names])
            else:
                results[side] = None

        return results.get("left"), results.get("right"), is_in_collision


class RealRobotIKController(BaseIKController):
    """IK controller for real robots via dexcontrol.

    Extends BaseIKController with dexcontrol.Robot integration for reading
    initial joint state and syncing with the real robot.

    Attributes:
        bot (Robot): Robot instance to control.
    """

    def __init__(
        self,
        bot: Robot | None = None,
        visualize: bool = True,
        joint_regions_to_lock: list[str] | None = None,
        mm_kwargs: dict | None = None,
    ):
        """Initialize the real robot IK controller.

        Args:
            bot: Robot instance. If None, a new Robot instance will be created.
            visualize: Whether to enable visualization.
            joint_regions_to_lock: Joint regions to lock (e.g. ["HEAD", "TORSO",
                "BASE"]). If None, uses MotionManager defaults.
            mm_kwargs: Extra keyword arguments forwarded verbatim to
                MotionManager via BaseIKController (e.g. ``custom_urdf_path``).
        """
        from omniteleop.read_only_robot import ReadOnlyRobot

        if bot is None:
            self.bot = ReadOnlyRobot()
        else:
            self.bot = bot

        # Get initial joint positions from real robot
        try:
            if self.bot.robot_model == "vega_1u":
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm"]
                )
            else:
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm", "torso"]
                )
            logger.info("Initial joint positions retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting joint positions: {e}")
            raise

        super().__init__(
            initial_joint_configuration_dict=joint_pos_dict,
            visualize=visualize,
            joint_regions_to_lock=joint_regions_to_lock,
            mm_kwargs=mm_kwargs,
        )

    def sync_state(self) -> None:
        """Sync MotionManager joint positions with real robot arm positions."""
        try:
            left_arm_qpos = self.bot.left_arm.get_joint_pos()
            right_arm_qpos = self.bot.right_arm.get_joint_pos()
            self.set_joint_state(left_arm_qpos, right_arm_qpos)
        except Exception as e:
            logger.warning(f"Error syncing with robot: {e}")


class ElbowConstrainedIKController(BaseIKController):
    """IK controller with elbow orientation constraints for 7-DOF redundancy resolution.

    Extends BaseIKController by adding elbow frames as orientation-only IK targets.
    This resolves the 7-DOF arm null-space ambiguity: given only an EE pose, the
    elbow can lie anywhere on a redundancy circle. Adding elbow orientation
    constraints pins the elbow direction.

    The elbow frames use position_cost=0 (position unconstrained) and
    orientation_cost=50 (orientation enforced).
    """

    _ARM_SIDE_TO_ELBOW_KEY = {"left": "L_arm_l4", "right": "R_arm_l4"}

    def __init__(
        self,
        initial_joint_configuration_dict: dict,
        visualize: bool = True,
        joint_regions_to_lock: list[str] | None = None,
        joints_to_lock: list[str] | None = None,
        elbow_frames: list[str] | None = None,
        mm_kwargs: dict | None = None,
    ):
        """Initialize the elbow-constrained IK controller.

        Args:
            initial_joint_configuration_dict: Dictionary mapping joint names to
                initial positions, used to seed the MotionManager.
            visualize: Whether to enable visualization.
            joint_regions_to_lock: Joint regions to lock.
            joints_to_lock: Explicit list of joint names to lock.
            elbow_frames: Robot frame names for elbow links. Defaults to
                ["L_arm_l4", "R_arm_l4"].
            mm_kwargs: Extra keyword arguments forwarded verbatim to
                MotionManager via BaseIKController.
        """
        if elbow_frames is None:
            elbow_frames = list(self._ARM_SIDE_TO_ELBOW_KEY.values())
        self._elbow_frames = elbow_frames

        # Include elbow frames in IK target frames
        target_frames = ["L_ee", "R_ee"] + elbow_frames

        super().__init__(
            initial_joint_configuration_dict=initial_joint_configuration_dict,
            visualize=visualize,
            joint_regions_to_lock=joint_regions_to_lock,
            joints_to_lock=joints_to_lock,
            target_frames=target_frames,
            mm_kwargs=mm_kwargs,
        )

        # Configure elbow tasks as orientation-only (position_cost=0)
        self._configure_elbow_orientation_only()

    def _configure_elbow_orientation_only(self) -> None:
        """Set elbow FrameTasks to orientation-only (position_cost=0, orientation_cost=50)."""
        solver = self.motion_manager.local_ik_solver
        for frame_name in self._elbow_frames:
            task = solver.task_dict[frame_name]
            task.position_cost = 0.0
            task.orientation_cost = 50.0
        logger.info(
            f"Elbow orientation-only constraints configured: {self._elbow_frames}"
        )

    def move_delta_cartesian_with_elbow_both(
        self,
        left_delta: tuple[np.ndarray, np.ndarray] | None,
        right_delta: tuple[np.ndarray, np.ndarray] | None,
        elbow_orientations: dict[str, np.ndarray],
        left_reference: np.ndarray | None = None,
        right_reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """Solve elbow-constrained IK for both arms simultaneously.

        Both EE targets and both elbow orientation constraints are included in
        a single solve_ik call, avoiding sequential solving artifacts.

        Args:
            left_delta: (delta_xyz, delta_R) for left arm, or None to hold at FK.
            right_delta: (delta_xyz, delta_R) for right arm, or None to hold at FK.
            elbow_orientations: Dict mapping elbow frame names to 3x3 orientation
                matrices in robot frame. E.g. {"L_arm_l4": R_3x3, "R_arm_l4": R_3x3}.
            left_reference: Optional 4x4 reference pose for left arm delta.
            right_reference: Optional 4x4 reference pose for right arm delta.

        Returns:
            Tuple of (left_arm_qpos(7)|None, right_arm_qpos(7)|None, is_in_collision).
            A side is None if its delta was None.
        """
        self._ensure_ik_ready()

        current_qpos = self.motion_manager.get_joint_pos()
        all_poses = self.motion_manager.fk(
            frame_names=self.motion_manager.target_frames,
            qpos=current_qpos,
            update_robot_state=False,
        )

        deltas = {"left": left_delta, "right": right_delta}
        references = {"left": left_reference, "right": right_reference}

        target_poses_dict = {}
        for side, ee_key in self._ARM_SIDE_TO_EE_KEY.items():
            delta = deltas[side]
            if delta is not None:
                delta_xyz, delta_R = delta
                ref = references[side]
                ee_pose_np = (
                    ref.copy() if ref is not None else all_poses[ee_key].np.copy()
                )
                ee_pose_np[:3, :3] = delta_R @ ee_pose_np[:3, :3]
                ee_pose_np[:3, 3] += delta_xyz
                target_poses_dict[ee_key] = ee_pose_np
            else:
                target_poses_dict[ee_key] = all_poses[ee_key].np

        for frame_name in self._elbow_frames:
            elbow_pose = np.eye(4)
            elbow_pose[:3, 3] = all_poses[frame_name].np[:3, 3]
            if frame_name in elbow_orientations:
                elbow_pose[:3, :3] = elbow_orientations[frame_name]
            else:
                elbow_pose[:3, :3] = all_poses[frame_name].np[:3, :3]
            target_poses_dict[frame_name] = elbow_pose

        self._update_target_visualization(target_poses_dict)
        qpos_dict, is_in_collision, _ = self.motion_manager.local_ik_solver.solve_ik(
            target_poses_dict
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        results: dict[str, np.ndarray | None] = {}
        for side, delta in deltas.items():
            if delta is not None:
                arm_prefix = side[0].upper()
                arm_joint_names = [f"{arm_prefix}_arm_j{i + 1}" for i in range(7)]
                results[side] = np.array([qpos_dict[name] for name in arm_joint_names])
            else:
                results[side] = None

        return results["left"], results["right"], is_in_collision


class RealRobotElbowConstrainedIKController(ElbowConstrainedIKController):
    """Elbow-constrained IK controller for real robots via dexcontrol.

    Extends ElbowConstrainedIKController with dexcontrol.Robot integration.
    """

    def __init__(
        self,
        bot: Robot | None = None,
        visualize: bool = True,
        elbow_frames: list[str] | None = None,
    ):
        """Initialize the real robot elbow-constrained IK controller.

        Args:
            bot: Robot instance. If None, a new Robot instance will be created.
            visualize: Whether to enable visualization.
            elbow_frames: Robot frame names for elbow links.
        """
        from omniteleop.read_only_robot import ReadOnlyRobot

        if bot is None:
            self.bot = ReadOnlyRobot()
        else:
            self.bot = bot

        try:
            if self.bot.robot_model == "vega_1u":
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm"]
                )
            else:
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm", "torso"]
                )
            logger.info("Initial joint positions retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting joint positions: {e}")
            raise

        super().__init__(
            initial_joint_configuration_dict=joint_pos_dict,
            visualize=visualize,
            elbow_frames=elbow_frames,
        )

    def sync_state(self) -> None:
        """Sync MotionManager joint positions with real robot arm positions."""
        try:
            left_arm_qpos = self.bot.left_arm.get_joint_pos()
            right_arm_qpos = self.bot.right_arm.get_joint_pos()
            self.set_joint_state(left_arm_qpos, right_arm_qpos)
        except Exception as e:
            logger.warning(f"Error syncing with robot: {e}")
