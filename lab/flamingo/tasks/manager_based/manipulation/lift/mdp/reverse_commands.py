# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.utils import configclass

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors  import FrameTransformer
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.managers import CommandTerm , CommandTermCfg

from isaaclab.markers import VisualizationMarkersCfg , VisualizationMarkers


class ReversePoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: ReversePoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ReversePoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.obstacle: RigidObject = env.scene[cfg.obstacle_name]
        self.ee: FrameTransformer = env.scene[cfg.ee_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- destiantion: (x, y, z, qw, qx, qy, qz) in world frame
        self.destination_w = torch.zeros_like(self.pose_command_b)
            
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self.cfg.resampling_time_range = (env.max_episode_length_s, env.max_episode_length_s)

    def __str__(self) -> str:
        msg = "ReversePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # object pose in robot base frame (rel)
        object_pos_abs = self.object.data.root_pos_w[env_ids]
        object_quat_abs = self.object.data.root_quat_w[env_ids]
        object_pos_rel, object_quat_rel = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids], self.robot.data.root_quat_w[env_ids],
            object_pos_abs, object_quat_abs
        )

        r = torch.empty(len(env_ids), device=self.device)

        # --- fill BASE-frame command (pose_command_b) ---
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = -object_pos_rel[:, 1]          # ✅ rel 사용
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        euler_angles = torch.zeros((len(env_ids), 3), device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        # --- optional: compute world destination for debug/visualization ---
        self.destination_w[env_ids, :3], self.destination_w[env_ids, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:],
        )

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_state_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_state_w[:, :3], body_link_state_w[:, 3:7])


@configclass
class ReversePoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = ReversePoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    object_name: str = MISSING
    """Name of the object in the environment related to the commands."""

    obstacle_name: str = MISSING
    """Name of the obstacle in the environment related to the commands."""

    ee_name: str = MISSING
    """Name of the end-effector frame in the environment related to the commands."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)