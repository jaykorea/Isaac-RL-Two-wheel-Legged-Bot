# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg , VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG,  BLUE_ARROW_X_MARKER_CFG



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformWorldframePositionCommand(CommandTerm):
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

    def __init__(self, cfg: UniformWorldframePositionCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Update the command buffer to include position xyz
        self.position_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.time_elapsed = torch.zeros(self.num_envs, device=self.device)  # Time tracker for each environment
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.current_robot_position = self.robot.data.root_pos_w

    def __str__(self) -> str:
        msg = "UniformWorldframePositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. 

        These command elements correspond to the world frame coordinate (x, y, z).
        """
        return self.position_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        x_error = self.position_command_b[:,0] - self.current_robot_position[:,0] 
        y_error = self.position_command_b[:,1] - self.current_robot_position[:,1]
        z_error = self.position_command_b[:,2] - self.current_robot_position[:,2]

        # Combine roll and pitch errors into a single tensor
        #error = torch.stack((x_error, y_error, z_error), dim=-1)

        # Compute the norm of the combined error vector
        self.metrics["position_x_error"] = abs(x_error)
        self.metrics["position_y_error"] = abs(y_error)
        self.metrics["position_z_error"] = abs(z_error)

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # world frame x command
        self.position_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.world_x)
        # world frame y command
        self.position_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.world_y)      
        # world frame z command
        self.position_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.world_z)
        
        # Position command - z direction
        x_values = self.cfg.ranges.world_x
        y_values = self.cfg.ranges.world_y
        z_values = self.cfg.ranges.world_z
        
        if (x_values[0] != 0.0 or y_values[1] != 0.0) and (y_values[0] != 0.0 or y_values[1] != 0.0):
            # Generate categorical distribution
            probabilities = torch.ones(5, device=self.device) / 5  # Uniform probabilities for 2 categories
            categories_x = torch.linspace(
                x_values[0], x_values[1], 5, device=self.device
            )  # Divide the range into 2 categories
            categories_y = torch.linspace(
                y_values[0], y_values[1], 5, device=self.device
            )

            sampled_categories_x = categories_x[torch.multinomial(probabilities, len(env_ids), replacement=True)]
            sampled_categories_y = categories_y[torch.multinomial(probabilities, len(env_ids), replacement=True)]
  
            self.position_command_b[env_ids, 0] = sampled_categories_x
            self.position_command_b[env_ids, 1] = sampled_categories_y

        else:
            self.position_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        
        # update current robot position
        self.current_robot_position = self.robot.data.root_pos_w
        
        self.time_elapsed += self._env.step_dt  # Increment time elapsed by time step
        #* Determine environments still within the initial 2.0 seconds
        initial_phase_env_ids = (self.time_elapsed < 2.0).nonzero(as_tuple=False).flatten()
        if len(initial_phase_env_ids) > 0:
            self.position_command_b[initial_phase_env_ids] = 0.0  # Set commands to zero

        #* Reset time_elapsed for reset environments
        reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.time_elapsed[reset_env_ids] = 0.0

        # Enforce standing for standing environments
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.position_command_b[standing_env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer") or not hasattr(self, "current_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return

        current_robot_position = self.current_robot_position.clone()
        current_robot_position[:,2] += 0.6
        
        goal_position = self.position_command_b.clone()
        goal_position[:,2] += 1.0 
        
        self.goal_pose_visualizer.visualize(self.position_command_b)
        self.current_pose_visualizer.visualize(self.current_robot_position)




GREEN_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/position_command",
    markers={
        "cuboid": sim_utils.SphereCfg(
            radius=0.075,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

RED_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/current_position",
    markers={
        "cuboid": sim_utils.SphereCfg(
            radius=0.075,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)


@configclass
class UniformWorldframePositionCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformWorldframePositionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""


    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        world_x: tuple[float, float] = MISSING
        """Range for the x coordinate (in m)."""

        world_y: tuple[float, float] = MISSING
        """Range for the y coordinate (in m)."""
        
        world_z: tuple[float, float] = MISSING
        """Range for the z coordinate (in m)."""              

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    goal_pose_visualizer_cfg : VisualizationMarkersCfg = GREEN_CUBOID_MARKER_CFG
    current_pose_visualizer_cfg : VisualizationMarkersCfg= RED_CUBOID_MARKER_CFG