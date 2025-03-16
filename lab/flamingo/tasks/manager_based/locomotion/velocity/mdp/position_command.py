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
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils.math import euler_xyz_from_quat


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformPositionCommand(CommandTerm):
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

    def __init__(self, cfg: UniformPositionCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Update the command buffer to include position z
        self.position_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.time_elapsed = torch.zeros(self.num_envs, device=self.device)  # Time tracker for each environment
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPositionCommand:\n"
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
        return self.position_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_link_quat_w)

        roll_error = self.position_command_b[:, 0] - roll
        pitch_error = self.position_command_b[:, 1] - pitch

        # Combine roll and pitch errors into a single tensor
        error = torch.stack((roll_error, pitch_error), dim=-1)

        # Compute the norm of the combined error vector
        self.metrics["position_error"] = torch.norm(error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # Linear velocity - x direction
        self.position_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.roll)
        # Linear velocity - y direction
        self.position_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch)
        # Position command - z direction
        roll_values = self.cfg.ranges.roll
        pitch_values = self.cfg.ranges.pitch
        if (roll_values[0] != 0.0 or roll_values[1] != 0.0) and (pitch_values[0] != 0.0 or pitch_values[1] != 0.0):
            # Generate categorical distribution
            probabilities = torch.ones(5, device=self.device) / 5  # Uniform probabilities for 2 categories
            categories_roll = torch.linspace(
                roll_values[0], roll_values[1], 5, device=self.device
            )  # Divide the range into 2 categories
            categories_pitch = torch.linspace(
                pitch_values[0], pitch_values[1], 5, device=self.device
            )
            sampled_categories_roll = categories_roll[torch.multinomial(probabilities, len(env_ids), replacement=True)]
            sampled_categories_pitch = categories_pitch[torch.multinomial(probabilities, len(env_ids), replacement=True)]
            self.position_command_b[env_ids, 0] = sampled_categories_roll
            self.position_command_b[env_ids, 1] = sampled_categories_pitch
        else:
            self.position_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
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
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UniformPositionCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPositionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""


    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    goal_pose_visualizer_cfg = None
    current_pose_visualizer_cfg = None