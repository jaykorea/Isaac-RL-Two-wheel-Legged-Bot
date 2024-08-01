from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.envs.mdp import UniformVelocityCommand
from omni.isaac.lab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class UniformVelocityWithZCommand(UniformVelocityCommand):
    r"""Command generator that generates a velocity command in SE(2) with an additional position command in Z from uniform distribution.

    The command comprises of a linear velocity in x and y direction, an angular velocity around
    the z-axis, and a position command in z. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    def __init__(self, cfg: UniformVelocityWithZCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator with an additional position command in Z.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # Call the base class constructor
        super().__init__(cfg, env)

        # Update the command buffer to include position z
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator with position z."""
        msg = super().__str__()
        msg += f"\n\tPosition z range: {self.cfg.ranges.pos_z}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity and position z command in the base frame. Shape is (num_envs, 4)."""
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # Linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # Linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # Angular velocity - z direction
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # Position command - z direction
        self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.pos_z)
        # Heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity and position z command."""
        super()._update_command()
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :3] = 0.0
        r = torch.empty(len(standing_env_ids), device=self.device)
        self.vel_command_b[standing_env_ids, 3] = r.uniform_(*self.cfg.ranges.pos_z)
        # self.robot.data.root_pos_w[
        # standing_env_ids, 2
        # ]  # Maintain current z position


@configclass
class UniformVelocityWithZCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity with z command generator."""

    class_type: type = UniformVelocityWithZCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity and position commands."""

        pos_z: tuple[float, float] = MISSING  # min max [m]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity and position commands."""
