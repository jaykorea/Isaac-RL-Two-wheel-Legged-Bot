# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from lab.flamingo.isaaclab.isaaclab.envs.manager_based_constraint_rl_env import ManagerBasedConstraintRLEnv as ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_base_velocity_range(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, mod_range: dict, num_steps: int
):
    """
    Modifies the range of a command term (e.g., base_velocity) in the environment after a specific number of steps.

    Args:
        env: The environment instance.
        term_name: The name of the command term to modify (e.g., "base_velocity").
        end_range: The target range for the term (e.g., {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.5, 1.5)}).
        activation_step: The step count after which the range modification is applied.
    """
    # Check if the curriculum step exceeds the activation step
    if env.common_step_counter >= num_steps:
        # Get the term object
        command_term = env.command_manager.get_term(term_name)

        # Update the ranges directly
        for key, target_range in mod_range.items():
            if hasattr(command_term.cfg.ranges, key):
                setattr(command_term.cfg.ranges, key, target_range)
