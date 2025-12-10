# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_fall_down(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,  # m/s
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
) -> torch.Tensor:
    """Terminate envs where obstacle roll/pitch exceeds threshold.

    Returns:
        torch.BoolTensor of shape (num_envs,)
    """
    obstacle: RigidObject = env.scene[obstacle_cfg.name]

    obstacle_lin_vel_b = obstacle.data.root_lin_vel_b[:, :3]  # (num_envs, 3)

    lin_vel_norm = torch.norm(obstacle_lin_vel_b[:, :], dim=-1)

    done = ( lin_vel_norm > threshold)   # (num_envs,)
    return done
