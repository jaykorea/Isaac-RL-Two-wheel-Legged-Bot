# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_binary(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    unoise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state using binary sampling with uniform noise.

    * For each axis, sample either the min or max value (binary sampling).
    * Apply uniform noise within ±unoise to the sampled value.
    * Orientation uses binary sampling in Euler XYZ and multiplied on top of the default root orientation.
    * Velocity uses binary + noise in the same manner.

    Args:
        env: Environment instance.
        env_ids: IDs of environments to reset.
        pose_range: Dict for position/orientation ranges {x,y,z,roll,pitch,yaw}.
        velocity_range: Dict for linear/angular velocity ranges {x,y,z,roll,pitch,yaw}.
        unoise: Noise scale. If e.g. 0.1, sampled value += uniform(-0.1, 0.1).
        asset_cfg: Asset configuration.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    # ──────────────────────────────
    # ① Binary sampling util
    # ──────────────────────────────
    def sample_binary_with_noise(ranges: torch.Tensor, shape):
        """
        ranges: Tensor[[min,max], ...] → shape (#dimensions, 2)
        shape: (N_envs, #dimensions)
        """
        device = ranges.device
        N, D = shape
        mins = ranges[:, 0].unsqueeze(0).repeat(N, 1)
        maxs = ranges[:, 1].unsqueeze(0).repeat(N, 1)

        # Binary choice (min or max)
        choice = (torch.rand((N, D), device=device) > 0.5).float()
        samples = choice * maxs + (1.0 - choice) * mins

        # Apply noise if unoise > 0
        if unoise > 0:
            noise = (2 * unoise) * torch.rand((N, D), device=device) - unoise
            samples += noise

        return samples

    # ──────────────────────────────
    # ② Pose (position + orientation)
    # ──────────────────────────────
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in pose_keys], device=asset.device)
    pose_samples = sample_binary_with_noise(pose_ranges, (len(env_ids), 6))

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # ──────────────────────────────
    # ③ Velocity (linear + angular)
    # ──────────────────────────────
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in pose_keys], device=asset.device)
    velocity_samples = sample_binary_with_noise(vel_ranges, (len(env_ids), 6))
    velocities = root_states[:, 7:13] + velocity_samples

    # ──────────────────────────────
    # ④ Apply to simulation
    # ──────────────────────────────
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)