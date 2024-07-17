# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def stand_still_base(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity on x or y when the command is zero, encouraging the robot to stand still."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :2]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate linear and angular velocity errors
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel), dim=1)

    ang_vel = asset.data.root_ang_vel_b[:, :2]
    ang_vel_error = torch.sum(torch.square(ang_vel), dim=1)

    # Penalize the linear and angular velocity errors
    velocity_penalty = (lin_vel_error + ang_vel_error) / std**2

    # Calculate deviation from origin position
    current_pos = asset.data.root_pos_w[:, :2]
    position_error = torch.sum(torch.square(current_pos - env.scene.env_origins[:, :2]), dim=1)

    # Penalize the deviation from the origin position
    position_penalty = position_error / std**2

    # Apply the penalty only when the command is zero
    penalty = (velocity_penalty) * is_zero_command.float()

    return penalty


def stand_still(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for standing still when the command is zero, penalizing movement, especially backward movement."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :3]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate wheel velocity error
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    wheel_vel_error = torch.sum(torch.abs(wheel_vel), dim=1)

    # Penalize backward movement by adding a higher penalty for negative velocities
    # backward_movement_penalty = torch.sum(torch.clamp(wheel_vel, max=0), dim=1)

    # Calculate the reward
    reward = wheel_vel_error / std**2

    # Make sure to only give non-zero reward where command is zero
    reward = reward * is_zero_command.float()

    return reward


def joint_align_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint mis-alignments.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    mis_aligned = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
    )

    return mis_aligned


def joint_soft_pos_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] * soft_ratio
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1] * soft_ratio
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_smoothness_hard(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using smoothing term."""
    sm1 = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    sm2 = torch.sum(
        torch.square(env.action_manager.action + env.action_manager.prev_action - 2 * env.action_manager.prev2_action),
        dim=1,
    )
    sm3 = 0.05 * torch.sum(torch.abs(env.action_manager.action), dim=1)

    return sm1 + sm2 + sm3


def base_height_range_reward(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height is within a specified range and penalize deviations."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_z = asset.data.root_pos_w[:, 2]

    # Check if the height is within the specified range
    in_range = (root_pos_z >= min_height) & (root_pos_z <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.abs(root_pos_z - torch.where(root_pos_z < min_height, min_height, max_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(root_pos_z), -out_of_range_penalty)

    return reward
