# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def track_pos_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    relative: bool = False,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the current z position of the robot's base
    current_pos_z = root_asset.data.root_pos_w[:, 2]
    # Get the mean z position of the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2].mean(dim=1)

    # Calculate the relative height difference
    if relative:
        pos_z = current_pos_z - wheel_pos_z
        contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
        # Check if any of the body parts are in contact
        in_contact = torch.all(contact_time > 0.0, dim=1)
        # Assign a small constant reward if in contact
        wheel_ground_reward = torch.where(in_contact, std**2, 0.0)
    else:
        pos_z = current_pos_z
        wheel_ground_reward = torch.zeros_like(current_pos_z)

    # Get the command z position relative to wheels from the command manager
    command_pos_z = env.command_manager.get_command(command_name)[:, 3]

    # Compute the error between the current height difference and the commanded height difference
    pos_z_error = torch.square(command_pos_z - pos_z)

    # Return the reward based on the exponential of the error
    # return torch.exp(-pos_z_error / std**2) + wheel_ground_reward
    return (pos_z_error / std**2) + wheel_ground_reward


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


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


class FlamingoAirTimeReward(ManagerTermBase):
    """Reward for longer feet air and contact time with stuck detection and reward for locomotion."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.stuck_threshold: float = cfg.params.get("stuck_threshold", 0.1)
        self.stuck_duration: int = cfg.params.get("stuck_duration", 5)
        self.threshold: float = cfg.params.get("threshold", 0.2)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.stuck_counter = torch.zeros(self.asset.data.root_lin_vel_b.shape[0], device=self.asset.device)

        if not self.contact_sensor.cfg.track_air_time:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stuck_threshold: float,
        stuck_duration: int,
        threshold: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward calculates the air-time for the feet and applies a reward when the robot is stuck.

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # Extract the necessary sensor data
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

        # Compute the base movement command and its progress
        base_velocity_tensor = env.command_manager.get_command("base_velocity")[:, :3]
        progress = torch.norm(base_velocity_tensor - self.asset.data.root_lin_vel_b, dim=1)
        is_stuck = progress > self.stuck_threshold  # Detect lack of progress

        # Manage the stuck counter and determine stuck status
        self.stuck_counter = torch.where(is_stuck, self.stuck_counter + 1, torch.zeros_like(self.stuck_counter))
        stuck = self.stuck_counter >= self.stuck_duration
        stuck = stuck.unsqueeze(1)

        # Compute the reward based on air time and first contact when stuck
        stuck_air_time_reward = torch.sum((last_air_time - self.threshold) * first_contact * stuck.float(), dim=1)
        # Ensure no reward is given if there is no movement command
        stuck_air_time_reward *= torch.norm(base_velocity_tensor[:, :2], dim=1) > 0.1

        # # Foot clearance reward
        # foot_z_target_error = torch.square(self.asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - self.target_height)
        # foot_velocity_tanh = torch.tanh(
        #     tanh_mult * torch.norm(self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
        # )
        # foot_clearance_reward = (
        #     torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std) * stuck.float()
        # )

        # Final reward: Encourage lifting legs when stuck
        reward = stuck_air_time_reward  # + foot_clearance_reward

        return reward


def stand_origin_base(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity on x or y when the command is zero, encouraging the robot to stand still."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate linear and angular velocity errors
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel), dim=1)

    ang_vel = asset.data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(ang_vel)

    # Penalize the linear and angular velocity errors
    velocity_penalty = lin_vel_error + ang_vel_error

    # Calculate deviation from origin position
    current_pos = asset.data.root_pos_w[:, :2]
    position_error = torch.sum(torch.square(current_pos - env.scene.env_origins[:, :2]), dim=1)

    # Penalize the deviation from the origin position
    position_penalty = position_error

    # Apply the penalty only when the command is zero
    penalty = (velocity_penalty + position_penalty) * is_zero_command.float()

    return penalty


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


def force_action_zero(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    force_zero_action = torch.abs(env.action_manager.action[:, asset_cfg.joint_ids])
    return torch.sum(force_zero_action, dim=1)


def base_height_range_l2(
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
    out_of_range_penalty = torch.square(root_pos_z - torch.where(root_pos_z < min_height, max_height, min_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(root_pos_z), -out_of_range_penalty)

    return reward


def base_height_range_relative_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height is within a specified range and penalize deviations."""
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]

    root_pos_z = root_asset.data.root_pos_w[:, 2]
    # Get the mean z position of the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2].mean(dim=1)

    # Calculate the height difference
    height_diff = root_pos_z - wheel_pos_z

    # Check if the height difference is within the specified range
    in_range = (height_diff >= min_height) & (height_diff <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(height_diff - torch.where(height_diff < min_height, max_height, min_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(height_diff), -out_of_range_penalty)

    return reward


def base_height_dynamic_wheel_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height relative to the furthest wheel is within a specified range and penalize deviations."""
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]

    root_pos_z = root_asset.data.root_pos_w[:, 2]
    # Get the z positions of all the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2]

    # Calculate the height differences for all wheels
    height_diffs = root_pos_z.unsqueeze(1) - wheel_pos_z

    # Find the maximum height difference for each instance (both positive and negative)
    max_height_diff, _ = torch.max(height_diffs, dim=1)
    min_height_diff, _ = torch.min(height_diffs, dim=1)

    # Choose the larger absolute value between max and min height differences
    furthest_height_diff = torch.where(
        torch.abs(max_height_diff) > torch.abs(min_height_diff), max_height_diff, min_height_diff
    )

    # Check if the furthest height difference is within the specified range
    in_range = (furthest_height_diff >= min_height) & (furthest_height_diff <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(
        furthest_height_diff - torch.where(furthest_height_diff < min_height, max_height, min_height)
    )

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(furthest_height_diff), -out_of_range_penalty)

    return reward


def joint_target_deviation_range_l1(
    env: ManagerBasedRLEnv,
    min_angle: float,
    max_angle: float,
    in_range_reward: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the joint angle is within a specified range and penalize deviations."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the current joint positions
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Check if the joint angles are within the specified range
    in_range = (current_joint_pos >= min_angle) & (current_joint_pos <= max_angle)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.abs(
        current_joint_pos - torch.where(current_joint_pos < min_angle, min_angle, max_angle)
    )

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty)

    # Sum the rewards over all joint ids
    return torch.sum(reward, dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)
