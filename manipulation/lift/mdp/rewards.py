# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms
from omni.isaac.lab.utils.math import matrix_from_quat
from omni.isaac.lab.utils.math import euler_xyz_from_quat, wrap_to_pi


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_orientation_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    orientation_weight: float = 0.25,
) -> torch.Tensor:
    """Reward the agent for aligning the object's orientation with the goal orientation."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # compute the desired orientation in the world frame
    des_ori_b = command[:, 3:7]  # Assuming the command includes desired orientation as a quaternion
    _, des_ori_w = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], None, des_ori_b)

    # Current orientation of the object
    current_ori = object.data.root_state_w[:, 3:7]  

    # Normalize quaternions to ensure they're valid
    current_ori = current_ori / torch.norm(current_ori, dim=1, keepdim=True)
    des_ori_w = des_ori_w / torch.norm(des_ori_w, dim=1, keepdim=True)

    # Compute the dot product between the current and desired orientations
    dot_product = torch.sum(current_ori * des_ori_w, dim=1)

    # Ensure the dot product is within the valid range for arccos (to avoid NaNs)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute angular distance
    orientation_distance = torch.acos(2 * dot_product**2 - 1)

    # Reward for orientation alignment
    reward_orientation = 1 - torch.tanh(orientation_weight * orientation_distance)

    return reward_orientation


def object_minimize_roll_pitch(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize non-zero roll and pitch angles."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]

    # Get the current orientation of the object as quaternions
    current_ori = object.data.root_state_w[:, 3:7]

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, _ = euler_xyz_from_quat(current_ori)

    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)

    # Compute penalties for roll and pitch (penalize deviation from 0)
    roll_penalty = roll_weight * torch.abs(roll)
    pitch_penalty = pitch_weight * torch.abs(pitch)

    # Return the combined penalty as a negative reward
    return roll_penalty + pitch_penalty


def grasp_plate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["object"].data.root_pos_w
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    distance_xy = torch.norm(handle_pos[:, :2] - ee_tcp_pos[:, :2], dim=-1, p=2)
    # distance_z = torch.norm(handle_pos[:, 2] - ee_tcp_pos[:, 2], dim=-1, p=2)
    is_close = (distance_xy > 0.0575 - 0.02) & (distance_xy < 0.0575 + 0.02)  # & (distance_z < 0.1)

    return is_close * torch.sum(0.04 - gripper_joint_pos, dim=1)
