# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from lab.flamingo.isaaclab.isaaclab.envs.manager_based_constraint_rl_env import ManagerBasedConstraintRLEnv as ManagerBasedRLEnv



def base_lin_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b

def base_lin_vel_x_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 0].unsqueeze(-1)

def base_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b

        
def base_pos_z_rel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
    else:
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
    
def base_pos_z_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
    else:
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def current_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current reward value. Returns zeros if the reward manager is not initialized."""
    if not hasattr(env, "reward_manager") or env.reward_manager is None:
        # Assuming the shape should be (num_envs,) based on the environment
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    try:
        return env.reward_buf.unsqueeze(-1)
    except AttributeError:
        # Fallback to zeros if the reward_manager is initialized but compute isn't ready
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def is_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()


def lift_mask_by_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    
    """
    Generate a lift mask for the robot's legs based on row-wise height scan gradients from separate left and right sensors.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        sensor_cfg_left (SceneEntityCfg): Configuration for the left raycast sensor.
        sensor_cfg_right (SceneEntityCfg): Configuration for the right raycast sensor.
        command_name (str): Command name to check movement intention.
        gradient_threshold (float): Threshold for row-wise height gradient to detect steps.

    Returns:
        torch.Tensor: Lift mask for left and right legs. Shape: [num_envs, 2].
    """
    #* Step 1: Extract ray hit positions (Z coordinates) from left and right sensors
    left_lift_mask_sensor = env.scene.sensors[sensor_cfg_left.name]
    right_lift_mask_sensor = env.scene.sensors[sensor_cfg_right.name]

    left_mask= left_lift_mask_sensor.data.mask 
    right_mask = right_lift_mask_sensor.data.mask  
    
    lift_mask = torch.stack([left_mask, right_mask], dim=1) 

    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)  # Shape: [num_envs]
    lift_mask *= (command_norm > 0.1).unsqueeze(-1).float()  # Apply movement condition

    return lift_mask

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def base_euler_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_com_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy

def base_euler_angle_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy


def joint_pos_rel_sin(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as sine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_sin = torch.sin(current_value)
    return current_value_sin


def joint_pos_rel_cos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as cosine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_cos = torch.cos(current_value)
    return current_value_cos


def height_scan_raw(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.ray_hits_w[..., 2]



def generated_partial_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, 0]