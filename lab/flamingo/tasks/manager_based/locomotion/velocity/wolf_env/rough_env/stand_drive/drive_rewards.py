
from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, yaw_quat


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def reward_ang_vel_z_link_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_y = torch.abs(env.command_manager.get_command(command_name)[:, 1])
    # compute the error
    ang_vel = torch.square(asset.data.root_link_ang_vel_b[:, 2]) * lin_vel_y
    return ang_vel

def reward_feet_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_feet_distance: float = 0.4885,
    max_feet_distance: float = 0.4885,
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    # foot positions in world frame
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]  # [N,2,2]
    dist = torch.norm(foot_pos[:,0,:] - foot_pos[:,1,:], dim=-1)
    penalize_min = torch.clip(min_feet_distance - dist, 0.0, 1.0)
    penalize_max = torch.clip(dist - max_feet_distance, 0.0, 1.0)
    return penalize_min + penalize_max

def reward_nominal_foot_position_adaptive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg_left: SceneEntityCfg | None = None,
    sensor_cfg_right: SceneEntityCfg | None = None,
    command_name: str = "base_velocity",
    base_height_target: float = 0.36288,
    foot_radius: float = 0.127,
    temperature: float = 200.0,
    sigma_wrt_v: float = 0.5,
) -> torch.Tensor:
    """
    Reward foot height tracking relative to a dynamic target height per foot.
    Each foot uses its corresponding height sensor to adapt the target height in real time:
    nominal_height = foot_radius - base_height_target + delta,
    where delta is the max detected terrain step for that foot.

    Args:
        env: ManagerBasedRLEnv
        asset_cfg: SceneEntityCfg for the robot asset (contains foot body_ids)
        sensor_cfg_left: SceneEntityCfg for left foot height sensor (RayCaster)
        sensor_cfg_right: SceneEntityCfg for right foot height sensor (RayCaster)
        command_name: name of the command to retrieve velocity commands
        base_height_target: static base height target (m)
        foot_radius: radius/offset of the foot link (m)
        sigma: Gaussian width for height error
        sigma_wrt_v: Gaussian width for velocity attenuation

    Returns:
        Tensor of shape [num_envs] with reward values
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmds = env.command_manager.get_command(command_name)
    num_envs = env.num_envs
    device = env.device

    # Build per-foot dynamic target heights [num_envs, 2]
    target_height = torch.full((num_envs, len(asset_cfg.body_ids)), base_height_target, device=device)
    # Left foot
    if sensor_cfg_left is not None:
        sl: RayCaster = env.scene[sensor_cfg_left.name]
        sensor_z_l = sl.data.pos_w[:, 2]                       # [N]
        hit_z_l    = torch.max(sl.data.ray_hits_w[..., 2], dim=1).values  # [N]
        delta_l    = (sensor_z_l - hit_z_l) + 0.05             # [N], +여유마진
        target_height[:, 0] = base_height_target - delta_l

    # Right foot
    if sensor_cfg_right is not None:
        sr: RayCaster = env.scene[sensor_cfg_right.name]
        sensor_z_r = sr.data.pos_w[:, 2]
        hit_z_r    = torch.max(sr.data.ray_hits_w[..., 2], dim=1).values
        delta_r    = (sensor_z_r - hit_z_r) + 0.05
        target_height[:, 1] = base_height_target - delta_r

    # Compute nominal foot height relative to base origin [N,2]
    # nominal_height = foot_radius - (base_height_target - delta)
    nominal_height = foot_radius - target_height  # [N,2]

    # World->base translation + rotation
    base_pos  = asset.data.root_link_pos_w   # [N,3]
    base_quat = asset.data.root_link_quat_w  # [N,4]
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [N,2,3]
    foot_base  = foot_world - base_pos.unsqueeze(1)                  # [N,2,3]

    # Calculate reward per foot
    reward = torch.zeros(num_envs, device=device)
    for i in range(len(asset_cfg.body_ids)):
        fb = quat_rotate_inverse(base_quat, foot_base[:, i, :])  # [N,3]
        err = nominal_height[:, i] - fb[:, 2]                    # [N]
        reward += torch.exp(-err.square() * temperature)

    # Average across feet and apply velocity attenuation
    vel_norm = torch.norm(cmds[:, :3], dim=1)                   # [N]
    reward = (reward / len(asset_cfg.body_ids)) * torch.exp(-vel_norm.square() / sigma_wrt_v)

    return reward

def reward_nominal_foot_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    command_name: str = "base_velocity",
    base_height_target: float = 0.36288,
    foot_radius: float = 0.127,
    sigma: float = 0.005,
    sigma_wrt_v: float = 0.5
) -> torch.Tensor:
    """
    Reward foot height tracking relative to nominal base height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmds = env.command_manager.get_command(command_name)
    # base frame data
    base_pos = asset.data.root_link_pos_w  # [N,3]
    base_quat = asset.data.root_link_quat_w
    # body-frame foot positions
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [N,2,3]
    foot_base = foot_world - base_pos.unsqueeze(1)
    reward = torch.zeros(env.num_envs, device=env.device)
    nominal_height = -(base_height_target - foot_radius)
    for i in range(len(asset_cfg.body_ids)):
        fb = quat_rotate_inverse(base_quat, foot_base[:,i,:])  # [N,3]
        err = nominal_height - fb[:,2]
        reward += torch.exp(-(torch.square(err))/sigma)
    vel_norm = torch.norm(cmds[:, :3], dim=1)
    reward = (reward/len(asset_cfg.body_ids)) * torch.exp(-(vel_norm**2)/sigma_wrt_v)
    return reward


def reward_leg_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    temperature = 50.0 # 0.001,
) -> torch.Tensor:
    """
    Encourage symmetry in Y direction between two feet.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_rotate_inverse(base_quat, foot_base[:,i,:])
    err = (foot_base[:,0,1].abs() - foot_base[:,1,1].abs())
    return torch.exp(-(temperature * torch.square(err)))


def reward_same_foot_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize X-axis displacement difference of two feet in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_rotate_inverse(base_quat, foot_base[:,i,:])
    dx = foot_base[:,0,0] - foot_base[:,1,0]
    return torch.abs(dx)

def reward_same_foot_y_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Encourage symmetry in Y direction between two feet.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_rotate_inverse(base_quat, foot_base[:,i,:])
    dy = (foot_base[:,0,1].abs() - foot_base[:,1,1].abs())
    return torch.abs(dy)

def reward_same_foot_z_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize difference in Z positions of two feet in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_rotate_inverse(base_quat, foot_base[:,i,:])
    dz = foot_base[:,0,2] - foot_base[:,1,2]
    return torch.square(dz)

# def reward_action_smooth(
#     env
# ) -> torch.Tensor:
#     """
#     Penalize second order action changes.
#     """
#     a = env.action_manager.action
#     pa = env.action_manager.prev_action
#     p2 = env.action_manager.prev2_action
#     return torch.sum((a - 2*pa + p2)**2, dim=1)


def reward_keep_balance(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Constant reward for being alive.
    """
    return torch.ones(env.num_envs, device=env.device)

def foot_lin_vel_z_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    max_up_vel: float = 3.0,
    up_vel_coef: float = 10.0,
    temperature: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    left_mask  = env.scene.sensors[sensor_cfg_left.name].data.mask   # [B]
    right_mask = env.scene.sensors[sensor_cfg_right.name].data.mask  # [B]
    mask = torch.stack((left_mask, right_mask), dim=-1).float()       # [B, 2]

    foot_lin_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # [B,2,3]

#    desired_ang_vel = stiffness * target_heading
#    diff = torch.square(ang_vel_z - desired_ang_vel)


    foot_lin_vel_z   = foot_lin_vel[:, :, 2]                                # [B,2]
    foot_lin_vel_mag = torch.norm(foot_lin_vel, dim=2) + 1e-6              # [B,2]

    alignment_reward = torch.abs(foot_lin_vel_z / foot_lin_vel_mag)        # [B,2]

    up_vel_reward = torch.exp(
        -torch.abs((max_up_vel - foot_lin_vel_z) / max_up_vel) * temperature
    )                                                                        # [B,2]

    reward = torch.sum(up_vel_reward  * up_vel_coef * mask, dim=1)

    return reward

def body_lin_vel_z_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: None | SceneEntityCfg = None,
    sensor_cfg_right: None | SceneEntityCfg = None,
    max_up_vel: float = 3.0,
    up_vel_coef: float = 10.0,
    temperature: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]

    left_lift_mask_sensor = env.scene.sensors[sensor_cfg_left.name]
    right_lift_mask_sensor = env.scene.sensors[sensor_cfg_right.name]

    left_mask= left_lift_mask_sensor.data.mask 
    right_mask = right_lift_mask_sensor.data.mask

    lift_mask = torch.logical_and(left_mask, right_mask).float()  # [B, N_foot]

    lin_vel = asset.data.root_lin_vel_w
    lin_vel_z = lin_vel[:, 2]
    lin_vel_mag = torch.norm(lin_vel, dim=1) + 1e-6

    z_axis = torch.tensor([0, 0, 1.0], device=lin_vel.device)
    alignment = torch.sum(lin_vel * z_axis, dim=1) / lin_vel_mag  # = cos(theta)

    alignment_reward = torch.abs(alignment)

    target_up_vel = max_up_vel

    up_vel_reward  = torch.exp(-torch.abs((target_up_vel - lin_vel_z)/max_up_vel) * temperature)

    reward = up_vel_reward * lift_mask * up_vel_coef # * alignment_reward

    return reward

def wheel_action_zero_event(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    event_command_name: str = "event",
    wheel_action_name: str = "wheel_vel",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    cmd = env.command_manager.get_command(command_name)         # [B, D]
    event_command = env.command_manager.get_command(event_command_name)  # [B, 2]
    
    wheel_action = env.action_manager.get_term(wheel_action_name).processed_actions
    wheel_action_l2 = torch.sum(torch.square(wheel_action), dim=1)  # [B]

    no_cmd_mask = torch.norm(cmd, dim=-1) < 1e-3

    return wheel_action_l2 * no_cmd_mask * event_command[:, 0]

def reward_push_ground_terrain(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: None | SceneEntityCfg = None,
    sensor_cfg_right: None | SceneEntityCfg = None,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:

    left_lift_mask_sensor = env.scene.sensors[sensor_cfg_left.name]
    right_lift_mask_sensor = env.scene.sensors[sensor_cfg_right.name]

    left_mask= left_lift_mask_sensor.data.mask 
    right_mask = right_lift_mask_sensor.data.mask

    lift_mask = torch.logical_and(left_mask, right_mask).float()  # [B, 1]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    z_axis = torch.tensor([0, 0, 1.0], device=foot_force.device).view(1, 1, 3)  # shape [1, 1, 3]

    force_mag = torch.norm(foot_force, dim=2) + 1e-6  # [B, N_foot]
    alignment = torch.sum(foot_force * z_axis, dim=2) / force_mag  # [B, N_foot]
    alignment = torch.abs(alignment)

    aligned_force = force_mag * alignment  # [B, N_foot]

    force_diff = torch.abs(aligned_force[:, 0] - aligned_force[:, 1])  # [B]

    total_force = aligned_force.sum(dim=1).clamp(max=300)  # [B]
    reward = total_force * torch.exp(-force_diff / 20)

    return reward * lift_mask