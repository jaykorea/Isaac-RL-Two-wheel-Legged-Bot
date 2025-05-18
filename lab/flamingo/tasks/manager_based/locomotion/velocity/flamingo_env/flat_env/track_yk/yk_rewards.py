
from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from lab.flamingo.tasks.manager_based.locomotion.velocity.sensors import LiftMask
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def track_lin_vel_xyz_link_event(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    event_command_name: str,
    lin_vel_z_target: float,
    active_time_range: tuple[float, float] = (0.4, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands.
    
    - If event_cmd == 0: track XY velocity commands
    - If event_cmd == 1: track Z velocity of 5.0 m/s
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = env.command_manager.get_command(event_command_name)  # shape: (num_envs, 2)

    # --- Get actual velocity ---
    lin_vel_b = asset.data.root_link_lin_vel_b  # shape: (num_envs, 3)

    # --- Masks ---
    event_active= event_cmd[:, 0] == 1.0

    # --- Event active ---
    time_elapsed = event_cmd[:, 1]
    in_event_window = (time_elapsed >= active_time_range[0]) & (time_elapsed <= active_time_range[1])
    use_event_target = event_active & in_event_window

    # --- Default target: xy velocity tracking ---
    cmd_vel = env.command_manager.get_command(command_name)  # shape: (num_envs, 3)
    lin_vel_error_xy = torch.sum(torch.square(cmd_vel[:, :2] - lin_vel_b[:, :2]), dim=1)  # shape: (num_envs,)
    reward_xy = torch.exp(-lin_vel_error_xy / std**2)

    # --- Event mode: track z velocity to 5.0 m/s ---
    lin_vel_error_z = torch.square(lin_vel_b[:, 2] - lin_vel_z_target)
    reward_z = torch.exp(-lin_vel_error_z / std**2)

    # --- Combine based on event flag ---
    reward = torch.where(use_event_target, reward_z, reward_xy)

    return reward

def track_ang_vel_z_link_exp_event(
    env: ManagerBasedRLEnv, std: float, command_name: str, event_command_name:str , asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = env.command_manager.get_command(event_command_name)  # shape: (num_envs, 2)

    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2) * (event_cmd[:, 0]!=1.0) 

def track_ang_vel_z_link_event(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    event_command_name: str,
    active_time_range: tuple[float, float] = (0.5, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel.
    
    - If event_cmd == 0: track commanded yaw angular velocity (command_name[:, 2])
    - If event_cmd == 1: maximize a yaw velocity (ang_vel_z)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = env.command_manager.get_command(event_command_name)  # shape: (num_envs, 2)

    # --- Get actual yaw angular velocity ---
    ang_vel_b_z = asset.data.root_link_ang_vel_b[:, 2]  # shape: (num_envs,)

    # --- Masks ---
    event_active= event_cmd[:, 0] == 1.0

    # --- Event active ---
    time_elapsed = event_cmd[:, 1]
    in_event_window = (time_elapsed >= active_time_range[0]) & (time_elapsed <= active_time_range[1])
    use_event_target = event_active & in_event_window

    # --- Default: track commanded yaw velocity (from command_name) ---
    cmd_ang_vel_z = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error_default = torch.square(cmd_ang_vel_z - ang_vel_b_z)
    reward_default = torch.exp(-ang_vel_error_default / std**2)

    # --- Event mode: maximize yaw velocity ---
    reward_event = torch.abs(ang_vel_b_z)

    # --- Combine based on event flag ---
    reward = torch.where(use_event_target, reward_event, reward_default)

    return reward

def ang_vel_z_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.15, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for yaw angular velocity using linear scaled abs value (clipped at 15 rad/s).

    r = clamp(|ang_vel_z| / 4.0, max=15 / 4.0)
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]
    ang_vel_b_z = asset.data.root_link_ang_vel_b[:, 2]

    reward = torch.clamp(torch.abs(ang_vel_b_z) / 3.0, max=5.0)

    return reward * event_command[:, 0] * torch.logical_and(event_time >= event_time_range[0], event_time <= event_time_range[1])

def lin_vel_z_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    event_command = env.command_manager.get_command(event_command_name)
    event_time    = event_command[:, 1]
    asset: RigidObject = env.scene[asset_cfg.name]

    lin_vel = asset.data.root_lin_vel_w
    lin_vel_z = lin_vel[:, 2]
    lin_vel_mag = torch.norm(lin_vel, dim=1) + 1e-6

    z_axis = torch.tensor([0, 0, 1.0], device=lin_vel.device)
    alignment = torch.sum(lin_vel * z_axis, dim=1) / lin_vel_mag  # = cos(theta)

    alignment_reward = torch.abs(alignment)

    max_up_vel = 6.0
    up_vel = torch.clamp(lin_vel_z, min=0, max=max_up_vel)
    down_vel = torch.clamp(-lin_vel_z, min=-max_up_vel, max=max_up_vel)

    pre_jump = (event_time < event_time_range[0]).float()

    descent_vel = torch.clamp(-lin_vel_z, min=0.0)
    max_descent_vel = 0.75

    penalty_coef = 1.0
    descent_penalty = torch.clamp(descent_vel - max_descent_vel, min=0.0) * penalty_coef

    jump_phase = torch.logical_and(event_time >= event_time_range[0], event_time <= event_time_range[1]).float()

    after_jump = torch.logical_and(event_time > event_time_range[1], event_time <= 1.2).float()

    reward = up_vel * 0.8 * event_command[:, 0] * jump_phase * alignment_reward
    reward += down_vel * 0.2 * event_command[:, 0] * after_jump * alignment_reward
    reward -= descent_penalty * event_command[:, 0]  * pre_jump

    return reward

def reward_push_ground_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:

    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]

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

    return reward * event_command[:, 0] * torch.logical_and(event_time >= event_time_range[0], event_time <= event_time_range[1])

def feet_air_time_event(
    env: ManagerBasedRLEnv,
    event_command_name: str,
    threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
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
    event_command = env.command_manager.get_command(event_command_name) 
    
    return (reward * event_command[:,0])
    
def feet_air_time_target_event(
    env: ManagerBasedRLEnv,
    event_command_name: str,
    target_air_time: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward the robot for matching foot air time to a target value.

    This reward encourages the robot to lift its feet off the ground for a specified duration (`target_air_time`),
    penalizing deviations using squared error (L2 loss).
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # shape: (N, num_feet)

    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]

    air_time_error = last_air_time - target_air_time
    reward = torch.sum(torch.square(air_time_error), dim=1)  # shape: (N,)

    return reward * event_command[:, 0] * torch.logical_and(event_time >= 0.3, event_time <= 0.8)

class RewardCompleteEvent(ManagerTermBase):

    def __init__(self,
                 cfg: RewardTermCfg,
                 env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]

        self.base_id = self.asset.find_bodies("base_link")[0]
        self.foot_ids = self.asset.find_bodies(".*wheel_static_link")[0]
        self.contact_foot_ids = self.asset.find_bodies(".*wheel_link")[0]

        self.jump_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.jump_land_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Thresholds
        self.BASE_THRESH = 0.55
        self.FOOT_THRESH = 0.25
        self.LANDING_TIME = 0.8

    def __call__(self,
                 env: ManagerBasedRLEnv,
                 event_command_name: str,
                 asset_cfg: SceneEntityCfg,
                 sensor_cfg: SceneEntityCfg):

        # === 1. Command and time ===
        command = env.command_manager.get_command(event_command_name)
        command_flag = command[:, 0]      # 0 or 1.0
        event_time = command[:, 1]

        # === 2. Observation ===
        pos = self.asset.data.body_pos_w
        base_z = pos[:, self.base_id, 2].squeeze(-1)
        foot_z = pos[:, self.foot_ids, 2]     # shape: (N, 2)

        # === 3. Sensor contact ===
        contact_time = self.contact_sensor.data.current_contact_time[:, self.contact_foot_ids]
        in_contact = contact_time > 0
        wheels_contact = in_contact.all(dim=1)

        # === 4. Boolean masks ===
        jump_window = (event_time > 0.5) & (event_time < 1.0)
        jumped_high_enough = (base_z > self.BASE_THRESH) & \
                             (foot_z[:, 0] > self.FOOT_THRESH) & \
                             (foot_z[:, 1] > self.FOOT_THRESH)
        jump_success = jump_window & jumped_high_enough

        landing_success = (event_time >= self.LANDING_TIME) & wheels_contact
        command_active = command_flag > 0.5
        is_dead = env.reset_buf > 0

        # === 5. Flag updates ===
        self.jump_flag |= jump_success
        self.jump_land_flag |= self.jump_flag & landing_success

        # === 6. Reward computation ===
        reward = torch.zeros_like(event_time)

        jump_success_mask = self.jump_flag & ~is_dead & command_active
        jump_land_success_mask = self.jump_land_flag & ~is_dead & command_active
        # death_mask       = self.jump_land_flag &  is_dead & command_active
        # total_fail_mask  = ~self.jump_land_flag & command_active

        reward[jump_success_mask]    =  1.0
        reward[jump_land_success_mask] = 1.0

        # === 7. Reset ===
        reset_mask = is_dead | (event_time < 0.02)
        self.jump_flag[reset_mask] = False
        self.jump_land_flag[reset_mask] = False

        return reward

def base_height_adaptive_l2_event(
    env: ManagerBasedRLEnv,
    target_height: float,
    event_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = env.command_manager.get_command(event_command_name)  # shape: (num_envs, 2)

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height) * (1 - event_cmd[:,0])

def over_height(
    env: ManagerBasedRLEnv,
    event_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_height :float = 0.65):

    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:,2]
    penalty = torch.where(current_height > max_height, torch.ones_like(current_height), torch.zeros_like(current_height))
    event_command = env.command_manager.get_command(event_command_name)

    return penalty * event_command[:,0]