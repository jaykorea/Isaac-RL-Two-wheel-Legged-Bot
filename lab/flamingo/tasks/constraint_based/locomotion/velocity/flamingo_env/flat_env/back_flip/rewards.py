# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

"""

    Backflip reward

"""

def base_target_range_height_v1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg, 
    min_target_height=0.25, 
    max_target_height=0.3, 
    minimum_height=0.1, 
    sharpness=2.0):
    # Ensure reward is zero when current height is at or below the minimum height
    
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:,2]
    #print(current_height)

    reward = torch.where(min_target_height <= current_height , torch.ones_like(current_height), 1 * (current_height - minimum_height) / (min_target_height - minimum_height))
    
    reward = torch.where(current_height <= max_target_height, reward, 1 * (1.0 - (current_height - max_target_height) / (min_target_height - minimum_height)))

    return (reward.clamp(min=0.0) ** sharpness) 


def penalty_over_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_height :float = 0.7
    ):

    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:,2]
    penalty = torch.where(current_height > max_height, torch.ones_like(current_height), torch.zeros_like(current_height))
    
    return penalty

def reward_ang_vel_y(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    
    backflip_command = env.command_manager.get_command("backflip_commands") 
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = backflip_command[:,1]
    y_ang_vel = -asset.data.root_link_ang_vel_b[:,1].clamp(max=7.2, min=-7.2)
    #print(backflip_command[:,0])
    
    return y_ang_vel * backflip_command[:,0] * torch.logical_and(current_time >= 0.2, current_time <= 0.8) #torch.logical_and(current_time > start_time , current_time < start_time + 0.5)

def penalty_ang_vel_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_ang_vel_b[:,0])

def penalty_ang_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    
    asset: RigidObject = env.scene[asset_cfg.name]

    return torch.square(asset.data.root_link_ang_vel_b[:,2])

def reward_linear_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    
    backflip_command = env.command_manager.get_command("backflip_commands") 
    current_time = backflip_command[:,1]
    asset: RigidObject = env.scene[asset_cfg.name]

    lin_vel = asset.data.root_link_lin_vel_w[:, 2].clamp(max=3,min=0)
    
    return lin_vel * backflip_command[:,0] * torch.logical_and(current_time >= 0.3, current_time <= 0.5)


class Reward_complete_backflip(ManagerTermBase):
    def __init__(self, cfg : RewardTermCfg, env : ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        self.base_ids = self.asset.find_bodies("base_link")[0]
        self.foot_ids = self.asset.find_bodies(".*wheel_link")[0]
        self.backflip_flag = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env : ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        backflip_command = env.command_manager.get_command("backflip_commands")
        backflip_time = backflip_command[:,1]
        foot_pos_z = self.asset.data.body_pos_w[:, self.foot_ids, 2].squeeze(0)
        base_pos_z = self.asset.data.body_pos_w[:, self.base_ids, 2].squeeze(0)
        time_condition = torch.logical_and(backflip_time > 0.3, backflip_time < 0.8)
        foot_condition = torch.logical_and(foot_pos_z[:,0] > base_pos_z.squeeze(1),foot_pos_z[:,1] > base_pos_z.squeeze(1))
        base_condition = base_pos_z.squeeze(1) > 0.25
        mask = time_condition
        # time_condition 을 만족하는 agent가 foot_condtion과 base_condition 을 동시에 만족했을 때 flag =1
        # 즉 특정 시간에 발의 높이가 몸의 높이보다 높은 적이 있는가 체크
        self.backflip_flag[mask] = torch.where(foot_condition[mask] & base_condition[mask],
        torch.ones_like(self.backflip_flag[mask]),
        self.backflip_flag[mask]) # 조건을 만족하지 않으면 0을 대입
        # corner case 방지를 위해 backflip_time <= 0.1 이하일 때 backflip_command 초기화.
        reset_mask = backflip_time <= 0.1 #(backflip_command[:,0] == 0.0)
        self.backflip_flag[reset_mask] = torch.zeros_like(self.backflip_flag[reset_mask])
        # baflio_command가 1이고, backflip_time동안에 agent가 살아 있다면 land한 것으로
        reward = torch.where(torch.logical_and(self.backflip_flag, backflip_time >=0.98),#env.episode_length_buf >= env.max_episode_length-10),
        torch.ones_like(self.backflip_flag),
        torch.zeros_like(self.backflip_flag))

        return reward * backflip_command[:,0]

def penalty_base_height_before_backflip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
    backflip_command = env.command_manager.get_command("backflip_commands") 
    
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = backflip_command[:,1]

    base_pos_z = asset.data.root_link_pos_w[:, 2]

    reward = torch.where(torch.logical_and(base_pos_z < 0.225, base_pos_z > 0.19972),
                        torch.zeros_like(base_pos_z),
                        torch.ones_like(base_pos_z))
    
    return  reward * backflip_command[:,0] * torch.logical_and(current_time >=0.0, current_time <= 0.2) #torch.logical_and(current_time > start_time +0.1 , current_time < start_time + 0.3) #* foot_vel_x.mean(dim=1)#* torch.exp(-error.sum(dim=1))


def reward_feet_height_during_backflip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
    
    backflip_command = env.command_manager.get_command("backflip_commands") 
    
    base_ids = [0]
    foot_ids = [7,9]
    
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = backflip_command[:,1]
    
    
    left_foot_pos_z = asset.data.body_pos_w[:, foot_ids[0], 2]
    right_foot_pos_z = asset.data.body_pos_w[:, foot_ids[1], 2]

    base_pos_z = asset.data.root_link_pos_w[:, 2]
    
    # error = torch.abs(foot_pos.mean(dim=1)- base_pos.squeeze(1))

    reward = torch.where(torch.logical_and(left_foot_pos_z > base_pos_z, right_foot_pos_z > base_pos_z),
                        torch.ones_like(left_foot_pos_z),
                        torch.zeros_like(left_foot_pos_z))
    
    return  reward * backflip_command[:,0] * torch.logical_and(current_time > 0.8, current_time < 1.2) #torch.logical_and(current_time > start_time +0.1 , current_time < start_time + 0.3) #* foot_vel_x.mean(dim=1)#* torch.exp(-error.sum(dim=1))

def penalty_xy_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    
    asset: RigidObject = env.scene[asset_cfg.name]
    backflip_command = env.command_manager.get_command("backflip_commands") 

    # Calculate linear and angular velocity errors
    lin_vel_error = torch.sum(asset.data.root_link_lin_vel_b[:,:2], dim=1)

    # return torch.exp(-lin_vel_error / 4)
    penalty = torch.square(lin_vel_error)
    
    return penalty * (1-backflip_command[:,0])

def penalty_xy_lin_vel_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    
    asset: RigidObject = env.scene[asset_cfg.name]
    backflip_command = env.command_manager.get_command("backflip_commands") 

    # Calculate linear and angular velocity errors
    lin_vel_error = torch.norm(asset.data.root_link_lin_vel_w[:,:2], dim=1)

    # return torch.exp(-lin_vel_error / 4)
    penalty = torch.square(lin_vel_error)
    
    return penalty * (1-backflip_command[:,0])

def reward_push_ground(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")):


    backflip_command = env.command_manager.get_command("backflip_commands")
    backflip_time = backflip_command[:,1]

    current_time = env.episode_length_buf * env.physics_dt * 4

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    left_foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[0]]
    #left_foot_force_history = contact_sensor.data.net_forces_w_history[:,:, wheel_link_id[0]]
    right_foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[1]]
    #right_foot_force_history = contact_sensor.data.net_forces_w_history[:,:, wheel_link_id[1]]
    
    foot_force_error = torch.abs(left_foot_force[:,2]-right_foot_force[:,2])
    
    push_force = (left_foot_force[:,2] + right_foot_force[:,2]).clamp(max=300)
    
    reward = (push_force) * torch.exp(-foot_force_error/50) 
    # mask = torch.logical_and(current_time > start_time + 0.1, current_time < start_time + 0.3)
    # if mask.any():  # 조건을 만족하는 값이 존재하는지 확인
    #     print(left_foot_force[:,2] + right_foot_force[:,2])
    
    return reward * backflip_command[:,0]

def joint_deviation_l1_backfilp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    backflip_command = env.command_manager.get_command("backflip_commands") 
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.sum(torch.abs(angle), dim=1) * (1 - backflip_command[:,0])

def flat_orientation_l2_backflip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    backflip_command = env.command_manager.get_command("backflip_commands")
    
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi

    rp = torch.stack((roll, pitch), dim=-1)

    penalty = torch.sum(torch.square(rp), dim=1)

    return penalty * (1-backflip_command[:,0])

def reward_action_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

    backflip_command = env.command_manager.get_command("backflip_commands") 
    # hip
    actions_diff = torch.square(env.action_manager.action[:,0] - env.action_manager.action[:,1])
    # shoulder
    actions_diff += torch.square(env.action_manager.action[:,2] - env.action_manager.action[:,3])
    # leg
    actions_diff += torch.square(env.action_manager.action[:,4] - env.action_manager.action[:,5])
    # wheel
    actions_diff += torch.square(env.action_manager.action[:,6] - env.action_manager.action[:,7])

    return actions_diff * (backflip_command[:,0])

def reward_gravity_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    backflip_command = env.command_manager.get_command("backflip_commands") 
    asset: RigidObject = env.scene[asset_cfg.name]
    #print( torch.square(asset.data.projected_gravity_b[:, 1]))
    return torch.square(asset.data.projected_gravity_b[:, 1]) * (backflip_command[:,0]) 

def base_flat_exp_v2(env: ManagerBasedRLEnv, temperature, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return torch.exp(-temperature * torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1))

def force_action_zero_backflip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    backflip_command = env.command_manager.get_command("backflip_commands")

    force_action_zero = torch.abs(env.action_manager.action[:, asset_cfg.joint_ids])

    return torch.sum(force_action_zero, dim=1)

"""
    skating

"""

def reward_period_push_ground_L(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    T : float = 1.0):

    # left : 4, right : 8
    wheel_link_id = [4, 8]
    current_time = env.episode_length_buf * env.physics_dt * 4

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    left_foot_force = contact_sensor.data.net_forces_w[:, wheel_link_id[0]]
    right_foot_force = contact_sensor.data.net_forces_w[:, wheel_link_id[1]]
    
    
    base_vel = asset.data.body_link_lin_vel_w[:, 0]
    left_foot_vel = asset.data.body_link_lin_vel_w[:,wheel_link_id[0]]
    right_foot_vel = asset.data.body_link_lin_vel_w[:,wheel_link_id[1]]
    
    
    
    reward_L = left_foot_force[:,2].clamp(max=10)  *  (base_vel[:,0]-left_foot_vel[:,0]).clamp(max=3)
   # print(base_vel[:,0]-left_foot_vel[:,0])
    reward_R = right_foot_force[:,2].clamp(max=10) *  (-left_foot_vel[:,0])
    # env.max_episode_length_s/T
    n = 1
    for i in range(1,10,int(T)):
        n = torch.where(torch.logical_and(current_time < 2*T*i, current_time >= 2*T*(i-1)), i, n)
    # left
    period_L = torch.logical_and(current_time >= T * (2*n-1), current_time < T * (2*n-1) + 0.1)

    # print(-left_foot_vel[:,0]*period_L)

    return (reward_L) * period_L  

def reward_period_push_ground_R(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    T : float = 1.0):

    # left : 4, right : 8
    wheel_link_id = [4, 8]
    current_time = env.episode_length_buf * env.physics_dt * 4

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    lin_vel_x = asset.data.root_lin_vel_w[:, 0]
    
    right_foot_force = contact_sensor.data.net_forces_w[:, wheel_link_id[1]]
    left_foot_force = contact_sensor.data.net_forces_w[:, wheel_link_id[0]]
    
    reward_R = right_foot_force[:,0] * 1000000
    reward_L = left_foot_force[:,0]  * 1000000 
    
    # env.max_episode_length_s/T
    n = 2
    for i in range(2,10,int(T)):
        n = torch.where(torch.logical_and(current_time < 2*T*i, current_time >= 2*T*(i-1)), i, n)
    
    period_R = torch.logical_and(current_time >= T * 2*(n-1), current_time < T*2*(n-1) + 0.1)

    return (reward_R.clamp(max=10.0)-reward_L.clamp(max=10.0)) * period_R * lin_vel_x

def reward_skating_motion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_target : float = 0.01,
    max_target : float = 0.2,
    minimum_target : float = 0.0,
    cmd_threshold : float = 0.2,
    T : float = 1.0):

    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    asset = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.physics_dt * 4
    
    wheel_link_ids = asset.find_bodies(".*wheel_link")[0]
    # foot pos x
    current_left_foot_pos = asset.data.body_pos_w[:, wheel_link_ids[0], 0] 
    current_right_foot_pos = asset.data.body_pos_w[:, wheel_link_ids[1], 0] 
    
    # target x
    reward = torch.where(min_target <= current_left_foot_pos , torch.ones_like(current_left_foot_pos), 1 * (current_left_foot_pos - minimum_target) / (min_target - minimum_target))
    
    reward = torch.where(current_left_foot_pos <= max_target, reward, 1 * (1.0 - (current_left_foot_pos - max_target) / (min_target - minimum_target)))
    
    return reward * torch.sin(2 * torch.pi / T * current_time) * (cmd > cmd_threshold)
    

"""
    one leg balancing
"""

def reward_one_leg_balancing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")):
    
    wheel_link_id = [4, 8]
#    current_time = env.episode_length_buf * env.physics_dt * 4

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    contact_time = contact_sensor.data.current_contact_time[:, wheel_link_id[0]]
    
    penalty = torch.where(contact_time > 0, torch.ones_like(contact_time), torch.zeros_like(contact_time))
    
    return penalty

def base_target_range_height_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg, 
    min_target_height=0.25, 
    max_target_height=0.3, 
    minimum_height=0.1, 
    sharpness=2.0,):
    # Ensure reward is zero when current height is at or below the minimum height
    
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:,2]
    backflip_command = env.command_manager.get_command("backflip_commands") 
    #print(current_height)
    reward = torch.where(min_target_height <= current_height , torch.ones_like(current_height), 1 * (current_height - minimum_height) / (min_target_height - minimum_height))
    
    reward = torch.where(current_height <= max_target_height, reward, 1 * (1.0 - (current_height - max_target_height) / (min_target_height - minimum_height)))

    return (reward.clamp(min=0.0) ** sharpness) * (1-backflip_command[:,0])

def base_height_adaptive_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
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

    backflip_command = env.command_manager.get_command("backflip_commands")

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height) * (1 - backflip_command[:,0])