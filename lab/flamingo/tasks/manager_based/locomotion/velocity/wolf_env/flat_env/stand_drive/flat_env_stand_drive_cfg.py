# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.wolf_env.rough_env.stand_drive.drive_rewards as mdp_drive
from lab.flamingo.tasks.manager_based.locomotion.velocity.wolf_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.wolf_rev01_0_0 import WOLF_CFG  # isort: skip


@configclass
class WolfRewardsCfg():
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Shank_front_left_link", "Shank_front_right_link", "Ankle_back_left_link", "Ankle_back_right_link"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Thigh_.*"]),
            "threshold": 1.0,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["HAA_.*"])},
    )
    
    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )

    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-25.0,
        params={
            "target_height": 0.6,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )

@configclass
class WolfFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: WolfRewardsCfg = WolfRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = WOLF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None
        
        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_policy.height_scan = None
        # self.observations.obs_info = None

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "Hip_.*",
            "Thigh_.*",
        ]

@configclass
class WolfFlatEnvCfg_PLAY(WolfFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True

        # scene
        self.scene.robot = WOLF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None
        
        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_policy.height_scan = None
        
        #! ****************** Observations setup ******************* !#
        # disable randomization for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]

        # randomize actuator gains
        self.events.randomize_joint_actuator_gains = None

        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        self.events.push_robot = None
        
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]
