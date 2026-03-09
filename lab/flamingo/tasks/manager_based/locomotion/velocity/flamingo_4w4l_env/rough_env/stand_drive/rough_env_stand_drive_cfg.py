# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_4w4l_env.rough_env.stand_drive.rough_rewards as mdp_rough
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_4w4l_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from lab.flamingo.assets.flamingo.flamingo_4w4l_rev02_0_4 import FLAMINGO4W4L_CFG  # isort: skip

@configclass
class Flamingo4w4lRewardsCfg():
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    #keep_alive = RewTerm(func=mdp_drive.reward_keep_balance, weight=0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    dof_pos_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_joint).*")})
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    action_smoothness_l2 = RewTerm(func=mdp_rough.ActionSmoothness, weight=-0.1)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.5,  #-0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.25,  #-0.25
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    joint_deviation_leg= RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.1, #-0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"])},
    )
    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
        }
    )

@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: Flamingo4w4lRewardsCfg = Flamingo4w4lRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO4W4L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")#UNITREE_B2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ] 
        
@configclass
class FlamingoRoughEnvCfg_PLAY(FlamingoRoughEnvCfg):
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
            #self.scene.terrain.terrain_generator.terrain_generator = ROUGH_TERRAINS_PLAY_CFG
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # scene
        self.scene.robot = FLAMINGO4W4L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator

        #! ****************** Observations setup ******************* !#
        #! ********************************************************* !#

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]

        # randomize actuator gains
        #self.events.randomize_joint_actuator_gains = None

        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot = None

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        self.lin_vel_cmd_levels = None

        # commands
        # self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ] 
        # self.terminations.base_contact=None
        # self.terminations.bad_orientation = None