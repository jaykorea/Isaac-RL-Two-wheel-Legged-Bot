# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import lab.flamingo.tasks.manager_based.locomotion.position.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.position.flamingo_env.flat_env.position_env_cfg import (
    LocomotionPositionFlatEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_rev01_5_1 import FLAMINGO_CFG  # isort: skip
from lab.flamingo.tasks.manager_based.locomotion.position.terrain_config.rough_config import ROUGH_TERRAINS_CFG

# @configclass
# class FlamingoCurriculumCfg(CurriculumCfg):



@configclass
class FlamingoRewardsCfg():
    # -- task
    
    reward_track_pos_xy_exp = RewTerm(
        func=mdp.track_pos_xy_exp, weight=2.5, params={"command_name": "pose_command",
                                                    "temperature": 1}
    )
    
    # # z-각속도가 낮으면 리워드.
    # reward_smoothing_ang_vel_z_exp = RewTerm(
    #     func=mdp.reward_smoothing_ang_vel_z_exp, 
    #     weight=0.5, 
    #     params={"temperature": 4,
    #             "k" : 1.0}
    # )

    # 목표 위치와의 거리에 따라서 desired 속도 조절.
    reward_smoothing_lin_vel_xy_exp = RewTerm(
        func=mdp.reward_smoothing_lin_vel_xy_exp, 
        weight=0.5, 
        params={"command_name": "pose_command",
                "max_distance" : 6.0,
                "max_vel" : 0.75,
                "tau" : 0.5,
                "temperature" : 4.0})
    
    # 목표 위치와의 거리 조절.
    # position_tracking = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=2.0,
    #     params={"std": 2.0, "command_name": "pose_command"},
    # )
    # position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=2.0,
    #     params={"std": 0.2, "command_name": "pose_command"},
    # )
    
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight= 0.5,
        params={"command_name": "pose_command",
                "temperature" :  4.0,
                "k" : 1},
    )
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_zero_l1,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    # )
    # joint_deviation_shoulder = RewTerm(
    #     func=mdp.joint_deviation_zero_l1,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    # )

    dof_pos_limits_hip = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    dof_pos_limits_leg = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_hip_link"]),
            "threshold": 1.0,
        },
    )
    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.2,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    leg_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.1,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-25.0,
        params={
            "target_height": 0.36288,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01



@configclass
class FlamingoFlatEnvCfg(LocomotionPositionFlatEnvCfg):

    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


        #! ****************** Observations setup ****************** !#
        self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None
        self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        # self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.base_pos_z = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        # self.observations.none_stack_policy.lift_mask = None
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-3.5, 3.5), "y": (-3.5, 3.5), "yaw": (-3.14, 3.14)},
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
        self.commands.pose_command.ranges.pos_x = (-6.0, 6.0)
        self.commands.pose_command.ranges.pos_y = (-6.0, 6.0)

        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]



@configclass
class FlamingoFlatEnvCfg_PLAY(FlamingoFlatEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # observations
        #! ****************** Observations setup ******************* !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        self.events.push_robot.interval_range_s = (5.5, 6.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
        }

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.curriculum.terrain_levels = None
        
        # commands
        self.commands.pose_command.ranges.pos_x = (-6.0, 6.0)
        self.commands.pose_command.ranges.pos_y = (-6.0, 6.0)
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]
