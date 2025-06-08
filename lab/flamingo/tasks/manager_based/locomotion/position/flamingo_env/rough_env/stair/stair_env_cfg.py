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
# TODO : position_rough_env_cfg 수정
from lab.flamingo.tasks.manager_based.locomotion.position.flamingo_env.rough_env.position_rough_env_cfg import (
    LocomotionPositionRoughEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_rev01_5_1 import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):

    # curriculum_dof_torques = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "dof_torques_l2", "weight": -2.5e-3, "num_steps": 50000}
    # )
    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            "mod_range": {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-3.14, 3.14)},
            "num_steps": 25000,
        },
    )
    # TODO : modify max_vel and command range !

@configclass
class FlamingoRewardsCfg():
    
    """
    Track Reward Set
    """
    # # pos command 에 가까울수록 보상.
    # reward_track_pos_xy_exp = RewTerm(
    #     func=mdp.track_pos_xy_exp, weight=3.0, params={
    #                                                 "command_name": "pose_command",
    #                                                 "temperature": 1.0}
    # )
    # reward_track_pos_xy_exp_fine_grained = RewTerm(
    #     func=mdp.track_pos_xy_exp, weight=1.0, params={
    #                                                 "command_name": "pose_command",
    #                                                 "temperature": 2.0}
    # )
    
    # reward_track_pos_xy_exp_fine_grained = RewTerm(
    #     func=mdp.track_pos_xy_exp, weight=1.5, params={
    #                                                 "command_name": "pose_command",
    #                                                 "temperature": 2.0}
    # )

    position_tracking = RewTerm(
        func=mdp.track_pos_xyz_exp,
        weight=5.0,
        params={"temperature": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.track_pos_xyz_exp,
        weight=2.5,
        params={"temperature": 1.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained2 = RewTerm(
        func=mdp.track_pos_xyz_exp,
        weight=1.25,
        params={"temperature": 0.5, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

    # reward_x_move = RewTerm(
    #     func=mdp.reward_x_axis_move,
    #     weight = 0.5,
    #     params = {"command_name": "pose_command",
    #         "temperature" : 4.0}
    # )

    reward_align_target = RewTerm(
        func=mdp.face_target_alignment,
        weight = 0.5,
        params = {"command_name": "pose_command",}
    )
    
    # # z-각속도가 낮으면 리워드.
    # reward_smoothing_ang_vel_z_exp = RewTerm(
    #     func=mdp.reward_smoothing_ang_vel_z_exp, 
    #     weight=0.1, 
    #     params={
    #         "command_name" : "pose_command",
    #         "temperature": 4,
    #             "k" : 1.0}
    # )

    # # 목표 위치와의 거리에 따라서 desired 속도 조절.
    # reward_smoothing_lin_vel_forward_x_exp = RewTerm(
    #     func=mdp.reward_smoothing_lin_vel_forward_x_exp, 
    #     weight=0.5, # 1.5
    #     params={"command_name": "pose_command",
    #             "max_vel" : 0.75,            # maybe 0.75 is fast...
    #             "tau" : 0.5,
    #             "temperature" : 4.0})

    """
    Base Reward Set
    """

    # More exploration
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    #lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

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
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_leg_link"]),
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
        weight=-0.5,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-40.0,
        params={
            "target_height": 0.42,  # 0.37073
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_euler_angle_l2, weight=-10.0)

    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"]),})
    wheel_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6, params={"asset_cfg" : SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01

    
@configclass
class FlamingoRoughEnvCfg(LocomotionPositionRoughEnvCfg):

    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        #! ****************** Observations setup ******************* !#
        # Using Lift_mask
        # self.observations.none_stack_policy.base_pos_z = None
        # self.observations.none_stack_policy.current_reward = None
        # self.observations.none_stack_policy.is_contact = None
        #self.observations.none_stack_critic.lift_mask = None

        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (1.0, 2.0), "y": (0.0, 0.0), "z": (1.0, 2.0)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
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
        
        self.commands.pose_command.resampling_time_range=(10.0, 20.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            # ".*_leg_link",
        ]


@configclass
class FlamingoRoughEnvCfg_PLAY(FlamingoRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 0.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.sim.render_interval = self.decimation
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = False

        #! ****************** Observations setup ******************* !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # events
        self.events.push_robot = None

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        self.commands.pose_command.resampling_time_range=(10.0,10.0)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]
