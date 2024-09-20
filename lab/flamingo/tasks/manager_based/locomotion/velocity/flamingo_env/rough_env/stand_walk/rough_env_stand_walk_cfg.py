# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

# import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    CommandsCfg,
    CurriculumCfg,
)

##
# Pre-defined configs
##
from lab.flamingo.assets.flamingo import FLAMINGO_WALK_CFG  # isort: skip


@configclass
class FalmingoCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityWithZCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityWithZCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.5, 1.5),
            pos_z=(0.1931942, 0.3531942),
        ),
    )


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class FlamingoRewardsCfg(RewardsCfg):
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.2,
    #     },
    # )
    gait = RewTerm(
        func=mdp.GaitReward,
        weight=4.0,
        params={
            "std": 0.02,
            "max_err": 0.75,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("left_wheel_link"), ("right_wheel_link")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "cmd_threshold": 0.0,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.203,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            # "cmd_threshold": 0.0,
        },
    )
    # joint_deviation_range_shoulder = RewTerm(
    #     func=mdp.joint_target_deviation_range_l1,
    #     weight=0.55,
    #     params={
    #         "min_angle": -0.261799,
    #         "max_angle": 0.1,
    #         "in_range_reward": 0.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]),
    #         "cmd_threshold": 0.2,
    #     },  # target: -0.261799
    # )
    # joint_deviation_range_leg = RewTerm(
    #     func=mdp.joint_target_deviation_range_l1,
    #     weight=0.55,
    #     params={
    #         "min_angle": 0.46810467,
    #         "max_angle": 0.66810467,
    #         "in_range_reward": 0.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]),
    #         "cmd_threshold": 0.2,
    #     },  # target: 0.56810467
    # )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"])
        },
    )
    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,  # default: -0.1
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"])
        },
    )
    force_action_zero = RewTerm(
        func=mdp.force_action_zero,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]),
        },
    )
    # shoulder_align_l1 = RewTerm(
    #     func=mdp.joint_align_l1,
    #     weight=-1.0,  # default: -0.5
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint"),
    #         "cmd_threshold": 0.1,
    #     },
    # )
    # leg_align_l1 = RewTerm(
    #     func=mdp.joint_align_l1,
    #     weight=-1.0,  # default: -0.5
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint"),
    #         "cmd_threshold": 0.1,
    #     },
    # )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height_dynamic_wheel = RewTerm(
        func=mdp.base_height_range_relative_l2,
        weight=15.0,
        params={
            "min_height": 0.34868,
            "max_height": 0.34868,
            "in_range_reward": 0.0,
            "root_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "wheel_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
        },
    )


@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Cassie rough environment configuration."""

    commands: FalmingoCommandsCfg = FalmingoCommandsCfg()
    curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()
    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.observations.policy.enable_corruption = True

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (
        #     0.0,
        #     0.01,
        # )  # Very gentle slope
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = (
        #     0.1  # Increase proportion if you want more of this terrain
        # )
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (
        #     0.0,
        #     0.01,
        # )  # Very gentle slope
        # # Adjust the inverted pyramid stairs terrain
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (
        #     0.01,
        #     0.075,
        # )  # Smaller step height for gentler steps
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_width = (
        #     0.2  # Increase step width for gentler steps
        # )

        # events
        self.events.push_robot = None
        # self.events.push_robot.params = {
        #     "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        # }

        # # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 2.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.0, 0.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]

        # rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5.0e-4
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.5)
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.5)
        self.rewards.lin_vel_z_l2.weight *= 1.0
        self.rewards.ang_vel_xy_l2.weight *= 1.0
        self.rewards.action_rate_l2.weight *= 1.0
        self.rewards.dof_acc_l2.weight *= 1.0
        # commands
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
