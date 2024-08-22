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
)

##
# Pre-defined configs
##
from lab.flamingo.assets.flamingo import FLAMINGO_WALK_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class FlamingoRewardsCfg(RewardsCfg):
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=200.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*wheel_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
    #     },
    # )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    # )
    # joint_deviation_wheel = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    # )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"])
        },
    )
    # joint_applied_torque_limits = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=-0.015,  # default: -0.1
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"])
    #     },
    # )
    # force_action_zero = RewTerm(
    #     func=mdp.force_action_zero,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    # )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)


@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Cassie rough environment configuration."""

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
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.004)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (
        #     0.0,
        #     0.025,
        # )  # Very gentle slope
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = (
        #     0.05  # Increase proportion if you want more of this terrain
        # )
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (
        #     0.0,
        #     0.01,
        # )  # Very gentle slope

        # events
        self.events.push_robot = None
        # self.events.push_robot.params = {
        #     "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        # }

        # # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)

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
        # self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -5.0e-6
        # self.rewards.track_lin_vel_xy_exp.weight = 2.0
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.action_rate_l2.weight *= 1.0
        # self.rewards.dof_acc_l2.weight *= 1.0
        # commands
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
