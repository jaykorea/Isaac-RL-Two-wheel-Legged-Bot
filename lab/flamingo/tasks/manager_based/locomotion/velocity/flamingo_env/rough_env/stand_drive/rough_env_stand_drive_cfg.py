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
from lab.flamingo.assets.flamingo import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # curriculum_stuck_air_time = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "stuck_air_time", "weight": 1.0, "num_steps": 10000}
    # )


@configclass
class FlamingoRewardsCfg(RewardsCfg):
    stuck_air_time = RewTerm(
        func=mdp.FlamingoAirTimeReward,
        weight=0.5,
        params={
            "stuck_threshold": 0.15,
            "stuck_duration": 10,
            "threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation_range_shoulder = RewTerm(
        func=mdp.joint_target_deviation_range_l1,
        weight=0.55,
        params={
            "min_angle": -0.261799,
            "max_angle": 0.1,
            "in_range_reward": 0.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]),
        },  # target: -0.261799
    )
    joint_deviation_range_leg = RewTerm(
        func=mdp.joint_target_deviation_range_l1,
        weight=0.55,
        params={
            "min_angle": 0.46810467,
            "max_angle": 0.66810467,
            "in_range_reward": 0.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]),
        },  # target: 0.56810467
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
        weight=-2.0,
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
        weight=-0.05,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-1.5,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    leg_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-1.5,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height_dynamic_wheel = RewTerm(
        func=mdp.base_height_range_relative_l2,
        weight=25.0,
        params={
            "min_height": 0.30182,
            "max_height": 0.30182,
            "in_range_reward": 0.0,
            "root_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "wheel_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
        },
    )


@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Cassie rough environment configuration."""

    curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()
    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.observations.policy.enable_corruption = True
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.004)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (
            0.0,
            0.005,
        )  # Very gentle slope
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = (
            0.05  # Increase proportion if you want more of this terrain
        )
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (
            0.0,
            0.005,
        )  # Very gentle slope
        # Adjust the inverted pyramid stairs terrain
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (
            0.005,
            0.015,
        )  # Smaller step height for gentler steps
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_width = (
            0.45  # Increase step width for gentler steps
        )

        # events
        self.events.push_robot = None
        # self.events.push_robot.params = {
        #     "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        # }

        # # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 2.5)

        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
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

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            # ".*_hip_link",
            # ".*_shoulder_link",
            ".*_leg_link",
        ]

        # rewards
        self.rewards.dof_torques_l2.weight = -5.0e-4  # default: -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight *= 1.0
        self.rewards.ang_vel_xy_l2.weight *= 1.0
        self.rewards.action_rate_l2.weight *= 1.0  # default: 1.5
        self.rewards.dof_acc_l2.weight *= 0.75  # default: 1.5
        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
