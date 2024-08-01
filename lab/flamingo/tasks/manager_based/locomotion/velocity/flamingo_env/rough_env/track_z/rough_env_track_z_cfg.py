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

from lab.flamingo.assets.flamingo import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    curriculum_stuck_air_time = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "stuck_air_time", "weight": 1.0, "num_steps": 10000}
    )


@configclass
class FlamingoRewardsCfg(RewardsCfg):
    track_pos_z_l2 = RewTerm(
        func=mdp.track_pos_z_exp,
        weight=1.0,
        params={
            "std": math.sqrt(0.025),
            "command_name": "base_velocity",
            "relative": True,
            "root_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "wheel_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )
    stuck_air_time = RewTerm(
        func=mdp.FlamingoAirTimeReward,
        weight=0.0,
        params={
            "stuck_threshold": 0.15,
            "stuck_duration": 50,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation_range_shoulder = RewTerm(
        func=mdp.joint_target_deviation_range_l1,
        weight=5.0,
        params={
            "min_angle": -0.301799,
            "max_angle": 0.154,
            "in_range_reward": 0.001,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]),
        },  # target: -0.261799
    )
    # joint_deviation_range_leg = RewTerm(
    #     func=mdp.joint_target_deviation_range_l1,
    #     weight=5.0,
    #     params={
    #         "min_angle": 0.51810467,
    #         "max_angle": 0.61810467,
    #         "in_range_reward": 0.001,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]),
    #     },  # target: 0.56810467
    # )
    dof_pos_limits_hip = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    dof_pos_limits_leg = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
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
    base_stand_still = RewTerm(
        func=mdp.stand_still_base,
        weight=-0.0,  # default: -0.005
        params={
            "std": math.sqrt(0.25),
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-2.0,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    leg_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-2.0,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # base_target_height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-100.0,
    #     params={"target_height": 0.32482, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )  # default: 0.35482, 28482 works better
    # base_range_height = RewTerm(
    #     func=mdp.base_height_range_reward,
    #     weight=15.0,
    #     params={
    #         "min_height": 0.32,
    #         "max_height": 0.35,
    #         "in_range_reward": 0.005,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #     },
    # )
    base_range_relative_height = RewTerm(
        func=mdp.base_height_range_relative_reward,
        weight=15.0,
        params={
            "min_height": 0.267,
            "max_height": 0.297,
            "in_range_reward": 0.005,
            "root_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "wheel_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
        },
    )
    # ! Terms below should be off if it is first training ! #
    # wheel_applied_torque_limits = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=-0.0,  # default: -0.025
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint")},
    # )
    # dof_vel_limits_wheel = RewTerm(
    #     func=mdp.joint_vel_limits,
    #     weight=-0.0,  # default: -0.01
    #     params={
    #         "soft_ratio": 0.3,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
    #     },
    # )
    # ! Terms above should be off if it is first training ! #

    # action_smoothness = RewTerm(func=mdp.action_smoothness_hard, weight=-0.0)


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
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (
            0.0,
            0.1,
        )  # Very gentle slope
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = (
            0.1  # Increase proportion if you want more of this terrain
        )
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (
            0.0,
            0.01,
        )  # Very gentle slope
        # Adjust the inverted pyramid stairs terrain
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (
            0.01,
            0.1,
        )  # Smaller step height for gentler steps
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_width = (
            0.25  # Increase step width for gentler steps
        )

        """ sub terrains
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,/
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
        },
        """
        # events
        self.events.push_robot = None
        # self.events.push_robot.params = {
        #     "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        # }
        # # reset_robot_joint_zero should be called here
        # self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot.interval_range_s = (12.0, 15.0)

        # # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
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

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            # ".*_shoulder_link",
            ".*_leg_link",
        ]

        # rewards
        # self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.dof_torques_l2.weight = -2.5e-7  # default: -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight *= 1.5  # default: 1.5
        self.rewards.dof_acc_l2.weight *= 1.0  # default: 1.5

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.1531942, 0.3731942)
