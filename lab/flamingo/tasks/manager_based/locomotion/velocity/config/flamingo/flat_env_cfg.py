# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

# import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    RewardsCfg,
)


##
# Pre-defined configs
##
# from omni.isaac.orbit_assets.flamingo import FLAMINGO_CFG
from lab.flamingo.assets.flamingo import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    # joint_deviation_range_shoulder = RewTerm(
    #     func=mdp.joint_target_deviation_range_l1,
    #     weight=15.0,
    #     params={
    #         "min_angle": -0.301799,
    #         "max_angle": 0.0,
    #         "in_range_reward": 0.001,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]),
    #     },  # target: -0.261799
    # )
    # joint_deviation_range_leg = RewTerm(
    #     func=mdp.joint_target_deviation_range_l1,
    #     weight=15.0,
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
    base_range_height = RewTerm(
        func=mdp.base_height_range_reward,
        weight=15.0,
        params={
            "min_height": 0.32,
            "max_height": 0.35,
            "in_range_reward": 0.005,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
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
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.observations.policy.enable_corruption = True

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot.interval_range_s = (12.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.5)
        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.8)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.0, 0.0),
            },
        }
        # rewards
        # self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.dof_torques_l2.weight = -2.5e-5  # default: -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight *= 1.5  # default: 1.5
        self.rewards.dof_acc_l2.weight *= 1.5  # default: 1.5

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # height scan
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.height_scanner.debug_vis = False

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]
