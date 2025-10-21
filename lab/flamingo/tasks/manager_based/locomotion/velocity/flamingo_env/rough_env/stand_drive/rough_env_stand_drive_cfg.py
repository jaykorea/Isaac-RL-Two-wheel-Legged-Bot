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
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env.rough_env.stand_drive.drive_rewards as mdp_drive
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_rev03_1_1 import FLAMINGO_CFG  # isort: skip


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


@configclass
class FlamingoRewardsCfg():
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    penalize_ang_vel_z_when_lin_vel_y = RewTerm(
        func=mdp_drive.reward_ang_vel_z_link_exp,
        weight=-5.5,
        params={"command_name": "base_velocity"},
    )

    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )
    # lin_vel_z_event = RewTerm(
    #     func=mdp_drive.foot_lin_vel_z_mask,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg_left": SceneEntityCfg("left_mask_sensor"),
    #         "sensor_cfg_right": SceneEntityCfg("right_mask_sensor"),
    #         "max_up_vel": 2.0,
    #         "up_vel_coef": 5.0,
    #         "temperature": 2.0,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_static_link"]),
    #     }
    # )
    # lin_vel_z_event = RewTerm(
    #     func=mdp_drive.body_lin_vel_z_mask,
    #     weight=1.5,
    #     params={
    #         "sensor_cfg_left": SceneEntityCfg("left_mask_sensor"),
    #         "sensor_cfg_right": SceneEntityCfg("right_mask_sensor"),
    #         "max_up_vel": 2.5,
    #         "up_vel_coef": 10.0,
    #         "temperature": 2.0,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #     }
    # )

    # push_ground_event = RewTerm(
    #     func = mdp_drive.reward_push_ground_terrain,
    #     weight=0.05,
    #     params= {
    #         "sensor_cfg_left": SceneEntityCfg("left_mask_sensor"),
    #         "sensor_cfg_right": SceneEntityCfg("right_mask_sensor"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
    #     }
    # )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-0.1)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    # feet_distance_reward = RewTerm(
    #     func=mdp_drive.reward_feet_distance,
    #     weight=-25.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_static_link"]),
    #         "min_feet_distance": 0.4785,
    #         "max_feet_distance": 0.4995,
    #     },
    # )
    # nominal_foot_position_tracking = RewTerm(
    #     func=mdp_drive.reward_nominal_foot_position_adaptive,
    #     weight=4.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_static_link"]),
    #         "sensor_cfg_left": SceneEntityCfg("left_wheel_height_scanner"),
    #         "sensor_cfg_right": SceneEntityCfg("right_wheel_height_scanner"),
    #         "command_name": "base_velocity",
    #         "base_height_target": 0.36288,
    #         "foot_radius": 0.127,
    #         "temperature": 400.0,
    #         "sigma_wrt_v": 0.5,
    #     },
    # )
    same_foot_x_position = RewTerm(
        func=mdp_drive.reward_same_foot_x_position,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_static_link"])},
    )
    # reward_same_foot_y_position = RewTerm(
    #     func=mdp_drive.reward_same_foot_y_position,
    #     weight=-100.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_static_link"])},
    # )
    # leg_symmetry = RewTerm(
    #     func=mdp_drive.reward_leg_symmetry,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_static_link"]),
    #         "temperature": 1000.0,
    #     },
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
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_leg_link"]),
            "threshold": 1.0,
        },
    )
    # shoulder_align_l1 = RewTerm(
    #     func=mdp.joint_align_l1,
    #     weight=-0.5,  # default: -0.5
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    # )

    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.025,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    base_range_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-50.0,
        params={
            "target_height": 0.40288,  # 0.36288
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_joint).*")})
    wheel_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint")})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)
    wheel_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01


@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

       #! ************** scene & observations setup - 0 *********** !#
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        self.observations.none_stack_critic.base_height_scan = None
        self.observations.none_stack_critic.left_wheel_height_scan = None
        self.observations.none_stack_critic.right_wheel_height_scan = None
        self.observations.none_stack_critic.lift_mask = None
        #! ********************************************************* !#

        #! ****************** Observations setup ****************** !#
        self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None
        self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.base_pos_z = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        self.observations.none_stack_policy.lift_mask = None

        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.event_commands = None
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.event_commands = None
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (10.0, 15.0)
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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
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
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True

        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        #! ****************** Observations setup ******************* !#
        # disable randomization for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-1.0, 0.5)},
        }
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
            ".*_shoulder_link",
            # ".*_leg_link",
        ]

        self.commands.base_velocity.resampling_time_range = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
 