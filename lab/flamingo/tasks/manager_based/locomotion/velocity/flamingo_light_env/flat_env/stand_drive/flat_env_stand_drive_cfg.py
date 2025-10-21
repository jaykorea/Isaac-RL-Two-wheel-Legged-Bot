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
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoEduActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_joint", "right_shoulder_joint"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=20.0,
        use_default_offset=False,
        preserve_order=True
    )

@configclass
class FlamingoRewardsCfg():
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.1)

    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
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
        weight=-0.25,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )

    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-10.0)
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-500.0,
        params={
            "target_height": 0.310, # default" 0.310
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    dof_torques_joints_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    dof_torques_wheels_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )

    dof_acc_joints_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    dof_acc_wheels_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    same_foot_x_position = RewTerm(
        func=mdp.reward_same_foot_x_position,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"])},
    )
    
@configclass
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    actions: FlamingoEduActionsCfg = FlamingoEduActionsCfg()
    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # environment
        self.episode_length_s = 20.0
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None
        
        #! ****************** Observations setup - 0 *************** !#
        # observations
        self.observations.stack_policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_shoulder_joint")
        self.observations.stack_policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_shoulder_joint", ".*_wheel_joint"])
        self.observations.stack_critic.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_shoulder_joint")
        self.observations.stack_critic.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_shoulder_joint", ".*_wheel_joint"])
        self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.base_pos_z = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        self.observations.none_stack_policy.lift_mask = None
        self.observations.none_stack_policy.height_scan = None
        
        if hasattr(self.observations.none_stack_policy.base_pos_z, "params"):
            self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None
        if hasattr(self.observations.none_stack_critic.base_pos_z, "params"):
            self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.event_commands = None
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.event_commands = None
        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_critic.base_height_scan = None
        self.observations.none_stack_critic.left_wheel_height_scan = None
        self.observations.none_stack_critic.right_wheel_height_scan = None
        self.observations.none_stack_critic.lift_mask = None
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
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

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "left_leg_link",
            "right_leg_link",
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
        #! ****************** Observations setup - 0 *************** !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        self.events.push_robot.interval_range_s = (5.5, 6.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        }
        # self.events.robot_wheel_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_wheel_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.5708, 1.5708)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "left_leg_link",
            "right_leg_link",
        ]