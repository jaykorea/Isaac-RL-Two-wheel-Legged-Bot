# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from lab.flamingo.isaaclab.isaaclab.managers import ConstraintTermCfg as DoneTerm
import lab.flamingo.tasks.constraint_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.constraint_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_rev01_5_1 import FLAMINGO_CFG  # isort: skip

@configclass
class FlamingoCurriculumCfg(CurriculumCfg):

    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            "mod_range": {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-3.14, 3.14)},
            "num_steps": 25000,
        },
    )

@configclass
class FlamingoActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_joint", "right_hip_joint", 
                     "left_shoulder_joint", "right_shoulder_joint", 
                     "left_leg_joint", "right_leg_joint"
                     ],
        scale=2.0,
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
class ConstraintsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(
        func=mdp.time_out_cons,
        time_out="truncate",
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds_cons,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out="truncate",
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact_hard,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_hip_link", ".*_shoulder_link", ".*_leg_link"]), "threshold": 1.0},
        p_max=1.0,
        time_out="terminate",
    )
    # dof_torques = DoneTerm(
    #     func=mdp.joint_torques_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    #     p_max=0.25,
    #     use_curriculum=True,
    #     time_out="constraint",
    # )
    # stand_still = DoneTerm(
    #     func=mdp.stand_still_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    #     p_max=0.25,
    #     use_curriculum=True,
    #     time_out="constraint",
    # )
    # joint_effort = DoneTerm(
    #     func=mdp.joint_effort_out_of_limit_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
    #     p_max=0.015,
    #     time_out="constraint",
    # )

    # vel_limits = DoneTerm(
    #     func=mdp.joint_vel_out_of_limit_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
    #     p_max=0.1,
    #     time_out="constraint",
    # )

    # joint_align_shoulder = DoneTerm(
    #     func=mdp.joint_align_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    #     p_max=0.25,
    #     time_out="constraint",
    # )
    # joint_align_leg = DoneTerm(
    #     func=mdp.joint_align_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"])},
    #     p_max=0.25,
    #     time_out="constraint",
    # )
    # flat_euler = DoneTerm(
    #     func=mdp.flat_euler_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    #     p_max=0.25,
    #     time_out="constraint",
    # )
    # joint_deviation_hip = DoneTerm(
    #     func=mdp.joint_deviation_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    #     p_max=0.25,
    #     time_out="constraint",
    # )
    # joint_deviation_shoulder = DoneTerm(
    #     func=mdp.joint_deviation_soft,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    #     p_max=0.1,
    #     time_out="constraint",
    # )
    # base_height = DoneTerm(
    #     func=mdp.base_height_soft,
    #     params={"target_height": 0.36288, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    #     p_max=0.25,
    #     time_out="constraint",
    # )
    # lin_vel_z = DoneTerm(
    #     func=mdp.lin_vel_z_soft,
    #     p_max=0.05,
    #     time_out="constraint",
    # )
    # ang_vel_xy = DoneTerm(
    #     func=mdp.ang_vel_xy_soft,
    #     p_max=0.1,
    #     time_out="constraint",
    # )
    # dof_torques = DoneTerm(
    #     func=mdp.joint_torques_soft,
    #     p_max=0.05,
    #     time_out="constraint",
    # )
    # dof_acc = DoneTerm(
    #     func=mdp.joint_acc_soft,
    #     p_max=0.0001,
    #     time_out="constraint",
    # )
    # action_rate = DoneTerm(
    #     func=mdp.action_rate_soft,
    #     p_max=0.05,
    #     time_out="constraint",
    # )


@configclass
class FlamingoRewardsCfg():
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.1)

    joint_deviation_hip = RewTerm(
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
        weight=-0.1,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    leg_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.1,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )

    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-10.0)
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-10.0,
        params={
            "target_height": 0.36288, # default" 0.36288
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01

    termination_penalty = RewTerm(func=mdp.is_terminated_cons, weight=-200.0)


@configclass
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    actions: FlamingoActionsCfg = FlamingoActionsCfg()
    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # environment
        self.episode_length_s = 20.0
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        #! ****************** Observations setup - 0 *************** !#
        # observations
        self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.base_pos_z = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        self.observations.none_stack_policy.lift_mask = None

        if hasattr(self.observations.none_stack_policy.base_pos_z, "params"):
            self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None
        if hasattr(self.observations.none_stack_critic.base_pos_z, "params"):
            self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.back_flip = None
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.back_flip = None
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.5, 1.5)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 3.0)

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

        # height scan
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # self.scene.height_scanner.debug_vis = False

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

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
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-2.0, 2.0)},
        }
        # self.events.robot_wheel_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_wheel_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 3.0)

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

        # height scan
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # self.scene.height_scanner.debug_vis = False

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)