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
    CommandsCfg,
)

import lab.flamingo.tasks.constraint_based.locomotion.velocity.flamingo_env.flat_env.back_flip.rewards as backflip_mdp

from lab.flamingo.assets.flamingo.flamingo_rev01_5_1 import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingocommandsCfg(CommandsCfg):
    backflip_commands = mdp.BackflipCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 2.0),
        debug_vis=True,
    )


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
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_shoulder_link", ".*_leg_link"]), "threshold": 1.0},
        p_max=1.0,
        time_out="constraint",
    )


@configclass
class FlamingoRewardsCfg():

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
        weight=-0.5,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    leg_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.5,  # default: -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-7)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)  # default: -0.01

    """
        Backflip Reward
        
        changed description : 
        
        flat_orientiation : -10.0 -> -20.0
        gravitiy_y : delete
        reward_angvel_y : 9 -> 5
        
    """

    flat_orientation_l2_backflip = RewTerm(func=backflip_mdp.flat_orientation_l2_backflip, weight=-10.0)
      
    joint_deviation_l1_backfilp = RewTerm(
        func=backflip_mdp.joint_deviation_l1_backfilp,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),}
    )


    penalty_action_symmetry = RewTerm(
        func=backflip_mdp.reward_action_symmetry,
        weight = -5.0,
        params = {"asset_cfg": SceneEntityCfg("robot")}
    )
    
    penalty_over_height = RewTerm(
        func = backflip_mdp.penalty_over_height,
        weight = -50.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "max_height" : 0.75,   
        }
    )

    # penalty_base_height = RewTerm(
    #     func=backflip_mdp.base_height_adaptive_l2,
    #     weight=-5.0,
    #     params={
    #         "target_height" : 0.36288,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #     }
    # )

    penalty_ang_vel_z = RewTerm(
        func=backflip_mdp.penalty_ang_vel_z,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),}
    )

    penalty_ang_vel_x = RewTerm(
        func=backflip_mdp.penalty_ang_vel_x,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),}
    )
        
    penalty_xy_lin_vel = RewTerm(
        func=backflip_mdp.penalty_xy_lin_vel_w,
        weight=-5.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),}
    )

    reward_ang_vel_y = RewTerm(
        func=backflip_mdp.reward_ang_vel_y,
        weight=7.0,
        params = {"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),}
    )

    reward_linear_vel_z = RewTerm(
        func=backflip_mdp.reward_linear_vel_z,
        weight=7.0,
        params = {"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),}
    )

    reward_complete_backflip = RewTerm(
        func=backflip_mdp.Reward_complete_backflip,
        weight=200.0, # 30
        params={"asset_cfg": SceneEntityCfg("robot"),}
    )

    termination_penalty = RewTerm(func=mdp.is_terminated_cons, weight=-200.0)


@configclass
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    actions: FlamingoActionsCfg = FlamingoActionsCfg()
    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    commands: FlamingocommandsCfg = FlamingocommandsCfg()
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 3.5
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
        self.observations.none_stack_critic.roll_pitch_commands = None
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot = None

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
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.0, 0.0),
            },
        }

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
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
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

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
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # constratins
        self.constraints.base_contact.params["sensor_cfg"].body_names = ["base_link"]