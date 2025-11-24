# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from lab.flamingo.assets.flamingo.a1_rev03_3_0 import A1_CFG, A1_HIGH_PD_CFG # isort: skip


##
# Rigid object lift environment.
##


@configclass
class A1CubeLiftEnvCfg(joint_pos_env_cfg.A1CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set A1 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = A1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (A1)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["dof.*_joint"],
            body_name="gripper_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",ik_params={"lambda_val": 0.2}),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1]),
        )


@configclass
class A1CubeLiftEnvCfg_PLAY(A1CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.none_stack_policy.enable_corruption = False
        self.observations.stack_policy.enable_corruption = False


##
# Deformable object lift environment.
##


@configclass
class A1TeddyBearLiftEnvCfg(A1CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        )

        # Make the end effector less stiff to not hurt the poor teddy bear
        self.scene.robot.actuators["gripper"].effort_limit_sim = 5.5
        self.scene.robot.actuators["gripper"].stiffness = 5.0
        self.scene.robot.actuators["gripper"].damping = 0.5

        # Disable replicate physics as it doesn't work for deformable objects
        # FIXME: This should be fixed by the PhysX replication system.
        self.scene.replicate_physics = False

        # Set events for the specific object type (deformable cube)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Remove all the terms for the state machine demo
        # TODO: Computing the root pose of deformable object from nodal positions is expensive.
        #       We need to fix that part before enabling these terms for the training.
        self.terminations.object_dropping = None
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.observations.stack_policy.object_position = None

@configclass
class A1TeddyBearLiftEnvCfg_PLAY(A1TeddyBearLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.none_stack_policy.enable_corruption = False
        self.observations.stack_policy.enable_corruption = False