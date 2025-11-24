# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.actuators.actuator_cfg import (
    GearDelayedPDActuatorCfg,
)
from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR


KOCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo_Arm/KOCH_rev_1_0_0/asset/follower.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "gripper_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint_12": DelayedPDActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            effort_limit=10.0,
            velocity_limit=100.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "joint_1": 20.0,
                "joint_2": 20.0,
            },
            damping={
                "joint_1": 2.0,
                "joint_2": 2.0,
            },
            friction={
                "joint_1": 0.0,
                "joint_2": 0.0,
            },
            armature={
                "joint_1": 0.01,
                "joint_2": 0.01,
            },
        ),
        "joint_345": DelayedPDActuatorCfg(
            joint_names_expr=["joint_3", "joint_4", "joint_5"],
            effort_limit=3.5,
            velocity_limit=100.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "joint_3": 15.0,
                "joint_4": 15.0,
                "joint_5": 15.0,
            },
            damping={
                "joint_3": 1.5,
                "joint_4": 1.5,
                "joint_5": 1.5,
            },
            friction={
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
            },
            armature={
                "joint_3": 0.01,
                "joint_4": 0.01,
                "joint_5": 0.01,
            },
        ),
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper_joint"],
            effort_limit=3.5,
            velocity_limit=100.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "gripper_joint": 15.0,
            },
            damping={
                "gripper_joint": 1.5,
            },
            friction={
                "gripper_joint": 0.0,
            },
            armature={
                "gripper_joint": 0.01,
            },
        ),
    },
)

KOCH_CFG_HIGH_PD_CFG = KOCH_CFG.copy()
KOCH_CFG_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True