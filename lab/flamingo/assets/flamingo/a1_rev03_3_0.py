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


A1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo_Arm/A1_rev_3_3_0/asset/A1_rev_3_3_0.usd",
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
        joint_pos={
            "dof1_joint": 0.0,
            "dof2_joint": -0.523599,
            "dof3_joint": 0.523599,
            "dof4_joint": 0.0,
            "dof5_joint": -3.49066,
            "dof6_joint": 0.0,
            "left_gripper_joint": 0.0,
            "right_gripper_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint_1": DelayedPDActuatorCfg(
            joint_names_expr=["dof1_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "dof1_joint": 45.0,
            },
            damping={
                "dof1_joint": 5.0,
            },
            friction={
                "dof1_joint": 0.0,
            },
            armature={
                "dof1_joint": 0.01,
            },
        ),
        "joint_2": DelayedPDActuatorCfg(
            joint_names_expr=["dof2_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "dof2_joint": 50.0,
            },
            damping={
                "dof2_joint": 5.0,
            },
            friction={
                "dof2_joint": 0.0,
            },
            armature={
                "dof2_joint": 0.01,
            },
        ),
        "joint_3": DelayedPDActuatorCfg(
            joint_names_expr=["dof3_joint"],
            effort_limit=36.0,
            velocity_limit=44.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "dof3_joint": 35.0,
            },
            damping={
                "dof3_joint": 3.5,
            },
            friction={
                "dof3_joint": 0.0,
            },
            armature={
                "dof3_joint": 0.01,
            },
        ),
        "joint_45": DelayedPDActuatorCfg(
            joint_names_expr=["dof4_joint", "dof5_joint"],
            effort_limit=14.0,
            velocity_limit=33.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "dof4_joint": 30.0,
                "dof5_joint": 30.0,
            },
            damping={
                "dof4_joint": 3.0,
                "dof5_joint": 3.0,
            },
            friction={
                "dof4_joint": 0.0,
                "dof5_joint": 0.0,
            },
            armature={
                "dof4_joint": 0.01,
                "dof5_joint": 0.01,
            },
        ),
        "joint_6": DelayedPDActuatorCfg(
            joint_names_expr=["dof6_joint"],
            effort_limit=5.5,
            velocity_limit=50.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "dof6_joint": 20.0,
            },
            damping={
                "dof6_joint": 1.0,
            },
            friction={
                "dof6_joint": 0.0,
            },
            armature={
                "dof6_joint": 0.01,
            },
        ),
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=[".*_gripper_joint"],
            effort_limit=5.5,
            velocity_limit=50.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_gripper_joint": 20.0,
            },
            damping={
                ".*_gripper_joint": 1.0,
            },
            friction={
                ".*_gripper_joint": 0.0,
            },
            armature={
                ".*_gripper_joint": 0.01,
            },
        ),
    },
)

A1_HIGH_PD_CFG = A1_CFG.copy()
A1_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True