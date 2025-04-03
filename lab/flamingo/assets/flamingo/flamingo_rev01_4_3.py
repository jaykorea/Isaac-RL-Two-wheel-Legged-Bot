# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
)

from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR


FLAMINGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo/flamingo_rev01_4_3/flamingo_rev01_4_3_merge_joints.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36288),  # default: 0.36288
        joint_pos={
            "left_hip_joint": 0.0,
            "left_shoulder_joint": 0.0,
            "left_leg_joint": -0.0,
            "left_wheel_joint": 0.0,
            "right_hip_joint": 0.0,
            "right_shoulder_joint": 0.0,
            "right_leg_joint": -0.0,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    actuators={
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"],
            effort_limit=45.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_hip_joint": 100.0,
                ".*_shoulder_joint": 100.0,
                ".*_leg_joint": 150.0,
            },
            damping={
                ".*_hip_joint": 2.5,
                ".*_shoulder_joint": 3.0,
                ".*_leg_joint": 2.5,
            },
            friction={
                ".*_hip_joint": 0.0,
                ".*_shoulder_joint": 0.0,
                ".*_leg_joint": 0.0,
            },
            armature={
                ".*_hip_joint": 0.01,
                ".*_shoulder_joint": 0.01,
                ".*_leg_joint": 0.01,
            },
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=45.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={".*_wheel_joint": 0.4},
            friction={
                ".*_wheel_joint": 0.0,
            },
            armature={
                ".*_wheel_joint": 0.01,
            },
        ),
    },
)
