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
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo/flamingo_edu_v1/flamingo_edu_v1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.21881),
        joint_pos={
            "left_shoulder_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_shoulder_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    actuators={
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_joint"],
            effort_limit=12.0,
            velocity_limit=30.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_shoulder_joint": 50.0,
            },
            damping={
                ".*_shoulder_joint": 1.5,
            },
            friction={
                ".*_shoulder_joint": 0.0,
            },
            armature={
                ".*_shoulder_joint": 0.01,
            },
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=12.0,
            velocity_limit=30.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=0,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={".*_wheel_joint": 0.2},
            friction={
                ".*_wheel_joint": 0.0,
            },
            armature={
                ".*_wheel_joint": 0.01,
            },
        ),
    },
)
