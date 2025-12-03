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
    ActuatorNetKANCfg,
    GearDelayedPDActuatorCfg,
)
from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR


WOLF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Wolf/wolf_rev_01_0_0/asset/Wolf_rev_1_0_0.usd",
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
        pos=(0.0, 0.0, 0.7),  # default: 여기에 울프 기본 높이지정
        joint_pos = {
            "HAA_front_left_joint": 0.0,
            "HFE_front_left_joint": 0.0,
            "KFE_front_left_joint": 0.0,

            "HAA_front_right_joint": 0.0,
            "HFE_front_right_joint": 0.0,
            "KFE_front_right_joint": 0.0,

            "HAA_back_left_joint": 0.0,
            "HFE_back_left_joint": 0.0,
            "KFE_back_left_joint": 0.0,
            "AFE_back_left_joint": 0.0,

            "HAA_back_right_joint": 0.0,
            "HFE_back_right_joint": 0.0,
            "KFE_back_right_joint": 0.0,
            "AFE_back_right_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    actuators={
        "joints_front_left_H": DelayedPDActuatorCfg(
            joint_names_expr=["HAA_front_left_joint", "HFE_front_left_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "HAA_front_left_joint": 80.0,
                "HFE_front_left_joint": 80.0,
            },
            damping={
                "HAA_front_left_joint": 1.0,
                "HFE_front_left_joint": 1.0,
            },
            friction={
                "HAA_front_left_joint": 0.0,
                "HFE_front_left_joint": 0.0,
            },
            armature={
                "HAA_front_left_joint": 0.01,
                "HFE_front_left_joint": 0.01,
            },
        ),
        "joints_front_left_K": DelayedPDActuatorCfg(
            joint_names_expr=["KFE_front_left_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps
            stiffness={
                "KFE_front_left_joint": 80.0,
            },
            damping={
                "KFE_front_left_joint": 1.0,
            },
            friction={
                "KFE_front_left_joint": 0.0,
            },
            armature={
                "KFE_front_left_joint": 0.01,
            },
        ),
        "joints_front_right_H": DelayedPDActuatorCfg(
            joint_names_expr=["HAA_front_right_joint", "HFE_front_right_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "HAA_front_right_joint": 80.0,
                "HFE_front_right_joint": 80.0,
            },
            damping={
                "HAA_front_right_joint": 1.0,
                "HFE_front_right_joint": 1.0,
            },
            friction={
                "HAA_front_right_joint": 0.0,
                "HFE_front_right_joint": 0.0,
            },
            armature={
                "HAA_front_right_joint": 0.01,
                "HFE_front_right_joint": 0.01,
            },
        ),
        "joints_front_right_K": DelayedPDActuatorCfg(
            joint_names_expr=["KFE_front_right_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps
            stiffness={
                "KFE_front_right_joint": 80.0,
            },
            damping={
                "KFE_front_right_joint": 1.0,
            },
            friction={
                "KFE_front_right_joint": 0.0,
            },
            armature={
                "KFE_front_right_joint": 0.01,
            },
        ),
        "joints_back_left_H": DelayedPDActuatorCfg(
            joint_names_expr=["HAA_back_left_joint", "HFE_back_left_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "HAA_back_left_joint": 80.0,
                "HFE_back_left_joint": 80.0,
            },
            damping={
                "HAA_back_left_joint": 1.0,
                "HFE_back_left_joint": 1.0,
            },
            friction={
                "HAA_back_left_joint": 0.0,
                "HFE_back_left_joint": 0.0,
            },
            armature={
                "HAA_back_left_joint": 0.01,
                "HFE_back_left_joint": 0.01,
            },
        ),
        "joints_back_left_KA": DelayedPDActuatorCfg(
            joint_names_expr=["KFE_back_left_joint", "AFE_back_left_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "KFE_back_left_joint": 80.0,
                "AFE_back_left_joint": 80.0,
            },
            damping={
                "KFE_back_left_joint": 1.0,
                "AFE_back_left_joint": 1.0,
            },
            friction={
                "KFE_back_left_joint": 0.0,
                "AFE_back_left_joint": 0.0,
            },
            armature={
                "KFE_back_left_joint": 0.01,
                "AFE_back_left_joint": 0.01,
            },
        ),
        "joints_back_right_H": DelayedPDActuatorCfg(
            joint_names_expr=["HAA_back_right_joint", "HFE_back_right_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "HAA_back_right_joint": 80.0,
                "HFE_back_right_joint": 80.0,
            },
            damping={
                "HAA_back_right_joint": 1.0,
                "HFE_back_right_joint": 1.0,
            },
            friction={
                "HAA_back_right_joint": 0.0,
                "HFE_back_right_joint": 0.0,
            },
            armature={
                "HAA_back_right_joint": 0.01,
                "HFE_back_right_joint": 0.01,
            },
        ),
        "joints_back_right_KA": DelayedPDActuatorCfg(
            joint_names_expr=["KFE_back_right_joint", "AFE_back_right_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "KFE_back_right_joint": 80.0,
                "AFE_back_right_joint": 80.0,
            },
            damping={
                "KFE_back_right_joint": 1.0,
                "AFE_back_right_joint": 1.0,
            },
            friction={
                "KFE_back_right_joint": 0.0,
                "AFE_back_right_joint": 0.0,
            },
            armature={
                "KFE_back_right_joint": 0.01,
                "AFE_back_right_joint": 0.01,
            },
        ),
    },
)