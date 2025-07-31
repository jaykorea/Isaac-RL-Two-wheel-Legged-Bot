# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
    ImplicitActuatorCfg
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.actuators.actuator_cfg import (
    ActuatorNetKANCfg,
)
from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR

# FLAMINGO_WHEEL_ACTUATOR_KAN_CFG = ActuatorNetKANCfg(
#             joint_names_expr=[".*_wheel_joint"],
#             symbolic_formula=f"{FLAMINGO_ASSETS_DATA_DIR}/ActuatorNets/Flamingo/kan/wheel/symbolic_formula.txt",
#             saturation_effort=60.0,
#             velocity_limit=20.0,
#             pos_scale=1.0,
#             vel_scale=1.0,
#             torque_scale=1.0,
#             input_order="vel_pos",
#             input_idx=[0,1,2],
#         )

HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo/humanoid_rev_2_1_0/humanoid_rev_2_1_0.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_yaw_joint": 0.0,
            ".*_elbow_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit=200,
            velocity_limit=100.0,
            stiffness={
                ".*_ankle_roll_joint": 40.0,
                ".*_ankle_pitch_joint": 40.0,
            },
            damping={
                ".*_ankle_roll_joint": 4.0,
                ".*_ankle_pitch_joint": 4.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_yaw_joint", ".*_elbow_pitch_joint"],
            effort_limit=150,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 30.0,
                ".*_shoulder_roll_joint": 30.0,
                ".*_shoulder_yaw_joint": 5.0,
                ".*_elbow_yaw_joint": 5.0,
                ".*_elbow_pitch_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_yaw_joint": 1.0,
                ".*_elbow_pitch_joint": 2.0,
            },
        ),
    },
)
