# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Agility robots.

The following configurations are available:

* :obj:`CASSIE_CFG`: Agility Cassie robot with simple PD controller for the legs

Reference: https://github.com/UMich-BipedLab/Cassie_Model/blob/master/urdf/cassie.urdf
"""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import (
    DelayedPDActuatorCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.actuators.actuator_cfg import (
    ForceZeroActuatorCfg,
)
from omni.isaac.lab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR

# from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR
# from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR

FLAMINGO_WALK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo/flamingo_rev01/flamingo_cylinder2.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40168),  # default: 0.2461942(initial), 0.35482, 0.40168(stand)
        joint_pos={
            "left_hip_joint": 0.0,
            "left_shoulder_joint": -0.436332,
            "left_leg_joint": 0.820305,
            "left_wheel_joint": 0.0,
            "right_hip_joint": 0.0,
            "right_shoulder_joint": -0.436332,
            "right_leg_joint": 0.820305,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    # joint_positions=[0.0, 0.0, -0.2161799, -0.2161799, 0.56810467, 0.56810467] (initial)
    # joint_positions=[0.0, 0.0, -0.436332, -0.436332, 0.820305, 0.820305] (stand)
    actuators={
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"],
            effort_limit=23.0,
            velocity_limit=20.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=8,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_hip_joint": 85.0,
                ".*_shoulder_joint": 85.0,
                ".*_leg_joint": 85.0,
            },
            damping={
                ".*_hip_joint": 0.65,
                ".*_shoulder_joint": 0.65,
                ".*_leg_joint": 0.65,
            },
            friction={
                ".*_hip_joint": 0.0,
                ".*_shoulder_joint": 0.0,
                ".*_leg_joint": 0.0,
            },
            armature={
                ".*_hip_joint": 0.0,
                ".*_shoulder_joint": 0.0,
                ".*_leg_joint": 0.0,
            },
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=5.0,
            velocity_limit=55.0,
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=8,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={
                ".*_wheel_joint": 0.3,
            },
            friction={
                ".*_wheel_joint": 0.0,
            },
            armature={
                ".*_wheel_joint": 0.0,
            },
        ),
    },
)
