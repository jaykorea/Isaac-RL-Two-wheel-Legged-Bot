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
    ImplicitActuatorCfg,
    DCMotorCfg,
    IdealPDActuatorCfg,
    DelayedPDActuatorCfg,
    ActuatorNetMLPCfg,
    ActuatorNetLSTMCfg,
)
from omni.isaac.lab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR

# from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR
# from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR

FLAMINGO_JOINT_ACTUATOR_MLP_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"],
    network_file=f"{FLAMINGO_ASSETS_DATA_DIR}/ActuatorNets/Flamingo/mlp/joint/Flamingo_ActuatorNet_Joint.pt",
    pos_scale=1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.0,  # taken from spec sheet
    velocity_limit=20.0,  # taken from spec sheet
    saturation_effort=23.0,  # same as effort limit
)

FLAMINGO_WHEEL_ACTUATOR_MLP_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_wheel_joint"],
    network_file=f"{FLAMINGO_ASSETS_DATA_DIR}/ActuatorNets/Flamingo/mlp/wheel/Flamingo_ActuatorNet_Wheel2.pt",
    pos_scale=1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="vel_pos",
    input_idx=[0, 1, 2],
    effort_limit=6.0,  # taken from spec sheet
    velocity_limit=25.0,  # taken from spec sheet
    saturation_effort=6.0,  # same as effort limit
)

FLAMINGO_JOINT_ACTUATOR_LSTM_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"],
    network_file=f"{FLAMINGO_ASSETS_DATA_DIR}/ActuatorNets/Flamingo/lstm/joint/Flamingo_ActuatorNet_Joint.pt",
    input_order="pos_vel",
    effort_limit=23.0,  # taken from spec sheet
    velocity_limit=20.0,  # taken from spec sheet
    saturation_effort=23.0,  # same as effort limit
)

FLAMINGO_WHEEL_ACTUATOR_LSTM_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*_wheel_joint"],
    network_file=f"{FLAMINGO_ASSETS_DATA_DIR}/ActuatorNets/Flamingo/lstm/wheel/Flamingo_ActuatorNet_Wheel.pt",
    input_order="vel_pos",
    effort_limit=6.0,  # taken from spec sheet
    velocity_limit=25.0,  # taken from spec sheet
    saturation_effort=6.0,  # same as effort limitd
)


FLAMINGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo/flamingo_rev01/flamingo_cylinder.usd",
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
        pos=(0.0, 0.0, 0.2561942),  # default: 0.0 0.0 0.2461942
        joint_pos={
            "left_hip_joint": 0.0,
            "left_shoulder_joint": -0.0,
            "left_leg_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_hip_joint": 0.0,
            "right_shoulder_joint": -0.0,
            "right_leg_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    # joint_positions=[0.0, 0.0, -0.2161799, -0.2161799, 0.56810467, 0.56810467],
    actuators={
        # "joints": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"],
        #     effort_limit=23.0,
        #     velocity_limit=20.0,
        #     stiffness={
        #         ".*_hip_joint": 45.0,
        #         ".*_shoulder_joint": 45.0,
        #         ".*_leg_joint": 45.0,
        #     },
        #     damping={
        #         ".*_hip_joint": 2.0,
        #         ".*_shoulder_joint": 2.0,
        #         ".*_leg_joint": 2.0,
        #     },
        #     friction={
        #         ".*_hip_joint": 0.01,
        #         ".*_shoulder_joint": 0.01,
        #         ".*_leg_joint": 0.01,
        #     },
        #     armature={
        #         ".*_hip_joint": 0.1,
        #         ".*_shoulder_joint": 0.1,
        #         ".*_leg_joint": 0.1,
        #     },
        # ),
        # "wheels": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_wheel_joint"],
        #     effort_limit=6.0,
        #     velocity_limit=25.0,
        #     stiffness={
        #         ".*_wheel_joint": 0.0,
        #     },
        #     damping={
        #         ".*_wheel_joint": 2.0,
        #     },
        #     friction={
        #         ".*_wheel_joint": 0.0,
        #     },
        #     armature={
        #         ".*_wheel_joint": 0.01,
        #     },
        # ),
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"],
            effort_limit=23.0,
            velocity_limit=20.0,
            min_num_time_lags=0,  # physics time steps (min: 5.0*0=0.0ms)
            max_num_time_lags=4,  # physics time steps (max: 5.0*5=25.0ms)
            stiffness={
                ".*_hip_joint": 75.0,
                ".*_shoulder_joint": 75.0,
                ".*_leg_joint": 75.0,
            },
            damping={
                ".*_hip_joint": 2.0,
                ".*_shoulder_joint": 2.0,
                ".*_leg_joint": 2.0,
            },
            friction={
                ".*_hip_joint": 0.01,
                ".*_shoulder_joint": 0.01,
                ".*_leg_joint": 0.01,
            },
            armature={
                ".*_hip_joint": 0.1,
                ".*_shoulder_joint": 0.1,
                ".*_leg_joint": 0.1,
            },
        ),
        # "wheels": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*_wheel_joint"],
        #     effort_limit=5.0,
        #     velocity_limit=25.0,
        #     min_num_time_lags=0,  # physics time steps (min: 2.0*0=0.0ms)
        #     max_num_time_lags=4,  # physics time steps (max: 2.0*4=8.0ms)
        #     stiffness={
        #         ".*_wheel_joint": 0.0,
        #     },
        #     damping={
        #         ".*_wheel_joint": 2.0,
        #     },
        #     friction={
        #         ".*_wheel_joint": 0.1,
        #     },
        #     armature={
        #         ".*_wheel_joint": 0.01,
        #     },
        # ),
        # "joints": FLAMINGO_JOINT_ACTUATOR_MLP_CFG,
        "wheels": FLAMINGO_WHEEL_ACTUATOR_LSTM_CFG,
    },
)
