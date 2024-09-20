# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from omni.isaac.lab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.assets.flamingo import FLAMINGO_WALK_CFG  # isort: skip
from lab.flamingo.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    CommandsCfg,
)


@configclass
class FalmingoCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityWithZCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityWithZCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.5, 1.5),
            pos_z=(0.1931942, 0.3531942),
        ),
    )


@configclass
class FlamingoRoughEnvCfg_PLAY(LocomotionVelocityRoughEnvCfg):

    commands: FalmingoCommandsCfg = FalmingoCommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 5.0
        # scene
        self.scene.robot = FLAMINGO_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False
        # # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # events
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.physics_material.params["asset_cfg"].body_names = [".*"]
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.0, 0.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]

        # commands
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
