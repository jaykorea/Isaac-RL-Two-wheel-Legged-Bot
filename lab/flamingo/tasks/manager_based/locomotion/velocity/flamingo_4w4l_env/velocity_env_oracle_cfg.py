# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import lab.cocelo.tasks.manager_based.locomotion.velocity.mdp as mdp


##
# Pre-defined configs
##
from lab.cocelo.tasks.manager_based.locomotion.velocity.terrain_config.rough_config import ROUGH_TERRAINS_CFG

##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75), intensity=4000.0
        ), # Warmer color with higher intensity
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.53, 0.81, 0.98), intensity=1500.0
        ), # Sky blue color with increased intensity
    )

    # camera = CameraCfg(
    # prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
    # update_period=0.1,
    # height=480,
    # width=640,
    # data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
    # semantic_filter=["cone"],
    # spawn=sim_utils.PinholeCameraCfg(
    # focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.6, 6)
    # ),
    # offset=CameraCfg.OffsetCfg(rot=(1, 0.0, 0.0, 0.0), convention="world"),
    # )
    # self.intrinsic_matrix = compute_intrinsic_matrix(focal_length=24.0, width=640, height=480, horizontal_aperture=20.955)



##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    #fix
    base_velocity = mdp.UniformVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=0.0,
    debug_vis=True,
    ranges=mdp.DiscreteAngularVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.4, 0.4)
    ),
    )

    # base_velocity = mdp.UniformLevelVelocityCommandCfg(
    # asset_name="robot",
    # resampling_time_range=(10.0, 10.0),
    # rel_standing_envs=0.02,
    # rel_heading_envs=0.0,
    # debug_vis=True,
    # ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    # lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.1, 0.1)
    # ),
    # limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    # lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0)
    # ),
    # )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    F_hip_joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["FL_hip_joint", "FR_hip_joint"],
    scale=1.0,
    use_default_offset=False,
    preserve_order=True,
    )
    F_shoulder_leg_joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["FL_shoulder_joint", "FR_shoulder_joint", "FL_leg_joint", "FR_leg_joint"],
    scale=1.0,
    use_default_offset=False,
    preserve_order=True,
    )
    F_wheel_vel = mdp.JointVelocityActionCfg(
    asset_name="robot",
    joint_names=["FL_wheel_joint", "FR_wheel_joint"],
    scale=40.0,
    use_default_offset=False,
    preserve_order=True
    )
    R_hip_joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["RL_hip_joint", "RR_hip_joint"],
    scale=1.0,
    use_default_offset=False,
    preserve_order=True,
    )
    R_shoulder_leg_joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["RL_shoulder_joint", "RR_shoulder_joint", "RL_leg_joint", "RR_leg_joint"],
    scale=1.0,
    use_default_offset=False,
    preserve_order=True,
    )
    R_wheel_vel = mdp.JointVelocityActionCfg(
    asset_name="robot",
    joint_names=["RL_wheel_joint", "RR_wheel_joint"],
    scale=40.0,
    use_default_offset=False,
    preserve_order=True
    )
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class StackCriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        joint_pos_f_nl = ObsTerm(
        func=mdp.joint_pos,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "FL_shoulder_joint", "FR_shoulder_joint"]),
        },
        )
        joint_pos_f_l = ObsTerm(
        func=mdp.joint_pos_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_leg_joint", "FR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        )
        joint_pos_r_nl = ObsTerm(
        func=mdp.joint_pos,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RR_hip_joint", "RL_shoulder_joint", "RR_shoulder_joint"]),
        },
        )
        joint_pos_r_l = ObsTerm(
        func=mdp.joint_pos_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_leg_joint", "RR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        )
        joint_vel_f_nl = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "FL_shoulder_joint", "FR_shoulder_joint"]),
        },
        scale=0.15) # default: -1.5
        joint_vel_f_l = ObsTerm(
        func=mdp.joint_vel_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_leg_joint", "FR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        scale=0.15) # default: -1.5
        joint_vel_f_wheel = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_wheel_joint", "FR_wheel_joint"]),
        },
        scale=0.15) # default: -1.5
        joint_vel_r_nl = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RR_hip_joint", "RL_shoulder_joint", "RR_shoulder_joint"]),
        },
        scale=0.15) # default: -1.5
        joint_vel_r_l = ObsTerm(
        func=mdp.joint_vel_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_leg_joint", "RR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        scale=0.15) # default: -1.5
        joint_vel_r_wheel = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_wheel_joint", "RR_wheel_joint"]),
        },
        scale=0.15) # default: -1.5
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, scale=0.25) # default: -0.15
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity) # default: -0.05
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            #self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True


        velocity_commands = ObsTerm(func=mdp.generated_scaled_commands, params={"command_name": "base_velocity", "scale": (2.0, 1.0, 0.25)})
        height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), 'offset': 0.0},
        clip=(-1.0, 1.0),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_link, scale=2.0)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque, scale=0.05)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc, scale=0.0025)
        contact_forces = ObsTerm(func=mdp.measure_contact_forces, scale=0.01, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"])})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StackPolicyCfg(ObsGroup):
        """Observations for Stack policy group."""
        joint_pos_f_nl = ObsTerm(
        func=mdp.joint_pos,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "FL_shoulder_joint", "FR_shoulder_joint"]),
        },
        noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_f_l = ObsTerm(
        func=mdp.joint_pos_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_leg_joint", "FR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_r_nl = ObsTerm(
        func=mdp.joint_pos,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RR_hip_joint", "RL_shoulder_joint", "RR_shoulder_joint"]),
        },
        noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_r_l = ObsTerm(
        func=mdp.joint_pos_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_leg_joint", "RR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_f_nl = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "FL_shoulder_joint", "FR_shoulder_joint"]),
        },
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.15) # default: -1.5
        joint_vel_f_l = ObsTerm(
        func=mdp.joint_vel_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_leg_joint", "FR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.15) # default: -1.5
        joint_vel_f_wheel = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_wheel_joint", "FR_wheel_joint"]),
        },
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.15) # default: -1.5
        joint_vel_r_nl = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RR_hip_joint", "RL_shoulder_joint", "RR_shoulder_joint"]),
        },
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.15) # default: -1.5
        joint_vel_r_l = ObsTerm(
        func=mdp.joint_vel_leg_gear,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_leg_joint", "RR_leg_joint"]),
        "gear_ratio": -1.5,
        },
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.15) # default: -1.5
        joint_vel_r_wheel = ObsTerm(
        func=mdp.joint_vel,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_wheel_joint", "RR_wheel_joint"]),
        },
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.15) # default: -1.5
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel_link, scale=2.0, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, noise=Unoise(n_min=-0.15, n_max=0.15), scale=0.25)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.005, n_max=0.005))
        def __post_init__(self):
            #self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True


        """Observations for None-Stack policy group."""
        velocity_commands = ObsTerm(func=mdp.generated_scaled_commands, params={"command_name": "base_velocity", "scale": (2.0, 1.0, 0.25)})
        height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), 'offset': 0.0},
        clip=(-1.0, 1.0),
        noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: StackPolicyCfg = StackPolicyCfg()
    critic: StackCriticCfg = StackCriticCfg()



@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),#(0.3, 1.2),
            "dynamic_friction_range": (0.6, 0.8),#(0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_joint_actuator_gains = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="startup",
    params={
    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
    "stiffness_distribution_params": (0.7, 1.3),
    "damping_distribution_params": (0.7, 1.3),
    "operation": "scale",
    "distribution": "log_uniform",
    },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_com_positions,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    # func=mdp.illegal_contact,
    # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    # )
    terrain_out_of_bounds = DoneTerm(
    func=mdp.terrain_out_of_bounds,
    params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    time_out=True,
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    #lin_vel_cmd_levels = CurrTerm(my_mdp.lin_vel_cmd_levels)
    #ang_vel_cmd_levels = CurrTerm(my_mdp.ang_vel_cmd_levels)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards = None # It will be defined in the task
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self. scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
