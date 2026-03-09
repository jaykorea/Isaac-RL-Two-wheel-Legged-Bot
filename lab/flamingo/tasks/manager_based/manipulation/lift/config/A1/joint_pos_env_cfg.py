# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from dataclasses import MISSING

from isaaclab_tasks.manager_based.manipulation.lift import mdp
import lab.flamingo.tasks.manager_based.manipulation.lift.mdp as lift_mdp

from lab.flamingo.tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from lab.flamingo.tasks.manager_based.manipulation.lift.lift_env_cfg import TerminationsCfg
from lab.flamingo.tasks.manager_based.manipulation.lift.lift_env_cfg import ObservationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lab.flamingo.assets.flamingo.a1_rev03_3_0 import A1_CFG, A1_HIGH_PD_CFG  # isort: skip

@configclass
class A1ObservationsCfg(ObservationsCfg):
    """Observations for the MDP."""

    @configclass
    class ObsInfoCfg(ObsGroup):
        """Observation info group."""

        cbf = ObsTerm(
            func=lift_mdp.obstacle_ee_aabb_distance,
            params={
                "obstacle_cfg": SceneEntityCfg("screen"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "margin": 0.025,
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    obs_info: ObsInfoCfg = ObsInfoCfg()

@configclass
class A1RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.15}, weight=1.5)

    reaching_object_fine_graned = RewTerm(func=mdp.object_ee_distance, params={"std": 0.075}, weight=3.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=35.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.25, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=5.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=10.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class A1CommandsCfg:
    """Command terms for the MDP."""

    object_pose = lift_mdp.ReversePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        obstacle_name="screen",
        ee_name="ee_frame",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=lift_mdp.ReversePoseCommandCfg.Ranges(
            pos_x=(0.25, 0.3), pos_y=(-0.3, 0.3), pos_z=(0.055, 0.075), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.25, 0.3), pos_y=(-0.3, 0.3), pos_z=(0.2, 0.25), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )


@configclass
class A1CubeTerminationsCfg(TerminationsCfg):
    obstacle_fall_down = DoneTerm(
        func=lift_mdp.obstacle_fall_down,
        params={
            "obstacle_cfg": SceneEntityCfg("screen"),
            "threshold": 0.25},
    )

@configclass
class A1EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=lift_mdp.reset_root_state_binary,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.4), "y": (-0.3, 0.3)},
            "velocity_range": {},
            "unoise": 0.05,
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.0, 0.35), "y": (-0.3, 0.3), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (1.5, 1.5),
    #         "dynamic_friction_range": (1.2, 1.2),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 1,
    #     },
    # )

@configclass
class A1CubeLiftEnvCfg(LiftEnvCfg):
    observations: A1ObservationsCfg = A1ObservationsCfg()
    rewards: A1RewardsCfg = A1RewardsCfg()
    commands: A1CommandsCfg = A1CommandsCfg()
    events: A1EventCfg = A1EventCfg()
    terminations: A1CubeTerminationsCfg = A1CubeTerminationsCfg()
    curriculum: None = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set A1 as robot
        self.scene.robot = A1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (A1)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["dof.*_joint"], scale=1.0, use_default_offset=True
        )
        # self.actions.gripper_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=[".*_gripper_joint"], scale=1.0, use_default_offset=True
        # )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_gripper_joint"],
            open_command_expr={".*_gripper_joint": 0.04},
            close_command_expr={".*_gripper_joint": 0.00},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "gripper_tip_link"

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["dof5_link", "dof6_link"]

        # Set Cube as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0, 0.055], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(0.7, 0.7, 0.7),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.055, 0.055, 0.11),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5, dynamic_friction=1.0, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.025], rot=[1, 0, 0 ,0]),
            debug_vis=False,
        )

        self.scene.screen = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Screen",
            spawn=sim_utils.CuboidCfg(
                size=(0.45, 0.035, 0.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=2.5, dynamic_friction=2.0, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.475, 0, 0.25], rot=[1, 0, 0 ,0]),
            debug_vis=False,
        )

        # sensors
        self.scene.tv_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/tv_camera_link",
            update_period=0.1,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
            ),
            # offset=CameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.04), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
            offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 1.3), rot=(0, 1, 0, 0), convention="ros"),
        )

        self.scene.ee_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_link/ee_camera_link",
            update_period=0.1,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
            ),
            # offset=CameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.04), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.70711, 0.0 ,-0.70711 ,0.0), convention="opengl"),
        )


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_tip_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )


@configclass
class A1CubeLiftEnvCfg_PLAY(A1CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.none_stack_policy.enable_corruption = False
        self.observations.stack_policy.enable_corruption = False
