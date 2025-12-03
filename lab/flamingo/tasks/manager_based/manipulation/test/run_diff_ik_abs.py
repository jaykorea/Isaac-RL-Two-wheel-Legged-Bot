# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="a1", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils  # ← math 유틸 전체 사용

from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.assets import RigidObjectCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from lab.flamingo.assets.flamingo.a1_rev03_3_0 import A1_CFG, A1_HIGH_PD_CFG  # isort:skip
from lab.flamingo.assets.flamingo.koch_rev01_0_0 import KOCH_CFG, KOCH_CFG_HIGH_PD_CFG  # isort:skip


# =============================================================================
# Differential IK "Action" 래퍼
# =============================================================================

class DifferentialIKAction:
    """Standalone Differential IK 'Action' wrapper.
    """

    def __init__(
        self,
        robot,
        robot_entity_cfg: SceneEntityCfg,
        ee_jacobi_idx: int,
        ik_cfg: DifferentialIKControllerCfg,
        num_envs: int,
        device: str,
        body_offset_pos=None,
        body_offset_rot=None,
    ):
        self._asset = robot
        self._joint_ids = robot_entity_cfg.joint_ids
        self._body_idx = robot_entity_cfg.body_ids[0]
        self._jacobi_body_idx = ee_jacobi_idx
        self._num_envs = num_envs
        self._device = device

        # IK 컨트롤러 생성
        self._ik = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)

        # body offset (EE local frame 기준)
        if body_offset_pos is not None:
            self._offset_pos = torch.tensor(body_offset_pos, device=device, dtype=torch.float32).repeat(num_envs, 1)
            if body_offset_rot is None:
                body_offset_rot = [1.0, 0.0, 0.0, 0.0]
            self._offset_rot = torch.tensor(body_offset_rot, device=device, dtype=torch.float32).repeat(num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """command: (num_envs, 7) = [pos(3), quat(4)] in base frame."""
        self._ik.set_command(command, ee_pos, ee_quat)

    def reset(self):
        self._ik.reset()

    # 현재 frame pose (offset 포함) in base frame
    def _compute_frame_pose_b(self):
        # body/world
        body_state = self._asset.data.body_state_w
        ee_pos_w = body_state[:, self._body_idx, 0:3]
        ee_quat_w = body_state[:, self._body_idx, 3:7]
        # root/world
        root_state = self._asset.data.root_state_w
        root_pos_w = root_state[:, 0:3]
        root_quat_w = root_state[:, 3:7]

        # EE pose in base frame
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # offset 적용 (EE local frame 기준)
        if self._offset_pos is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pos_b, ee_quat_b

    # Jacobian in base frame (+ offset 반영)
    def _compute_jacobian_b(self):
        # Jacobian in world frame
        jac_w = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._joint_ids
        ]  # (num_envs, 6, n_joints)

        # base rotation
        root_state = self._asset.data.root_state_w
        root_quat_w = root_state[:, 3:7]
        base_rot_mat = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        # 선속도/각속도 둘 다 base frame 으로 회전
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_rot_mat, jac_w[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot_mat, jac_w[:, 3:, :])

        # offset 있는 경우 Jacobian 보정
        if self._offset_pos is not None:
            # v_link = v_ee + ω_ee × r_link_ee
            jac_b[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos), jac_b[:, 3:, :]
            )
            # ω_link = R_offset * ω_ee
            jac_b[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jac_b[:, 3:, :])

        return jac_b

    def apply(self):
        """현재 pose/jacobian/joint state로 IK 수행 후 joint targets 설정."""
        ee_pos_b, ee_quat_b = self._compute_frame_pose_b()
        jac_b = self._compute_jacobian_b()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jac_b, joint_pos)
        self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)


class BinaryGripperAction:
    def __init__(self, robot, joint_names, open_positions, close_positions, num_envs, device):
        self._robot = robot
        self._joint_ids, _ = robot.find_joints(joint_names)
        self._open = torch.tensor(open_positions, device=device).repeat(num_envs, 1)
        self._close = torch.tensor(close_positions, device=device).repeat(num_envs, 1)
        self._action = torch.zeros(num_envs, 1, device=device)

    def set_action(self, action: torch.Tensor):
        """ action: float tensor (num_envs, 1) """
        self._action[:] = action
    
    def apply(self):
        mask = self._action <= 0  # negative = close
        target = torch.where(mask, self._close, self._open)
        self._robot.set_joint_position_target(target, joint_ids=self._joint_ids)

    def reset(self):
        self._action[:] = 1.0  # open on reset

# =============================================================================
# Scene Configuration
# =============================================================================

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple tabletop scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    screen: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/screen",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.075, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=150.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.5, dynamic_friction=2.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, 0, 0.25], rot=[1, 0, 0 ,0]),
    )
    # cube = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0 ,0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.8, 0.8, 0.8),
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

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.5, dynamic_friction=2.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0 ,0]),
    )
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # articulation
    if args_cli.robot == "a1":
        robot = A1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        cube.init_state.rot = [0, -0.707, -0.707, 0]
    elif args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "koch":
        robot = KOCH_CFG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        table.init_state.pos = [-0.5, 0.0, 0.0]
        table.init_state.rot = [-0.707, 0, 0, 0.707]

        cube.init_state.pos = [-0.3, 0.0, 0.055]
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, a1, koch")


# =============================================================================
# Simulator Loop
# =============================================================================

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    screen = scene["screen"]
    cube = scene["cube"]

    sim_dt = sim.get_physics_dt()

    if scene["contact_forces"] is not None:
        scene["contact_forces"].update_period = sim_dt

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # IK command buffer (pose: pos[0:3], quat[3:7])
    # DifferentialIKController.action_dim 이 보통 7 (position+orientation)
    ik_commands = torch.zeros(scene.num_envs, 7, device=robot.device)

    # Robot-specific parameters
    if args_cli.robot == "a1":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["dof.*_joint"], body_names=["gripper_tip_link"])
        body_offset_pos = None
        gripper_joint_names = ["left_gripper_joint", "right_gripper_joint"]
        open_positions = [0.04, 0.04]
        close_positions = [0.0, 0.0]
    elif args_cli.robot == "koch":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["gripper_tip_link"])
        body_offset_pos = None
        gripper_joint_names = ["gripper_joint"]
        open_positions = [0.04]
        close_positions = [0.0]
    elif args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        body_offset_pos = [0.0, 0.0, 0.1]
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
        body_offset_pos = None
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: a1, franka_panda, ur10")

    robot_entity_cfg.resolve(scene)

    # EE Jacobian index
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]



    # IK controller config (Action 내부에서 사용)
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    
    ik_action = DifferentialIKAction(
        robot=robot,
        robot_entity_cfg=robot_entity_cfg,
        ee_jacobi_idx=ee_jacobi_idx,
        ik_cfg=diff_ik_cfg,
        num_envs=scene.num_envs,
        device=sim.device,
        body_offset_pos=body_offset_pos,
        body_offset_rot=None,
    )

    gripper_action = BinaryGripperAction(
        robot=robot,
        joint_names=gripper_joint_names,
        open_positions=open_positions,
        close_positions=close_positions,
        num_envs=scene.num_envs,
        device=robot.device,
    )

    count = 0

    while simulation_app.is_running():
        if count % 150 == 0:
            count = 0
            # ① reset robot
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_state_to_sim(robot.data.default_root_state)
            robot.reset()

            screen.write_root_state_to_sim(screen.data.default_root_state.clone())
            screen.reset()

            # ② reset cube (with randomization)
            cube_state = cube.data.default_root_state.clone()

            # x ∈ [0.3, 0.5]
            cube_state[:, 0] = 0.5 + 0.2 * torch.rand(scene.num_envs, device=robot.device)

            # y ∈ [-0.25, -0.15] U [0.15, 0.25]
            y_abs = 0.15 + 0.10 * torch.rand(scene.num_envs, device=robot.device)  # [0.15, 0.25]
            sign = torch.where(torch.rand(scene.num_envs, device=robot.device) > 0.5, 1.0, -1.0)
            cube_state[:, 1] = y_abs * sign

            # 적용
            cube.write_root_state_to_sim(cube_state)
            cube.reset()

            # ③ reset IK action
            ik_action.reset()

            # ④ reset gripper action
            gripper_action.reset()

        else:
            # ① Simulation data fetch
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]

            # ② Cube pose → Base 좌표계로 변환
            cube_pos_w = cube.data.root_state_w[:, 0:3]
            cube_quat_w = cube.data.root_state_w[:, 3:7]

            goal_pos_b, goal_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                cube_pos_w,
                cube_quat_w,
            )

            # ③ IK 명령을 Object 위치로 업데이트
            ik_commands[:, 0:3] = goal_pos_b
            ik_commands[:, 3:7] = goal_quat_b
            ik_action.set_command(ik_commands)

            # 4 - gripper command 설정
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            # ③ Pose error
            pos_err = torch.norm(ee_pos_w - cube_pos_w, p=2, dim=1)
            quat_err = torch.abs(torch.sum(ee_quat_w * cube_quat_w, dim=1))  # → 1에 가까울수록 유사

            # ④ Threshold 판단
            GRIPPER_THRESHOLD = 0.03  # 3 cm
            close_mask = pos_err < GRIPPER_THRESHOLD

            gripper_cmd = torch.where(
                close_mask.unsqueeze(-1),
                torch.tensor([0.0], device=robot.device),   # close
                torch.tensor([1.0], device=robot.device)    # open
            )

            # ⑤ 적용
            gripper_action.set_action(gripper_cmd)

            # 5 IK 적용 (joint target 설정)
            ik_action.apply()

            # 6 gripper 적용 (joint target 설정)
            gripper_action.apply()

        # ⑤ Sim step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        # marker 업데이트 (EE & goal)
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()