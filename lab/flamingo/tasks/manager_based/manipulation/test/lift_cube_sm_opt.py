# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 32
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os

from isaaclab.app import AppLauncher

import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the task.")
parser.add_argument("--plot", action="store_true", default=True, help="Plot the data.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-A1-IK-Abs-v0-ppo", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")

parser.add_argument("--collect_data", action="store_true", default=False, help="Whether to collect data during simulation.")
parser.add_argument("--cbf_inference", action="store_true", default=True, help="Whether to run CBF inference during simulation.")
# append CO-RL cli arguments
cli_args.add_co_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# update super args
args_cli.video = True
if args_cli.video:
    args_cli.enable_cameras = True
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from collections.abc import Sequence

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlVecEnvWrapper,
)

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from lab.flamingo.tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from lab.flamingo.tasks.manager_based.manipulation.test.utils.utils import *

import numpy as np

import osqp
import scipy.sparse as sp

from lab.flamingo.tasks.manager_based.manipulation.test.dataset.sequencedata_storage import SequenceDataStorage


# initialize warp
wp.init()

N_PTS = 20
EPS_OUT = 1e-3  # obstacle 바깥으로 확실히 빼기 위한 작은 값

# ============================
# QP / CBF helper functions
# ============================

def build_length_cost_matrix(N: int):
    """
    경로 길이(속도) 최소화를 위한 P 행렬
    """
    n_nodes = N
    n_vars = 2 * n_nodes 

    def idx_x(k): return k
    def idx_y(k): return n_nodes + k

    P_data = []
    P_row = []
    P_col = []

    for k in range(n_nodes - 1):
        # X term (Minimize velocity squared)
        for i0, i1 in [(idx_x(k), idx_x(k)), (idx_x(k+1), idx_x(k+1))]:
            P_row.append(i0); P_col.append(i1); P_data.append(1.0)
        P_row.append(idx_x(k)); P_col.append(idx_x(k+1)); P_data.append(-1.0)
        P_row.append(idx_x(k+1)); P_col.append(idx_x(k)); P_data.append(-1.0)

        # Y term
        for i0, i1 in [(idx_y(k), idx_y(k)), (idx_y(k+1), idx_y(k+1))]:
            P_row.append(i0); P_col.append(i1); P_data.append(1.0)
        P_row.append(idx_y(k)); P_col.append(idx_y(k+1)); P_data.append(-1.0)
        P_row.append(idx_y(k+1)); P_col.append(idx_y(k)); P_data.append(-1.0)

    P = sp.coo_matrix((P_data, (P_row, P_col)), shape=(n_vars, n_vars))
    return P.tocsc()

def build_cbf_constraints_rect(
    N: int,
    p_start: np.ndarray,
    p_goal: np.ndarray,
    obs_pos: np.ndarray,
    obs_size_x: float,
    obs_size_y: float,
    margin: float,
):
    """
    [핵심 수정] 장애물과 겹치는 '구간(Time/Index)'에만 제약을 거는 로직 적용
    """
    n_nodes = N
    n_vars = 2 * n_nodes

    def idx_x(k): return k
    def idx_y(k): return n_nodes + k

    xc, yc = obs_pos[0], obs_pos[1]
    hx, hy = 0.5 * obs_size_x, 0.5 * obs_size_y

    # 마진 포함 경계
    x_left   = xc - hx - margin
    x_right  = xc + hx + margin
    y_bottom = yc - hy - margin
    y_top    = yc + hy + margin

    # 1. 회피 방향 결정 (이전과 동일)
    travel_vec = p_goal[0:2] - p_start[0:2]
    mid = 0.5 * (p_start[0:2] + p_goal[0:2])
    dx = mid[0] - xc
    dy = mid[1] - yc
    
    if abs(travel_vec[1]) > abs(travel_vec[0]): # Y축 주행 -> X축(좌/우) 회피
        side = "right" if dx >= 0 else "left"
        main_axis = 1 # Y축이 주행축
    elif abs(travel_vec[0]) > abs(travel_vec[1]): # X축 주행 -> Y축(상/하) 회피
        side = "top" if dy >= 0 else "bottom"
        main_axis = 0 # X축이 주행축
    else:
        side = "right" if dx >= 0 else "left" # Default
        main_axis = 1

    # 2. [NEW] 어떤 인덱스(k)가 장애물 구간에 걸리는지 추정 (Heuristic)
    # 직선 경로로 가정했을 때, 장애물의 유효 범위(Range) 안에 들어오는 k만 찾음
    
    indices_to_constrain = []
    
    # 선형 보간을 통해 각 스텝 k에서의 예상 위치 계산
    for k in range(1, n_nodes - 1):
        alpha = float(k) / float(n_nodes - 1)
        p_guess = (1.0 - alpha) * p_start + alpha * p_goal
        
        # 현재 회피 방향이 좌/우(X축 제약)라면, Y축 위치를 보고 겹치는지 판단
        if side in ["left", "right"]:
            # 예상되는 Y 위치가 장애물의 Y범위 안에 있는가?
            if y_bottom <= p_guess[1] <= y_top:
                indices_to_constrain.append(k)
                
        # 현재 회피 방향이 상/하(Y축 제약)라면, X축 위치를 보고 겹치는지 판단
        elif side in ["top", "bottom"]:
            # 예상되는 X 위치가 장애물의 X범위 안에 있는가?
            if x_left <= p_guess[0] <= x_right:
                indices_to_constrain.append(k)
    
    # [안전 장치] 만약 겹치는 구간이 너무 좁거나 계산상 없더라도, 
    # 장애물 중심에 가장 가까운 포인트 몇 개는 강제로 포함시킴 (스치듯 지나가는 경우 대비)
    if not indices_to_constrain:
        # 중간 지점 인덱스라도 추가
        indices_to_constrain.append(n_nodes // 2)

    # 3. 제약 조건 행렬 구성 (선별된 인덱스에만 적용)
    A_data, A_row, A_col = [], [], []
    l, u = [], []
    row = 0

    for k in indices_to_constrain:
        if side == "right": # x >= x_right
            A_data.append(1.0); A_row.append(row); A_col.append(idx_x(k))
            l.append(x_right); u.append(np.inf)
        elif side == "left": # x <= x_left
            A_data.append(1.0); A_row.append(row); A_col.append(idx_x(k))
            l.append(-np.inf); u.append(x_left)
        elif side == "top": # y >= y_top
            A_data.append(1.0); A_row.append(row); A_col.append(idx_y(k))
            l.append(y_top); u.append(np.inf)
        else: # y <= y_bottom
            A_data.append(1.0); A_row.append(row); A_col.append(idx_y(k))
            l.append(-np.inf); u.append(y_bottom)
        row += 1

    A = sp.coo_matrix((A_data, (A_row, A_col)), shape=(row, n_vars))
    
    # 디버깅을 위해 선택된 인덱스 반환에 포함 가능 (여기선 생략)
    return A.tocsc(), np.array(l), np.array(u), side



class GripperState:
    """States for the gripper."""
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    PLACE_OBJECT = wp.constant(5)
    RELEASE_OBJECT = wp.constant(6)
    BACKTO_ABOVE_OBJECT = wp.constant(7)
    ORIGIN = wp.constant(8)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""
    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(1.5)
    GRASP_OBJECT = wp.constant(0.5)
    LIFT_OBJECT = wp.constant(1.5)
    PLACE_OBJECT = wp.constant(1.0)
    RELEASE_OBJECT = wp.constant(1.0)
    BACKTO_ABOVE_OBJECT = wp.constant(0.5)
    ORIGIN = wp.constant(0.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_origin: wp.array(dtype=wp.transform),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    obstacle_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),

    ctrl_wp: wp.array(dtype=wp.vec3),

    traj_buf: wp.array(dtype=wp.vec3),   # (N*N_PTS,3) flattened
    traj_idx: wp.array(dtype=int),       # (N,) legacy/debug
    traj_s: wp.array(dtype=float),       # (N,) 0..1 progress
    traj_len: wp.array(dtype=float),     # (N,) total length
    lift_speed: float,                   # m/s along path

    position_threshold: float,
    obstacle_size_x: float,
    obstacle_size_y: float,
    traj_margin: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        gripper_state[tid] = GripperState.OPEN

        # QP로 생성된 경로가 있다면 -> traj를 따라감
        if traj_len[tid] > 1e-3:
            # progress update
            s = traj_s[tid]
            s = s + (lift_speed * dt[tid]) / wp.max(traj_len[tid], 1e-3)
            s = wp.clamp(s, 0.0, 1.0)
            traj_s[tid] = s

            # interpolate desired point
            u = s * float(N_PTS - 1)
            i = int(wp.floor(u))
            a = u - float(i)

            if i >= (N_PTS - 1):
                i = N_PTS - 2
                a = 1.0
            if i < 0:
                i = 0
                a = 0.0

            base = tid * N_PTS
            p0 = traj_buf[base + i]
            p1 = traj_buf[base + i + 1]
            p_des = (1.0 - a) * p0 + a * p1

            # offset상 목표는 offset*object_pose 이므로 회전은 그대로 사용
            q_des = wp.transform_get_rotation(object_pose[tid])
            des_ee_pose[tid] = wp.transform(p_des, q_des)

            ee_p = wp.transform_get_translation(ee_pose[tid])
            p_goal = traj_buf[base + (N_PTS - 1)]
            reach_goal = distance_below_threshold(ee_p, p_goal, position_threshold / 4.0)

            if (s >= 1.0 - 1e-5) and reach_goal:
                if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                    sm_state[tid] = PickSmState.APPROACH_OBJECT
                    sm_wait_time[tid] = 0.0
        else:
            # 아직 경로가 없다면, 기존 방식으로 fallback
            des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
            if distance_below_threshold(
                wp.transform_get_translation(ee_pose[tid]),
                wp.transform_get_translation(des_ee_pose[tid]),
                position_threshold,
            ):
                if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                    sm_state[tid] = PickSmState.APPROACH_OBJECT
                    sm_wait_time[tid] = 0.0


    elif state == PickSmState.APPROACH_OBJECT:
            # 기존: 목표 위치로 즉시 이동 (삭제 혹은 주석 처리)
            # des_ee_pose[tid] = object_pose[tid]
            # gripper_state[tid] = GripperState.OPEN
            
            # 수정: 경로(Trajectory)를 따라가도록 변경
            gripper_state[tid] = GripperState.OPEN

            # 경로가 설정되지 않았을 경우(초기 진입 시), 일단 제자리 유지
            if traj_len[tid] <= 1e-3:
                des_ee_pose[tid] = ee_pose[tid]
                return

            # --- 아래는 LIFT_OBJECT의 로직과 동일 ---
            s = traj_s[tid]
            # 내려가는 속도 조절이 필요하면 lift_speed를 조절하거나 별도 변수 사용
            s = s + (lift_speed * dt[tid]) / wp.max(traj_len[tid], 1e-3)
            s = wp.clamp(s, 0.0, 1.0)
            traj_s[tid] = s

            u = s * float(N_PTS - 1)
            i = int(wp.floor(u))
            a = u - float(i)

            if i >= (N_PTS - 1):
                i = N_PTS - 2
                a = 1.0
            if i < 0:
                i = 0
                a = 0.0

            base = tid * N_PTS
            p0 = traj_buf[base + i]
            p1 = traj_buf[base + i + 1]
            p_des = (1.0 - a) * p0 + a * p1



            # 회전은 물체의 회전을 그대로 따름
            q_des = wp.transform_get_rotation(object_pose[tid])
            des_ee_pose[tid] = wp.transform(p_des, q_des)

            # 목표 도달 확인
            ee_p = wp.transform_get_translation(ee_pose[tid])
            p_goal = traj_buf[base + (N_PTS - 1)]
            
            # 정확한 도달을 위해 threshold 체크
            if (s >= 1.0 - 1e-5) and distance_below_threshold(ee_p, p_goal, position_threshold):
                if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                    sm_state[tid] = PickSmState.GRASP_OBJECT
                    sm_wait_time[tid] = 0.0

    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
                # 여기서는 단순히 상태만 변경.
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0

                # 경로 생성은 PYTHON(PickAndLiftSm.compute) 쪽에서 처리.
                traj_s[tid] = 0.0
                traj_len[tid] = 1e-3


    elif state == PickSmState.LIFT_OBJECT:
        gripper_state[tid] = GripperState.CLOSE


        # QP 경로가 아직 세팅되지 않은 경우: 일단 제자리 유지
        if traj_len[tid] <= 1e-3:
            des_ee_pose[tid] = ee_pose[tid]
            return

        # progress update
        s = traj_s[tid]
        s = s + (lift_speed * dt[tid]) / wp.max(traj_len[tid], 1e-3)
        s = wp.clamp(s, 0.0, 1.0)
        traj_s[tid] = s

        # interpolate desired point
        u = s * float(N_PTS - 1)
        i = int(wp.floor(u))
        a = u - float(i)

        if i >= (N_PTS - 1):
            i = N_PTS - 2
            a = 1.0
        if i < 0:
            i = 0
            a = 0.0

        base = tid * N_PTS
        p0 = traj_buf[base + i]
        p1 = traj_buf[base + i + 1]
        p_des = (1.0 - a) * p0 + a * p1

        q_des = wp.transform_get_rotation(des_object_pose[tid])
        des_ee_pose[tid] = wp.transform(p_des, q_des)

        # ✅ 추가: 실제 EE가 "끝점" 근처인지 확인
        ee_p = wp.transform_get_translation(ee_pose[tid])
        p_goal = traj_buf[base + (N_PTS - 1)]

        # LIFT에서만 더 빡세게 잡고 싶으면 /4 같은 스케일을 유지
        reach_goal = distance_below_threshold(ee_p, p_goal, position_threshold / 4.0)

        # ✅ done when: s 거의 1 + 실제 도착 + wait time
        if (s >= 1.0 - 1e-5) and reach_goal:
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.RELEASE_OBJECT:
                sm_state[tid] = PickSmState.BACKTO_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.BACKTO_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.BACKTO_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.ORIGIN
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.ORIGIN:
        des_ee_pose[tid] = ee_origin[tid]
        gripper_state[tid] = GripperState.OPEN
        sm_state[tid] = PickSmState.ORIGIN
        sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.ee_origin = None

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

        # debug ctrl point
        self.ctrl_wp = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_wp_wp = wp.from_torch(self.ctrl_wp, wp.vec3)

        # traj buffer
        self.traj_buf = torch.zeros((self.num_envs * N_PTS, 3), device=self.device)
        self.traj_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.traj_buf_wp = wp.from_torch(self.traj_buf, wp.vec3)
        self.traj_idx_wp = wp.from_torch(self.traj_idx, wp.int32)

        # NEW: smooth progression buffers (warp)
        self.traj_s = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.traj_len = torch.full((self.num_envs,), 1e-3, device=self.device, dtype=torch.float32)
        self.traj_s_wp = wp.from_torch(self.traj_s, wp.float32)
        self.traj_len_wp = wp.from_torch(self.traj_len, wp.float32)

        # speed along path
        self.lift_speed = 0.25

        # obstacle geometry
        self.obstacle_size_x = 0.45
        self.obstacle_size_y = 0.035
        self.traj_margin = 0.1

        # === QP 관련 사전 계산 ===
        self.qp_P = build_length_cost_matrix(N_PTS)
        self.qp_P = (self.qp_P + self.qp_P.T) * 0.5  # ensure symmetric
        self.qp_q = np.zeros(2 * N_PTS, dtype=float)
        self.qp_solver = None  # 필요시 재사용 가능 (단, obstacle/CBF가 바뀌면 다시 세팅)

    def _plan_path_qp_single(
        self,
        p_start: np.ndarray,
        p_goal: np.ndarray,
        obs_pos: np.ndarray,
    ) -> np.ndarray:
        """
        QP + CBF 제약으로 2D 경로를 생성.
        실패 시에는 직선 보간으로 fallback.
        return: (N_PTS, 3) world frame 경로 (z는 선형 보간)
        """
        N = N_PTS
        n_vars = 2 * N

        def idx_x(k): return k
        def idx_y(k): return N + k

        # cost
        P = self.qp_P
        q = self.qp_q.copy()

        # CBF 제약 (장애물 회피)
        A_cbf, l_cbf, u_cbf, _ = build_cbf_constraints_rect(
            N,
            p_start,
            p_goal,
            obs_pos,
            self.obstacle_size_x,
            self.obstacle_size_y,
            self.traj_margin,
        )

        # Endpoint equality 제약: x_0, y_0, x_{N-1}, y_{N-1}
        A_eq_data = []
        A_eq_row = []
        A_eq_col = []
        beq = []

        row = 0

        # x_0 = p_start[0]
        A_eq_data.append(1.0)
        A_eq_row.append(row)
        A_eq_col.append(idx_x(0))
        beq.append(p_start[0])
        row += 1

        # y_0 = p_start[1]
        A_eq_data.append(1.0)
        A_eq_row.append(row)
        A_eq_col.append(idx_y(0))
        beq.append(p_start[1])
        row += 1

        # x_{N-1} = p_goal[0]
        A_eq_data.append(1.0)
        A_eq_row.append(row)
        A_eq_col.append(idx_x(N-1))
        beq.append(p_goal[0])
        row += 1

        # y_{N-1} = p_goal[1]
        A_eq_data.append(1.0)
        A_eq_row.append(row)
        A_eq_col.append(idx_y(N-1))
        beq.append(p_goal[1])
        row += 1

        A_eq = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(row, n_vars)).tocsc()
        beq = np.array(beq, dtype=float)

        # OSQP 형식: l <= A z <= u
        # eq 제약은 l_eq = u_eq = beq 로 만들어서 stack
        A = sp.vstack([A_cbf, A_eq], format="csc")
        l = np.concatenate([l_cbf, beq])
        u = np.concatenate([u_cbf, beq])

        # solver
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        res = prob.solve()

        if res.info.status.lower() not in ["solved", "solved_inaccurate"]:
            print("[QP] infeasible or no solution. status =", res.info.status)
            # fallback: 직선 보간
            ts = np.linspace(0.0, 1.0, N)
            xs = (1 - ts) * p_start[0] + ts * p_goal[0]
            ys = (1 - ts) * p_start[1] + ts * p_goal[1]
        else:
            z = res.x
            xs = z[0:N]
            ys = z[N:2*N]

        # z좌표는 단순 선형 보간
        ts = np.linspace(0.0, 1.0, N)
        zs = (1 - ts) * p_start[2] + ts * p_goal[2]

        path = np.stack([xs, ys, zs], axis=-1)
        return path

    def _plan_and_write_traj(
        self,
        env_idx: int,
        p_start: torch.Tensor,
        p_goal: torch.Tensor,
        obs_pos: torch.Tensor,
    ):
        """
        단일 env에 대해 QP로 경로 생성 후 self.traj_buf, self.traj_len 에 기록.
        """
        # torch -> numpy (CPU)
        ps = p_start.detach().cpu().numpy()
        pg = p_goal.detach().cpu().numpy()
        po = obs_pos.detach().cpu().numpy()

        path = self._plan_path_qp_single(ps, pg, po)   # (N_PTS, 3)

        # path 길이 계산
        diffs = np.diff(path, axis=0)
        L = np.sum(np.linalg.norm(diffs, axis=1))
        if L < 1e-3:
            L = 1e-3

        # 다시 torch 텐서에 쓰기
        base = env_idx * N_PTS
        self.traj_buf[base:base+N_PTS, :] = torch.from_numpy(path).to(self.device)
        self.traj_s[env_idx] = 0.0
        self.traj_len[env_idx] = float(L)


    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        self.ctrl_wp[env_ids] = 0.0
        self.traj_idx[env_ids] = 0
        self.traj_s[env_ids] = 0.0
        self.traj_len[env_ids] = 1e-3

        if isinstance(env_ids, slice):
            self.traj_buf[:] = 0.0
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
            for eid in ids.tolist():
                self.traj_buf[eid * N_PTS : (eid + 1) * N_PTS] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor, obstacle_pose: torch.Tensor) -> torch.Tensor:
         # 이전 state 저장 (REST→APPROACH, GRASP→LIFT 전이를 감지하기 위함)
        prev_state = self.sm_state.clone()
        # (w,x,y,z) -> (x,y,z,x,y,z,w)
        ee_pose         = ee_pose[:,         [0, 1, 2, 4, 5, 6, 3]]
        object_pose     = object_pose[:,     [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        obstacle_pose   = obstacle_pose[:,   [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp         = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp     = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        obstacle_pose_wp   = wp.from_torch(obstacle_pose.contiguous(), wp.transform)

        if self.ee_origin is None:
            self.ee_origin = wp.from_torch(ee_pose.contiguous(), wp.transform)

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.ee_origin,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                obstacle_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.ctrl_wp_wp,
                self.traj_buf_wp,
                self.traj_idx_wp,
                self.traj_s_wp,
                self.traj_len_wp,
                float(self.lift_speed),
                self.position_threshold,
                self.obstacle_size_x,
                self.obstacle_size_y,
                self.traj_margin,
            ],
            device=self.device,
        )

        # === 상태 전이 후 QP 경로 생성 ===

        # 1) REST -> APPROACH_ABOVE_OBJECT 사이 경로 생성
        mask_approach = (prev_state == int(PickSmState.REST)) & (self.sm_state == int(PickSmState.APPROACH_ABOVE_OBJECT))
        idxs_approach = mask_approach.nonzero(as_tuple=False).view(-1)
        if idxs_approach.numel() > 0:
            # ee_pose: (env, 7)  [x,y,z,qx,qy,qz,qw]
            # object_pose: (env,7), offset*object_pose 가 목표
            # 여기서는 "현재 EE 위치 -> offset*object_pose 위치"로 경로 생성
            with torch.no_grad():
                for eid in idxs_approach.tolist():
                    p_start = ee_pose[eid, 0:3]
                    # offset은 world 기준이 아니라 현재 offset transform 으로 EE 목표를 정의했으므로,
                    # 여기서는 간단하게 "object_pose + offset_z" 정도로 목표를 둠
                    p_goal = object_pose[eid, 0:3].clone()
                    p_goal[2] += self.offset[eid, 2]  # z offset 적용

                    obs_pos = obstacle_pose[eid, 0:3]
                    self._plan_and_write_traj(eid, p_start, p_goal, obs_pos)

        # [추가됨] 1.5) APPROACH_ABOVE_OBJECT -> APPROACH_OBJECT (직선 하강 경로 생성)
        mask_descend = (prev_state == int(PickSmState.APPROACH_ABOVE_OBJECT)) & (self.sm_state == int(PickSmState.APPROACH_OBJECT))
        idxs_descend = mask_descend.nonzero(as_tuple=False).view(-1)
        
        if idxs_descend.numel() > 0:
            with torch.no_grad():
                for eid in idxs_descend.tolist():
                    # 시작점: 현재 EE 위치 (또는 object_pose + offset)
                    # 끝점: 물체 위치 (object_pose)
                    
                    # 더 엄격한 수직 하강을 원한다면 p_start의 x,y를 p_goal의 x,y로 강제할 수도 있음
                    # 여기서는 부드러운 연결을 위해 현재 위치에서 시작
                    p_start = ee_pose[eid, 0:3] 
                    p_goal  = object_pose[eid, 0:3]
                    
                    # 직선 경로 생성 (QP 불필요, 단순 선형 보간)
                    # (N_PTS, 3) 생성
                    alpha = torch.linspace(0, 1, N_PTS, device=self.device).unsqueeze(-1) # (N, 1)
                    path = (1 - alpha) * p_start + alpha * p_goal
                    
                    # 경로 길이 계산
                    diffs = torch.norm(path[1:] - path[:-1], dim=1).sum()
                    L = float(diffs)
                    if L < 1e-3: L = 1e-3
                    
                    # Buffer에 쓰기
                    base = eid * N_PTS
                    self.traj_buf[base:base+N_PTS, :] = path
                    self.traj_s[eid] = 0.0
                    self.traj_len[eid] = L

        # 2) GRASP_OBJECT -> LIFT_OBJECT 사이 경로 생성
        mask_lift = (prev_state == int(PickSmState.GRASP_OBJECT)) & (self.sm_state == int(PickSmState.LIFT_OBJECT))
        idxs_lift = mask_lift.nonzero(as_tuple=False).view(-1)
        if idxs_lift.numel() > 0:
            with torch.no_grad():
                for eid in idxs_lift.tolist():
                    # object 를 잡은 상태이므로, object 위치 -> des_object 위치로 경로 생성
                    p_start = object_pose[eid, 0:3]
                    p_goal  = des_object_pose[eid, 0:3]
                    obs_pos = obstacle_pose[eid, 0:3]
                    self._plan_and_write_traj(eid, p_start, p_goal, obs_pos)


        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    env_cfg: LiftEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks

    env_cfg.episode_length_s = 12.0
    env_cfg.terminations.illegal_contact.params["sensor_cfg"].body_names = []

    env = gym.make("Isaac-Lift-Cube-A1-IK-Abs-v0-ppo", cfg=env_cfg)
    env = CoRlVecEnvWrapper(env, agent_cfg)
    env.reset()

    traj_cfg_base = FRAME_MARKER_CFG.copy()
    traj_cfg_base.prim_path = "/Visuals/Traj/pts"
    traj_cfg_base.markers["frame"].scale = (0.02, 0.02, 0.02)
    traj_vis = VisualizationMarkers(traj_cfg_base)

    qI = torch.tensor([1, 0, 0, 0], device=env.unwrapped.device, dtype=env.unwrapped.scene.env_origins.dtype).view(1, 4).repeat(N_PTS, 1)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 2] = 1.0

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.015
    )

    if args_cli.plot:
        tv_rgb_viz   = CameraViz(H=224, W=224, title="camera rgb",   env_id=0, mode="rgb")
        tv_depth_viz = CameraViz(H=224, W=224, title="camera depth", env_id=0, mode="depth")
        tv_depth_viz.set_depth_display(auto_scale=True, near=0.1, far=4.0, invert=False)

        ee_rgb_viz   = CameraViz(H=224, W=224, title="ee camera rgb",   env_id=0, mode="rgb")
        ee_depth_viz = CameraViz(H=224, W=224, title="ee camera depth", env_id=0, mode="depth")
        ee_depth_viz.set_depth_display(auto_scale=True, near=0.1, far=4.0, invert=False)

        if args_cli.cbf_inference:
            cbf_viz = CbfLineViz(title="CBF Monitor", env_id=0, window_len=200, y_range=(-0.2, 2.0))

    if args_cli.collect_data:
        from datetime import datetime

        root_dir = os.path.dirname(os.path.abspath(__file__))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # dataset/logs/lift_pick_and_lift_sm_YYYYMMDD_HHMMSS
        zarr_dir = os.path.join(root_dir, "dataset", "logs")
        zarr_name = f"lift_pick_and_lift_sm_{timestamp}"
        zarr_path = os.path.join(zarr_dir, zarr_name)

        print(f"Recording data to: {zarr_path}")
        
        cam_cfg = {
            "tv_cam":  {"rgb": False, "depth": True, "shape": (224, 224), "depth_dtype": "float16"},
            "ee_cam":  {"rgb": True, "depth": False, "shape": (224, 224),   "depth_dtype": "float16"},
        }

        storage = SequenceDataStorage(
            num_envs=env.unwrapped.num_envs,
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            zarr_path=zarr_path,
            camera_config=cam_cfg,
            include_ee=True,
            include_cbf=True,
            flush_every_episodes=128,
        )

    if args_cli.cbf_inference:
        from cbf.network.model import MultiModalTransformer

        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = "logs/lift_pick_and_lift_sm_20260215_230551/best.pt" 
        
        ckpt = os.path.join(script_dir, "cbf", ckpt_path)
        # -------------------------------------------------------------------------

        inference_cam_cfg = {
            "tv_cam": {"rgb": False, "depth": True, "shape": (224, 224)},
            "ee_cam": {"rgb": True,  "depth": False, "shape": (224, 224)},
        }
        
        # 전처리 파라미터 (학습 arguments와 동일하게 설정)
        DEPTH_CLIP = (0.0, 4.0)
        DEPTH_SCALE = 4.0
        DEPTH_DOWNSAMPLE = 1
        
        # 모델 입력 Config 생성 (train.py 로직과 동일)
        model_input_configs = {}
        for cam_name, cfg in inference_cam_cfg.items():
            orig_h, orig_w = cfg.get("shape", (224, 224))
            
            if cfg.get("rgb"):
                key = f"{cam_name}_rgb"
                model_input_configs[key] = {"shape": (orig_h, orig_w), "ch": 3}
            
            if cfg.get("depth"):
                key = f"{cam_name}_depth"
                # 다운샘플링 적용된 크기 계산
                ds = DEPTH_DOWNSAMPLE
                new_h = (orig_h + ds - 1) // ds
                new_w = (orig_w + ds - 1) // ds
                model_input_configs[key] = {"shape": (new_h, new_w), "ch": 1}

        # 모델 초기화
        cbf_model = MultiModalTransformer(
            input_configs=model_input_configs,
            d_model=256,
            patch=16,
            n_layers=4,
            n_heads=8
        ).to(env.unwrapped.device)

        if os.path.exists(ckpt):
            print(f"[INFO] Loading model from: {ckpt}")
            checkpoint = torch.load(ckpt, map_location=env.unwrapped.device)
            cbf_model.load_state_dict(checkpoint["model"])
            cbf_model.train() # 추론 모드
        else:
            print(f"[WARN] Checkpoint NOT found at: {ckpt}")
            print("       Model will output random garbage.")

    observations, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            prev_obs = observations
            # ================================= DATA ================================= #
            observations, rewards, dones, extras  = env.step(actions)
            truncateds = extras["time_outs"]

            cbf = extras["observations"]["obs_info"]

            tv_camera_rgb = env.unwrapped.scene["tv_camera"].data.output["rgb"]
            tv_camera_depth = env.unwrapped.scene["tv_camera"].data.output["distance_to_image_plane"]

            ee_camera_rgb = env.unwrapped.scene["ee_camera"].data.output["rgb"]
            ee_camera_depth = env.unwrapped.scene["ee_camera"].data.output["distance_to_image_plane"]
            # ================================= DATA ================================= #

            origins = env.unwrapped.scene.env_origins

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            tcp_rest_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)

            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - origins

            obstacle_data: RigidObjectData = env.unwrapped.scene["screen"].data
            obstacle_position = obstacle_data.root_pos_w - origins
            obstacle_quat = obstacle_data.root_quat_w

            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # =========================== CAMERA DATA =========================== #
            cam_data_input = {
                "tv_cam": {"depth": tv_camera_depth},
                "ee_cam": {"rgb": ee_camera_rgb},
            }
            
            # =========================== DATASET COLLECTION =========================== #
            if args_cli.collect_data:
                storage.add_step_data(
                    prev_obs,
                    actions,
                    rewards,
                    dones,
                    truncateds,
                    observations,
                    cbf=cbf if storage.include_cbf else None,
                    camera_data=cam_data_input,
                    ee_pos=tcp_rest_pose if storage.include_ee else None,
                )

                if storage.get_total_flush_count() >= 10: # Stop after 10*128 = 1280 episodes
                    print(f"[INFO] Reached {storage.get_total_flush_count() * storage.flush_every_episodes} episodes. Exiting...")
                    break
            # =========================== DATASET COLLECTION =========================== #
            
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
                torch.cat([obstacle_position, obstacle_quat], dim=-1),
            )            

            path_mask = (pick_sm.sm_state == int(PickSmState.LIFT_OBJECT)) | \
                        (pick_sm.sm_state == int(PickSmState.APPROACH_ABOVE_OBJECT)) | \
                        (pick_sm.sm_state == int(PickSmState.APPROACH_OBJECT))
            traj_vis_mask = path_mask.nonzero(as_tuple=False).squeeze(-1)

            # ==============================모델 추론 (Inference)===================================
            if args_cli.cbf_inference:
                target_env_id = 0
                model_inputs = {}

                # 1. [동적 처리 핵심] Raw Data Mapping
                # 카메라 이름과 모달리티를 키로 사용하여 실제 데이터에 접근할 수 있게 맵핑합니다.
                raw_data_map = {
                    "tv_cam": {"rgb": tv_camera_rgb, "depth": tv_camera_depth},
                    "ee_cam": {"rgb": ee_camera_rgb, "depth": ee_camera_depth},
                }

                # 2. Config 루프를 돌며 전처리 수행
                for cam_name, cfg in inference_cam_cfg.items():
                    # 해당 카메라 데이터가 map에 없으면 스킵 (안전장치)
                    if cam_name not in raw_data_map:
                        continue

                    # (A) Depth 처리
                    if cfg.get("depth"):
                        # Data Fetch
                        raw_depth = raw_data_map[cam_name]["depth"][target_env_id].clone()
                        
                        # Dimension Check: (H, W) -> (H, W, 1)
                        if raw_depth.ndim == 2:
                            raw_depth = raw_depth.unsqueeze(-1)
                        
                        # Permute: (H, W, 1) -> (1, H, W)
                        raw_depth = raw_depth.permute(2, 0, 1)
                        
                        # Preprocessing (Clip & Scale)
                        if DEPTH_CLIP:
                            raw_depth = torch.clamp(raw_depth, min=DEPTH_CLIP[0], max=DEPTH_CLIP[1])
                        if DEPTH_SCALE:
                            raw_depth = raw_depth / DEPTH_SCALE
                        
                        # Downsample
                        if DEPTH_DOWNSAMPLE > 1:
                            raw_depth = raw_depth[:, ::DEPTH_DOWNSAMPLE, ::DEPTH_DOWNSAMPLE]
                        
                        # Result Key: "tv_cam_depth" etc.
                        key = f"{cam_name}_depth"
                        model_inputs[key] = raw_depth.unsqueeze(0).to(env.unwrapped.device)

                    # (B) RGB 처리
                    if cfg.get("rgb"):
                        # Data Fetch
                        raw_rgb = raw_data_map[cam_name]["rgb"][target_env_id].clone()
                        
                        # Permute: (H, W, 3) -> (3, H, W)
                        raw_rgb = raw_rgb.permute(2, 0, 1)
                        
                        # Normalize
                        if raw_rgb.dtype == torch.uint8:
                            raw_rgb = raw_rgb.float() / 255.0
                        elif raw_rgb.max() > 1.0:
                            raw_rgb = raw_rgb / 255.0
                        
                        # Result Key: "ee_cam_rgb" etc.
                        key = f"{cam_name}_rgb"
                        model_inputs[key] = raw_rgb.unsqueeze(0).to(env.unwrapped.device)

                # 3. EE Position (Vector Data)
                model_ee = tcp_rest_pose[target_env_id].unsqueeze(0).to(env.unwrapped.device)

                # 4. 모델 예측
                pred_cbf_tensor = cbf_model(model_inputs, model_ee)
                
                pred_val = pred_cbf_tensor.item()
                true_val = cbf[target_env_id].item()
            # =========================================================================

            # ==================== 시각화 업데이트 ====================
            if args_cli.plot:
                tv_rgb_viz.update(tv_camera_rgb)
                tv_depth_viz.update(tv_camera_depth)
                ee_rgb_viz.update(ee_camera_rgb)
                ee_depth_viz.update(ee_camera_depth)
                if args_cli.cbf_inference:
                    cbf_viz.update(true_val, pred_val)
            # ======================================================

            if traj_vis_mask.numel() > 0:
                flag = False if args_cli.collect_data else False
                traj_vis.set_visibility(flag)

                traj_view = pick_sm.traj_buf.view(env.unwrapped.num_envs, N_PTS, 3)  # (N,n,3)
                pts_all = traj_view[traj_vis_mask] + origins[traj_vis_mask].unsqueeze(1)
                pts_flat = pts_all.reshape(-1, 3)
                qI_flat = qI.repeat(traj_vis_mask.numel(), 1)
                traj_vis.visualize(pts_flat, qI_flat)
            else:
                traj_vis.set_visibility(False)

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    env.close()

    if args_cli.collect_data:
        storage.flush_to_zarr()


if __name__ == "__main__":
    main()
    simulation_app.close()