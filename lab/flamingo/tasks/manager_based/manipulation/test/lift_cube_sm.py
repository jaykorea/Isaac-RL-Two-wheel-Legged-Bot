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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = False
# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaaclab_tasks  # noqa: F401

from lab.flamingo.tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from lab.flamingo.tasks.manager_based.manipulation.test.utils import *

import matplotlib.pyplot as plt
import numpy as np


# initialize warp
wp.init()

N_PTS = 40
EPS_OUT = 1e-3  # obstacle 바깥으로 확실히 빼기 위한 작은 값


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
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.5)
    GRASP_OBJECT = wp.constant(0.5)
    LIFT_OBJECT = wp.constant(1.0)
    PLACE_OBJECT = wp.constant(1.0)
    RELEASE_OBJECT = wp.constant(1.0)
    BACKTO_ABOVE_OBJECT = wp.constant(0.5)
    ORIGIN = wp.constant(0.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.func
def segment_intersects_aabb_xy(p0: wp.vec3, p1: wp.vec3, hx: float, hy: float) -> bool:
    # AABB centered at origin in XY: x in [-hx,hx], y in [-hy,hy]
    tmin = 0.0
    tmax = 1.0
    eps = 1e-8

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    # X slab
    if wp.abs(dx) < eps:
        if wp.abs(p0[0]) > hx:
            return False
    else:
        inv = 1.0 / dx
        tx1 = (-hx - p0[0]) * inv
        tx2 = ( hx - p0[0]) * inv
        t_enter = wp.min(tx1, tx2)
        t_exit  = wp.max(tx1, tx2)
        tmin = wp.max(tmin, t_enter)
        tmax = wp.min(tmax, t_exit)
        if tmin > tmax:
            return False

    # Y slab
    if wp.abs(dy) < eps:
        if wp.abs(p0[1]) > hy:
            return False
    else:
        inv = 1.0 / dy
        ty1 = (-hy - p0[1]) * inv
        ty2 = ( hy - p0[1]) * inv
        t_enter = wp.min(ty1, ty2)
        t_exit  = wp.max(ty1, ty2)
        tmin = wp.max(tmin, t_enter)
        tmax = wp.min(tmax, t_exit)
        if tmin > tmax:
            return False

    return True


@wp.func
def sign_or_pos(v: float) -> float:
    # v==0이면 +로
    if v >= 0.0:
        return 1.0
    else:
        return -1.0


@wp.func
def quad_bezier(p0: wp.vec3, pc: wp.vec3, p1: wp.vec3, t: float) -> wp.vec3:
    u = 1.0 - t
    return (u*u) * p0 + (2.0*u*t) * pc + (t*t) * p1


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
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        # ✅ 다시 추가: grasp 위치에 들어와 있고 + 충분히 대기했을 때만 lift 전환
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0

                p_start = wp.transform_get_translation(object_pose[tid])
                p_goal  = wp.transform_get_translation(des_object_pose[tid])

                # obstacle local frame
                T_obs = obstacle_pose[tid]
                T_inv = wp.transform_inverse(T_obs)

                p0l = wp.transform_point(T_inv, p_start)
                p1l = wp.transform_point(T_inv, p_goal)

                hx = 0.5 * obstacle_size_x + traj_margin
                hy = 0.5 * obstacle_size_y + traj_margin

                # straight가 박스를 관통하면 detour 필요
                need_detour = segment_intersects_aabb_xy(p0l, p1l, hx, hy)

                # 기본은 straight
                A = 0.5 * (p0l + p1l)
                B = A
                best_cost = float(1e20)
                found = False

                if need_detour:
                    # p0/p1가 박스 두께 안에 걸려있으면 바깥으로 밀기(안전)
                    y0 = p0l[1]
                    y1 = p1l[1]
                    if wp.abs(y0) < (hy + EPS_OUT):
                        y0 = sign_or_pos(y0) * (hy + EPS_OUT)
                    if wp.abs(y1) < (hy + EPS_OUT):
                        y1 = sign_or_pos(y1) * (hy + EPS_OUT)

                    x0 = p0l[0]
                    x1 = p1l[0]
                    if wp.abs(x0) < (hx + EPS_OUT):
                        x0 = sign_or_pos(x0) * (hx + EPS_OUT)
                    if wp.abs(x1) < (hx + EPS_OUT):
                        x1 = sign_or_pos(x1) * (hx + EPS_OUT)

                    # +x
                    ex = hx + EPS_OUT
                    A1 = wp.vec3(ex, y0, 0.5 * (p0l[2] + p1l[2]))
                    B1 = wp.vec3(ex, y1, 0.5 * (p0l[2] + p1l[2]))
                    ok = (not segment_intersects_aabb_xy(p0l, A1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(A1, B1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(B1, p1l, hx, hy))
                    if ok:
                        c = wp.length(p0l - A1) + wp.length(A1 - B1) + wp.length(B1 - p1l)
                        if c < best_cost:
                            best_cost = c
                            A = A1
                            B = B1
                            found = True

                    # -x
                    ex = -(hx + EPS_OUT)
                    A1 = wp.vec3(ex, y0, 0.5 * (p0l[2] + p1l[2]))
                    B1 = wp.vec3(ex, y1, 0.5 * (p0l[2] + p1l[2]))
                    ok = (not segment_intersects_aabb_xy(p0l, A1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(A1, B1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(B1, p1l, hx, hy))
                    if ok:
                        c = wp.length(p0l - A1) + wp.length(A1 - B1) + wp.length(B1 - p1l)
                        if c < best_cost:
                            best_cost = c
                            A = A1
                            B = B1
                            found = True

                    # +y
                    ey = hy + EPS_OUT
                    A1 = wp.vec3(x0, ey, 0.5 * (p0l[2] + p1l[2]))
                    B1 = wp.vec3(x1, ey, 0.5 * (p0l[2] + p1l[2]))
                    ok = (not segment_intersects_aabb_xy(p0l, A1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(A1, B1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(B1, p1l, hx, hy))
                    if ok:
                        c = wp.length(p0l - A1) + wp.length(A1 - B1) + wp.length(B1 - p1l)
                        if c < best_cost:
                            best_cost = c
                            A = A1
                            B = B1
                            found = True

                    # -y
                    ey = -(hy + EPS_OUT)
                    A1 = wp.vec3(x0, ey, 0.5 * (p0l[2] + p1l[2]))
                    B1 = wp.vec3(x1, ey, 0.5 * (p0l[2] + p1l[2]))
                    ok = (not segment_intersects_aabb_xy(p0l, A1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(A1, B1, hx, hy)) and \
                         (not segment_intersects_aabb_xy(B1, p1l, hx, hy))
                    if ok:
                        c = wp.length(p0l - A1) + wp.length(A1 - B1) + wp.length(B1 - p1l)
                        if c < best_cost:
                            best_cost = c
                            A = A1
                            B = B1
                            found = True

                    # 전부 실패하면(거의 없음) 그래도 +x로 강제
                    if not found:
                        ex = hx + EPS_OUT
                        A = wp.vec3(ex, y0, 0.5 * (p0l[2] + p1l[2]))
                        B = wp.vec3(ex, y1, 0.5 * (p0l[2] + p1l[2]))

                # local -> world (A,B)
                Aw = wp.transform_point(T_obs, A)
                Bw = wp.transform_point(T_obs, B)

                # ─────────────────────────────────────────────
                # smooth traj_buf: line + bezier(fillet) + line + bezier(fillet) + line
                # p_start -> P1 ->(bezier via Aw)-> Q1 -> P2 ->(bezier via Bw)-> Q2 -> p_goal
                # ─────────────────────────────────────────────
                traj_idx[tid] = 0  # legacy/debug
                traj_s[tid] = 0.0  # progress reset
                base = tid * N_PTS

                FILLET_FRAC = 2.0
                FILLET_MAX  = 0.1
                eps = 1e-8

                v01 = Aw - p_start
                v12 = Bw - Aw
                v23 = p_goal - Bw

                l01 = wp.length(v01)
                l12 = wp.length(v12)
                l23 = wp.length(v23)

                # degenerate 방어
                if (l01 < 1e-6) or (l12 < 1e-6) or (l23 < 1e-6):
                    # fallback: 그냥 기존 3-seg로
                    s1 = N_PTS // 3
                    s2 = (2 * N_PTS) // 3
                    for j in range(N_PTS):
                        if j <= s1:
                            t = float(j) / float(wp.max(1, s1))
                            pt = p_start + t * (Aw - p_start)
                        elif j <= s2:
                            t = float(j - s1) / float(wp.max(1, s2 - s1))
                            pt = Aw + t * (Bw - Aw)
                        else:
                            t = float(j - s2) / float(wp.max(1, (N_PTS - 1) - s2))
                            pt = Bw + t * (p_goal - Bw)
                        traj_buf[base + j] = pt
                else:
                    u01 = v01 / (l01 + eps)
                    u12 = v12 / (l12 + eps)
                    u23 = v23 / (l23 + eps)

                    r1 = wp.min(FILLET_MAX, FILLET_FRAC * wp.min(l01, l12))
                    r2 = wp.min(FILLET_MAX, FILLET_FRAC * wp.min(l12, l23))

                    # 코너 주변에서 직선 위 점을 잘라냄 (안전: 항상 기존 직선 구간 위에 있음)
                    P1 = Aw - r1 * u01
                    Q1 = Aw + r1 * u12
                    P2 = Bw - r2 * u12
                    Q2 = Bw + r2 * u23

                    # 각 구간 길이(Bezier는 근사치로 chord+handles)
                    L0 = wp.length(p_start - P1)
                    L1 = wp.length(P1 - Aw) + wp.length(Aw - Q1)
                    L2 = wp.length(Q1 - P2)
                    L3 = wp.length(P2 - Bw) + wp.length(Bw - Q2)
                    L4 = wp.length(Q2 - p_goal)
                    Ls = L0 + L1 + L2 + L3 + L4 + eps

                    # 누적 비율
                    t0 = L0 / Ls
                    t1 = L1 / Ls
                    t2 = L2 / Ls
                    t3 = L3 / Ls

                    c0 = t0
                    c1 = c0 + t1
                    c2 = c1 + t2
                    c3 = c2 + t3

                    for j in range(N_PTS):
                        s = float(j) / float(N_PTS - 1)

                        if s < c0:
                            tt = s / wp.max(eps, c0)
                            pt = p_start + tt * (P1 - p_start)

                        elif s < c1:
                            tt = (s - c0) / wp.max(eps, (c1 - c0))
                            pt = quad_bezier(P1, Aw, Q1, tt)

                        elif s < c2:
                            tt = (s - c1) / wp.max(eps, (c2 - c1))
                            pt = Q1 + tt * (P2 - Q1)

                        elif s < c3:
                            tt = (s - c2) / wp.max(eps, (c3 - c2))
                            pt = quad_bezier(P2, Bw, Q2, tt)

                        else:
                            tt = (s - c3) / wp.max(eps, (1.0 - c3))
                            pt = Q2 + tt * (p_goal - Q2)

                        traj_buf[base + j] = pt

                # --- compute total trajectory length for time-based progression ---
                L = float(0.0)
                for j in range(wp.static(N_PTS - 1)):
                    dp = traj_buf[base + j + 1] - traj_buf[base + j]
                    L = L + wp.length(dp)
                traj_len[tid] = wp.max(L, 1e-3)

    elif state == PickSmState.LIFT_OBJECT:
        gripper_state[tid] = GripperState.CLOSE

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

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-A1-IK-Abs-v0-ppo",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 10.0
    env_cfg.terminations.illegal_contact.params["sensor_cfg"].body_names = []

    env = gym.make("Isaac-Lift-Cube-A1-IK-Abs-v0-ppo", cfg=env_cfg)
    env.reset()

    traj_cfg_base = FRAME_MARKER_CFG.copy()
    traj_cfg_base.prim_path = "/Visuals/Traj/pts"
    traj_cfg_base.markers["frame"].scale = (0.02, 0.02, 0.02)
    traj_vis = VisualizationMarkers(traj_cfg_base)

    qI = torch.tensor([1, 0, 0, 0], device=env.unwrapped.device, dtype=env.unwrapped.scene.env_origins.dtype).view(1, 4).repeat(N_PTS, 1)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.015
    )

    ENABLE_PLOT = True
    if args_cli.enable_cameras and ENABLE_PLOT:
        rgb_viz   = CameraViz(H=480, W=640, title="camera rgb",   env_id=0, mode="rgb")
        depth_viz = CameraViz(H=480, W=640, title="camera depth", env_id=0, mode="depth")
        depth_viz.set_depth_display(auto_scale=True, near=0.1, far=4.0, invert=False)

    while simulation_app.is_running():
        with torch.inference_mode():
            observations, rewards, terminateds, truncateds, infos  = env.step(actions)
            dones = terminateds | truncateds

            camera_rgb = env.unwrapped.scene["camera"].data.output["rgb"]
            camera_depth = env.unwrapped.scene["camera"].data.output["distance_to_image_plane"]

            if args_cli.enable_cameras and ENABLE_PLOT:
                rgb_viz.update(camera_rgb)
                depth_viz.update(camera_depth)

            origins = env.unwrapped.scene.env_origins

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - origins

            obstacle_data: RigidObjectData = env.unwrapped.scene["screen"].data
            obstacle_position = obstacle_data.root_pos_w - origins
            obstacle_quat = obstacle_data.root_quat_w

            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
                torch.cat([obstacle_position, obstacle_quat], dim=-1),
            )

            lift_mask = (pick_sm.sm_state == int(PickSmState.LIFT_OBJECT))
            active = lift_mask.nonzero(as_tuple=False).squeeze(-1)  # (K,)

            if active.numel() > 0:
                traj_vis.set_visibility(True)

                traj_view = pick_sm.traj_buf.view(env.unwrapped.num_envs, N_PTS, 3)  # (N,n,3)
                pts_all = traj_view[active] + origins[active].unsqueeze(1)

                pts_flat = pts_all.reshape(-1, 3)
                qI_flat = qI.repeat(active.numel(), 1)
                traj_vis.visualize(pts_flat, qI_flat)
            else:
                traj_vis.set_visibility(False)

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()