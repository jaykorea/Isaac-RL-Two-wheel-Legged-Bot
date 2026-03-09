# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_binary(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    unoise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state using binary sampling with uniform noise only for provided keys.
    
    Args:
        env: Environment instance.
        env_ids: IDs of environments to reset.
        pose_range: Dict for position/orientation ranges {x,y,z,roll,pitch,yaw}.
                    Only keys present in this dict will be modified.
        velocity_range: Dict for linear/angular velocity ranges {x,y,z,roll,pitch,yaw}.
        unoise: Noise scale.
        asset_cfg: Asset configuration.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # 기본 상태 가져오기 (이 값들을 베이스로 수정함)
    root_states = asset.data.default_root_state[env_ids].clone()
    num_envs = len(env_ids)
    device = root_states.device

    # ──────────────────────────────
    # ① Helper: 특정 텐서의 특정 인덱스에만 노이즈/샘플링 적용
    # ──────────────────────────────
    def apply_binary_noise(target_tensor: torch.Tensor, ranges: dict, key_map: dict):
        """
        target_tensor: 수정할 텐서 (In-place modification)
        ranges: 범위 딕셔너리 (예: pose_range)
        key_map: 키('x')와 텐서 인덱스(0) 매핑
        """
        for key, (min_val, max_val) in ranges.items():
            if key not in key_map:
                continue
            
            idx = key_map[key]
            
            # Binary choice (0 or 1) -> (min or max)
            # shape: (num_envs,)
            choice = (torch.rand(num_envs, device=device) > 0.5).float()
            sample = choice * max_val + (1.0 - choice) * min_val
            
            # Apply uniform noise
            if unoise > 0:
                noise = (2 * unoise) * torch.rand(num_envs, device=device) - unoise
                sample += noise
            
            # 타겟 텐서에 더하기 (Offset 적용)
            target_tensor[:, idx] += sample

    # ──────────────────────────────
    # ② Pose (Position) 처리
    # ──────────────────────────────
    # 위치는 default + env_origins + offset(샘플링)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    
    # 위치에 대한 Key Map
    pos_map = {"x": 0, "y": 1, "z": 2}
    
    # positions 텐서에 직접 오프셋 더하기
    apply_binary_noise(positions, pose_range, pos_map)

    # ──────────────────────────────
    # ③ Pose (Orientation) 처리
    # ──────────────────────────────
    # 회전은 쿼터니언 곱셈이 필요하므로, Euler Delta를 먼저 만듦
    # 초기값 0 (변화 없음)
    euler_deltas = torch.zeros((num_envs, 3), device=device)
    rot_map = {"roll": 0, "pitch": 1, "yaw": 2}
    
    # euler_deltas에 값 채우기
    apply_binary_noise(euler_deltas, pose_range, rot_map)
    
    # Euler -> Quaternion 변환 후 기존 회전값에 곱하기
    orientations_delta = math_utils.quat_from_euler_xyz(
        euler_deltas[:, 0], euler_deltas[:, 1], euler_deltas[:, 2]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # ──────────────────────────────
    # ④ Velocity (Linear + Angular) 처리
    # ──────────────────────────────
    velocities = root_states[:, 7:13].clone()
    
    # 속도 맵 (Linear 0~2, Angular 3~5)
    vel_map = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    
    # velocities 텐서에 직접 오프셋 더하기
    apply_binary_noise(velocities, velocity_range, vel_map)

    # ──────────────────────────────
    # ⑤ 시뮬레이션 적용
    # ──────────────────────────────
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)