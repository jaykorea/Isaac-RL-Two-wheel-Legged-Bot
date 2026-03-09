# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_rotate_inverse


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def obstacle_ee_aabb_distance(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("screen"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Compute point-to-OBB (Oriented Bounding Box) distance.
    Correctly handles rotated obstacles.
    """

    # entities
    obstacle: RigidObject = env.scene[obstacle_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 1. World positions & Orientations
    c_w = obstacle.data.root_pos_w                  # (N, 3) Obstacle Center
    q_w = obstacle.data.root_quat_w                 # (N, 4) Obstacle Orientation
    p_w = ee_frame.data.target_pos_w[..., 0, :]     # (N, 3) EE Position

    # 2. Transform EE position into Obstacle's LOCAL Frame
    # (World 상의 거리 벡터를 물체의 회전 반대 방향으로 회전시킴)
    rel_pos_w = p_w - c_w
    rel_pos_local = quat_rotate_inverse(q_w, rel_pos_w)

    # 3. Obstacle size (Half-extent)
    obstacle_size = torch.tensor(
        obstacle.cfg.spawn.size,
        device=env.device,
    )
    half = 0.5 * obstacle_size + margin

    # 4. Calculate Distance in Local Frame (OBB Logic)
    # d = |p_local| - half
    d = rel_pos_local.abs() - half

    # 5. Outside distance only
    outside = torch.clamp(d, min=0.0)
    dist = torch.linalg.norm(outside, dim=-1)

    return dist