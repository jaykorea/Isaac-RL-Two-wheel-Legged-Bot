# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def object_flipped(
    env: ManagerBasedRLEnv,
    angle_threshold: float = 90.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object being flipped beyond a certain angle.

    Args:
        env: The environment.
        angle_threshold: The angle threshold in degrees. Defaults to 90.0 degrees.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    Returns:
        A tensor indicating if the object has flipped beyond the threshold for each environment.
    """
    # Convert the angle threshold from degrees to radians
    angle_threshold_rad = math.radians(angle_threshold)

    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]

    # extract the object's orientation in the world frame
    object_orientation = object.data.root_state_w[:, 3:7]

    # compute the object's Euler angles (roll, pitch, yaw) from the quaternion
    euler_angles = quaternion_to_euler_angles(object_orientation)

    # check if any of the absolute Euler angles exceed the threshold
    flipped = torch.any(torch.abs(euler_angles[:, :2]) > angle_threshold_rad, dim=1)

    return flipped


@staticmethod
def quaternion_to_euler_angles(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: A tensor representing quaternions (N, 4).

    Returns:
        A tensor representing the Euler angles (roll, pitch, yaw) in radians (N, 3).
    """
    # Extract components of the quaternion
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute Euler angles
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=1)