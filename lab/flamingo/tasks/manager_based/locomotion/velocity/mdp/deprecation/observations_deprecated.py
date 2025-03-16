# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
from collections import defaultdict

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


"""
For saving history
"""


state_storage = defaultdict(
    lambda: {"current": None, "previous": None, "previous2": None}
)


def update_state_storage(func_name, current_value):
    state = state_storage[func_name]
    if state["current"] is None:
        state["current"] = state["previous"] = state["previous2"] = current_value
    else:
        state["previous2"] = state["previous"]
        state["previous"] = state["current"]
        state["current"] = current_value
    return state


"""
Root state.
"""


def base_pos_z(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.root_pos_w[:, 2].unsqueeze(-1)
    update_state_storage("base_pos_z", current_value)
    return current_value


def prev_base_pos_z(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["base_pos_z"]["previous"]


def prev_prev_base_pos_z(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["base_pos_z"]["previous"]


def base_lin_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.root_lin_vel_b
    update_state_storage("base_lin_vel", current_value)
    return current_value


def prev_base_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["base_lin_vel"]["previous"]


def prev_prev_base_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["base_lin_vel"]["previous2"]


def base_ang_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.root_ang_vel_b
    update_state_storage("base_ang_vel", current_value)
    return current_value


def prev_base_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["base_ang_vel"]["previous"]


def prev_prev_base_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["base_ang_vel"]["previous2"]


def base_lin_acc(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :]
    update_state_storage("base_lin_acc", current_value)
    return current_value


def prev_base_lin_acc(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear acceleration in the asset's root frame."""
    return state_storage["base_lin_acc"]["previous"]


def prev_prev_base_lin_acc(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear acceleration in the asset's root frame."""
    return state_storage["base_lin_acc"]["previous2"]


def projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.projected_gravity_b
    update_state_storage("projected_gravity", current_value)
    return current_value


def prev_projected_gravity(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["projected_gravity"]["previous"]


def prev_prev_projected_gravity(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["projected_gravity"]["previous2"]


def root_pos_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.root_pos_w - env.scene.env_origins
    update_state_storage("root_pos_w", current_value)
    return current_value


def prev_root_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous Asset root position in the environment frame."""
    return state_storage["root_pos_w"]["previous"]


def prev_prev_root_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous Asset root position in the environment frame."""
    return state_storage["root_pos_w"]["previous2"]


def root_quat_w(
    env: ManagerBasedEnv,
    make_quat_unique: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    current_value = math_utils.quat_unique(quat) if make_quat_unique else quat
    update_state_storage("root_quat_w", current_value)
    # make the quaternion real-part positive if configured
    return current_value


def prev_root_quat_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous Asset root orientation in the environment frame."""
    return state_storage["root_quat_w"]["previous"]


def prev_prev_root_quat_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous Asset root orientation in the environment frame."""
    return state_storage["root_quat_w"]["previous2"]


def root_euler_angle(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    current_value = asset.data.root_quat_w
    roll, pitch, yaw = euler_xyz_from_quat(current_value)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    update_state_storage("root_euler_angle", rpy)
    return rpy


def prev_root_euler_angle(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous asset root orientation in the environment frame as Euler angles."""
    return state_storage["root_euler_angle"]["previous"]


def prev_prev_root_euler_angle(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous asset root orientation in the environment frame as Euler angles."""
    return state_storage["root_euler_angle"]["previous2"]


def root_lin_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.root_lin_vel_w
    update_state_storage("root_lin_vel_w", current_value)
    return current_value


def prev_root_lin_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["root_lin_vel_w"]["previous"]


def prev_prev_root_lin_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["root_lin_vel_w"]["previous2"]


def root_ang_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_value = asset.data.root_ang_vel_w
    update_state_storage("root_ang_vel_w", current_value)
    return current_value


def prev_root_ang_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root angular velocity in the asset's root frame."""
    return state_storage["root_ang_vel_w"]["previous"]


def prev_prev_root_ang_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root angular velocity in the asset's root frame."""
    return state_storage["root_ang_vel_w"]["previous2"]


"""
Joint state.
"""


def joint_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids]
    update_state_storage("joint_pos", current_value)
    return current_value


def prev_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["joint_pos"]["previous"]


def prev_prev_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["joint_pos"]["previous2"]


def joint_pos_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    update_state_storage("joint_pos_rel", current_value)
    return current_value


def prev_joint_pos_rel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["joint_pos_rel"]["previous"]


def prev_prev_joint_pos_rel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["joint_pos_rel"]["previous2"]


def joint_pos_rel_cos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as cosine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    current_value_cos = torch.cos(current_value)
    update_state_storage("joint_pos_rel_cos", current_value_cos)
    return current_value_cos


def prev_joint_pos_rel_cos(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous joint positions of the asset as cosine values."""
    return state_storage["joint_pos_rel_cos"]["previous"]


def prev_prev_joint_pos_rel_cos(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous joint positions of the asset as cosine values."""
    return state_storage["joint_pos_rel_cos"]["previous2"]


def joint_pos_rel_sin(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as sine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    current_value_sin = torch.sin(current_value)
    update_state_storage("joint_pos_rel_sin", current_value_sin)
    return current_value_sin


def prev_joint_pos_rel_sin(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous joint positions of the asset as sine values."""
    return state_storage["joint_pos_rel_sin"]["previous"]


def prev_prev_joint_pos_rel_sin(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous joint positions of the asset as sine values."""
    return state_storage["joint_pos_rel_sin"]["previous2"]


def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )
    update_state_storage("joint_pos_limit_normalized", current_value)
    return current_value


def prev_joint_pos_limit_normalized(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous joint positions of the asset as sine values."""
    return state_storage["joint_pos_limit_normalized"]["previous"]


def prev_prev_joint_pos_limit_normalized(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous joint positions of the asset as sine values."""
    return state_storage["joint_pos_limit_normalized"]["previous2"]


def joint_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_vel[:, asset_cfg.joint_ids]
    update_state_storage("joint_vel", current_value)
    return current_value


def prev_joint_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["joint_vel"]["previous"]


def prev_prev_joint_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["joint_vel"]["previous2"]


def joint_vel_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = (
        asset.data.joint_vel[:, asset_cfg.joint_ids]
        - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    )
    update_state_storage("joint_vel_rel", current_value)
    return current_value


def prev_joint_vel_rel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["joint_vel_rel"]["previous"]


def prev_prev_joint_vel_rel(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["joint_vel_rel"]["previous2"]


"""
Sensors.
"""


def height_scan(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    current_value = (
        sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    )
    update_state_storage("height_scan", current_value)

    return current_value


def prev_height_scan(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["height_scan"]["previous"]


def prev_prev_height_scan(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["height_scan"]["previous2"]


def body_incoming_wrench(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[
        :, asset_cfg.body_ids
    ]
    current_value = link_incoming_forces.view(env.num_envs, -1)
    update_state_storage("body_incoming_wrench", current_value)
    return current_value


def prev_body_incoming_wrench(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["body_incoming_wrench"]["previous"]


def prev_prev_body_incoming_wrench(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["body_incoming_wrench"]["previous2"]


"""
Actions.
"""


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    current_value = (
        env.action_manager.action
        if action_name is None
        else env.action_manager.get_term(action_name).raw_actions
    )
    update_state_storage("last_action", current_value)
    return current_value


def prev_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["last_action"]["previous"]


def prev_prev_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["last_action"]["previous2"]


def delayed_last_action(
    env: ManagerBasedEnv, action_name: str | None = None
) -> torch.Tensor:
    """The last input action to the environment with delay."""
    current_value = (
        env.action_manager.action
        if action_name is None
        else env.action_manager.get_term(action_name).raw_actions
    )
    # Apply delay
    delay = torch.rand((env.num_envs, 1), device=env.device)  # 0~1
    delayed_action = (1 - delay) * current_value + delay * (
        state_storage["last_action"]["previous"]
        if state_storage["last_action"]["previous"] is not None
        else current_value
    )
    # Update state storage
    update_state_storage("delayed_last_action", delayed_action)
    return delayed_action


def prev_delayed_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous delayed action."""
    return state_storage["delayed_last_action"]["previous"]


def prev_prev_delayed_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous-previous delayed action."""
    return state_storage["delayed_last_action"]["previous2"]


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    current_value = env.command_manager.get_command(command_name)
    update_state_storage("generated_commands", current_value)
    return current_value


def prev_generated_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous root linear velocity in the asset's root frame."""
    return state_storage["generated_commands"]["previous"]


def prev_prev_generated_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous-previous root linear velocity in the asset's root frame."""
    return state_storage["generated_commands"]["previous2"]

