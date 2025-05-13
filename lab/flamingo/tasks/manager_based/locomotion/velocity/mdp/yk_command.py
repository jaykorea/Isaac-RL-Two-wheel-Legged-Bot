# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.managers import CommandTerm , CommandTermCfg
from isaaclab.assets import Articulation

from isaaclab.markers import VisualizationMarkersCfg , VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG,  BLUE_ARROW_X_MARKER_CFG

from isaaclab.utils import configclass



from collections.abc import Sequence

from dataclasses import MISSING

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class YKCommand(CommandTerm):
    """Command generator that generates a event flag.

    The command comprises of True of False.

    """
    cfg : YKCommandCfg
    
    def __init__(self, cfg : YKCommandCfg, env : ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg,env)
        
        self.env = env
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        self.event_command = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "EventCommand:\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The event command. Shape is (num_envs)."""
        return self.event_command
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resample event commands for specified environments.

        - Samples discrete angular and linear velocities from categorical distributions.
        - Resets the time_elapsed (event_command[:, 2]) to 0.0.
        """
        r = torch.empty(len(env_ids), device=self.device)

        ang_vel_range = self.cfg.ranges.ang_vel_z  # (low, high)
        lin_vel_range = self.cfg.ranges.lin_vel_z  # (low, high)

        # 카테고리 개수 설정
        num_categories = 10
        probabilities = torch.ones(num_categories, device=self.device) / num_categories  # Uniform categorical

        # 각 카테고리 값 생성
        ang_vel_categories = torch.linspace(ang_vel_range[0], ang_vel_range[1], num_categories, device=self.device)
        lin_vel_categories = torch.linspace(lin_vel_range[0], lin_vel_range[1], num_categories, device=self.device)

        # 환경 수만큼 샘플링
        sampled_ang_idx = torch.multinomial(probabilities, len(env_ids), replacement=True)
        sampled_lin_idx = torch.multinomial(probabilities, len(env_ids), replacement=True)

        # 카테고리 값으로 변환
        sampled_ang_vals = ang_vel_categories[sampled_ang_idx]
        sampled_lin_vals = lin_vel_categories[sampled_lin_idx]

        # 결과 저장
        self.event_command[env_ids, 0] = sampled_ang_vals
        self.event_command[env_ids, 1] = sampled_lin_vals
        self.event_command[env_ids, 2] = 0.0  # time_elapsed 초기화

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """
        Update the event command.

        - If ang_vel_z or lin_vel_z is nonzero, increase time_elapsed (event_command[:, 2]).
        - Reset all values when time_elapsed exceeds duration.
        - Reset values on environment reset.
        """

        is_active = torch.logical_or(
            torch.abs(self.event_command[:, 0]) > 1e-3,
            torch.abs(self.event_command[:, 1]) > 1e-3
        )

        self.event_command[:, 2] = torch.where(
            is_active,
            self.event_command[:, 2] + self._env.step_dt,
            torch.zeros_like(self.event_command[:, 2])
        )

        event_done = self.event_command[:, 2] > self.cfg.event_during_time
        self.event_command[event_done] = 0.0


        reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.event_command[reset_env_ids] = 0.0

       # Enforce standing for standing environments
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.event_command[standing_env_ids] = 0.0
        
    def _set_debug_vis_impl(self, debug_vis: bool):
            if debug_vis:
            # create markers if necessary for the first tomes
                if not hasattr(self, "command_active_visualizer") or not hasattr(self, "command_inactive_visualizer"):
                    self.command_active_visualizer = VisualizationMarkers(self.cfg.command_active_visualizer_cfg)
                    self.command_inactive_visualizer = VisualizationMarkers(self.cfg.command_inactive_visualizer_cfg)
                self.command_active_visualizer.set_visibility(True)
                self.command_inactive_visualizer.set_visibility(True)
            else:
                if hasattr(self, "command_active_visualizer"):
                    self.command_active_visualizer.set_visibility(False)
                if hasattr(self, "command_inactive_visualizer"):
                    self.command_inactive_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return

        # Clone base positions and raise them slightly for visualization
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.65

        # Get indices where event_command command is active
        active_indices = (self.event_command[:, 0] == 1.0).nonzero(as_tuple=True)[0]
        inactive_indices = (self.event_command[:, 0] == 0.0).nonzero(as_tuple=True)[0]

        # Visualize only for active indices
        if active_indices.numel() > 0:
            self.command_active_visualizer.visualize(base_pos_w[active_indices])
        if inactive_indices.numel() > 0:
            self.command_inactive_visualizer.visualize(base_pos_w[inactive_indices])


GREEN_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/event_command",
    markers={
        "cuboid": sim_utils.SphereCfg(
            radius=0.075,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

RED_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/event_command",
    markers={
        "cuboid": sim_utils.SphereCfg(
            radius=0.075,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)


@configclass
class YKCommandCfg(CommandTermCfg):
    
    class_type : type = YKCommand
    
    asset_name : str = MISSING
    rel_standing_envs: float = 0.1
    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for angular velocity around z-axis (yaw)."""

        lin_vel_z: tuple[float, float] = MISSING
        """Range for linear velocity along z-axis (vertical)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    event_during_time: float = 1.0
    
    command_active_visualizer_cfg : VisualizationMarkersCfg = GREEN_CUBOID_MARKER_CFG
    command_inactive_visualizer_cfg : VisualizationMarkersCfg = RED_CUBOID_MARKER_CFG

    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    """The sampled probability of environments where the robots should do event
    (the others follow the sampled angular velocity command). Defaults to 1.0."""