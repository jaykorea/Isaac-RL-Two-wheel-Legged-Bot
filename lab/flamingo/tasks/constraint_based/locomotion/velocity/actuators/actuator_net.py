import torch
from collections.abc import Sequence
from isaaclab.utils.assets import read_file

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators.actuator_net import (
    ActuatorNetLSTM as BaseActuatorNetLSTM,
    ActuatorNetMLP as BaseActuatorNetMLP,
)


class ActuatorNetLSTM(BaseActuatorNetLSTM):
    """Extended Actuator model based on recurrent neural network (LSTM)."""

    def __init__(self, cfg, *args, **kwargs):
        # Delayed import to avoid circular dependency
        from .actuator_cfg import ActuatorNetLSTMCfg

        self.cfg: ActuatorNetLSTMCfg = cfg
        super().__init__(cfg, *args, **kwargs)

        # Additional initializations or overrides if necessary
        if self.cfg.input_order not in ["pos_vel", "vel_pos"]:
            raise ValueError(
                f"Invalid input order for LSTM actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        input_dim = 2 if self.cfg.input_order == "pos_vel" else 3
        self.sea_input = torch.zeros(self._num_envs * self.num_joints, 1, input_dim, device=self._device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.input_order == "pos_vel":
            self.sea_input[:, 0, 0] = (control_action.joint_positions - joint_pos).flatten()
            self.sea_input[:, 0, 1] = joint_vel.flatten()
        elif self.cfg.input_order == "vel_pos":
            self.sea_input[:, 0, 0] = (control_action.joint_velocities - joint_vel).flatten()
            self.sea_input[:, 0, 1] = torch.sin(joint_pos).flatten()
            self.sea_input[:, 0, 2] = torch.cos(joint_pos).flatten()

        self._joint_vel[:] = joint_vel

        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )

        self.computed_effort = torques.reshape(self._num_envs, self.num_joints)
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class ActuatorNetMLP(BaseActuatorNetMLP):
    """Extended Actuator model based on multi-layer perceptron and joint history."""

    def __init__(self, cfg, *args, **kwargs):
        # Delayed import to avoid circular dependency
        from .actuator_cfg import ActuatorNetMLPCfg

        self.cfg: ActuatorNetMLPCfg = cfg
        super().__init__(cfg, *args, **kwargs)

        # Additional initializations or overrides if necessary
        history_length = max(self.cfg.input_idx) + 1
        self._joint_pos_error_history = torch.zeros(
            self._num_envs, history_length, self.num_joints, device=self._device
        )
        self._joint_vel_history = torch.zeros(self._num_envs, history_length, self.num_joints, device=self._device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
        self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
        self._joint_vel_history = self._joint_vel_history.roll(1, 1)
        self._joint_vel_history[:, 0] = joint_vel
        self._joint_vel[:] = joint_vel

        pos_input = torch.cat(
            [self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2
        ).view(self._num_envs * self.num_joints, -1)
        vel_input = torch.cat([self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2).view(
            self._num_envs * self.num_joints, -1
        )

        if self.cfg.input_order == "pos_vel":
            network_input = torch.cat([pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale], dim=1)
        elif self.cfg.input_order == "vel_pos":
            network_input = torch.cat([vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1)
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        torques = self.network(network_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


# class ActuatorNetLSTM_temp(BaseActuatorNetLSTM):
#     """Actuator model based on recurrent neural network (LSTM).

#     Unlike the MLP implementation :cite:t:`hwangbo2019learning`, this class implements
#     the learned model as a temporal neural network (LSTM) based on the work from
#     :cite:t:`rudin2022learning`. This removes the need of storing a history as the
#     hidden states of the recurrent network captures the history.

#     Note:
#         Only the desired joint positions are used as inputs to the network.
#     """

#     cfg: ActuatorNetLSTMCfg
#     """The configuration of the actuator model."""

#     def __init__(self, cfg: ActuatorNetLSTMCfg, *args, **kwargs):
#         super().__init__(cfg, *args, **kwargs)

#         # load the model from JIT file
#         file_bytes = read_file(self.cfg.network_file)
#         self.network = torch.jit.load(file_bytes, map_location=self._device)

#         # extract number of lstm layers and hidden dim from the shape of weights
#         num_layers = len(self.network.lstm.state_dict()) // 4
#         hidden_dim = self.network.lstm.state_dict()["weight_hh_l0"].shape[1]

#         # create buffers for storing LSTM inputs
#         if self.cfg.input_order == "pos_vel":
#             input_dim = 2  # position error + velocity
#         elif self.cfg.input_order == "vel_pos":
#             input_dim = 3  # velocity error + sin(position) + cos(position)
#         else:
#             raise ValueError(
#                 f"Invalid input order for LSTM actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
#             )

#         self.sea_input = torch.zeros(self._num_envs * self.num_joints, 1, input_dim, device=self._device)
#         self.sea_hidden_state = torch.zeros(
#             num_layers,
#             self._num_envs * self.num_joints,
#             hidden_dim,
#             device=self._device,
#         )
#         self.sea_cell_state = torch.zeros(
#             num_layers,
#             self._num_envs * self.num_joints,
#             hidden_dim,
#             device=self._device,
#         )
#         # reshape via views (doesn't change the actual memory layout)
#         layer_shape_per_env = (num_layers, self._num_envs, self.num_joints, hidden_dim)
#         self.sea_hidden_state_per_env = self.sea_hidden_state.view(layer_shape_per_env)
#         self.sea_cell_state_per_env = self.sea_cell_state.view(layer_shape_per_env)

#     """
#     Operations.
#     """

#     def reset(self, env_ids: Sequence[int]):
#         # reset the hidden and cell states for the specified environments
#         with torch.no_grad():
#             self.sea_hidden_state_per_env[:, env_ids] = 0.0
#             self.sea_cell_state_per_env[:, env_ids] = 0.0

#     def compute(
#         self,
#         control_action: ArticulationActions,
#         joint_pos: torch.Tensor,
#         joint_vel: torch.Tensor,
#     ) -> ArticulationActions:
#         # compute network inputs
#         if self.cfg.input_order == "pos_vel":
#             self.sea_input[:, 0, 0] = (control_action.joint_positions - joint_pos).flatten()
#             self.sea_input[:, 0, 1] = joint_vel.flatten()
#         elif self.cfg.input_order == "vel_pos":
#             self.sea_input[:, 0, 0] = (control_action.joint_velocities - joint_vel).flatten()
#             self.sea_input[:, 0, 1] = torch.sin(joint_pos).flatten()
#             self.sea_input[:, 0, 2] = torch.cos(joint_pos).flatten()
#         else:
#             raise ValueError(
#                 f"Invalid input order for LSTM actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
#             )

#         # save current joint vel for dc-motor clipping
#         self._joint_vel[:] = joint_vel

#         # run network inference
#         with torch.inference_mode():
#             # If the network does not expect hidden state and cell state as inputs
#             # torques = self.network(self.sea_input)

#             # If the network expects hidden state and cell state, use the following line instead
#             torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
#                 self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
#             )

#         self.computed_effort = torques.reshape(self._num_envs, self.num_joints)

#         # clip the computed effort based on the motor limits
#         self.applied_effort = self._clip_effort(self.computed_effort)

#         # return torques
#         control_action.joint_efforts = self.applied_effort
#         control_action.joint_positions = None
#         control_action.joint_velocities = None
#         return control_action


# class ActuatorNetMLP_temp(BaseActuatorNetMLP):
#     cfg: ActuatorNetMLPCfg

#     def __init__(self, cfg: ActuatorNetMLPCfg, *args, **kwargs):
#         super().__init__(cfg, *args, **kwargs)

#         # load the model from JIT file
#         file_bytes = read_file(self.cfg.network_file)
#         self.network = torch.jit.load(file_bytes, map_location=self._device)

#         # create buffers for MLP history
#         history_length = max(self.cfg.input_idx) + 1
#         self._joint_pos_error_history = torch.zeros(
#             self._num_envs, history_length, self.num_joints, device=self._device
#         )
#         self._joint_vel_history = torch.zeros(self._num_envs, history_length, self.num_joints, device=self._device)
#         self._joint_vel_error_history = torch.zeros(
#             self._num_envs, history_length, self.num_joints, device=self._device
#         )
#         self._joint_sin_pos_history = torch.zeros(self._num_envs, history_length, self.num_joints, device=self._device)
#         self._joint_cos_pos_history = torch.zeros(self._num_envs, history_length, self.num_joints, device=self._device)

#     def reset(self, env_ids: Sequence[int]):
#         self._joint_pos_error_history[env_ids] = 0.0
#         self._joint_vel_history[env_ids] = 0.0
#         self._joint_vel_error_history[env_ids] = 0.0
#         self._joint_sin_pos_history[env_ids] = 0.0
#         self._joint_cos_pos_history[env_ids] = 0.0

#     def compute(
#         self,
#         control_action: ArticulationActions,
#         joint_pos: torch.Tensor,
#         joint_vel: torch.Tensor,
#     ) -> ArticulationActions:
#         device = self._device

#         # move history queue by 1 and update top of history
#         self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
#         self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos

#         self._joint_vel_history = self._joint_vel_history.roll(1, 1)
#         self._joint_vel_history[:, 0] = joint_vel

#         self._joint_vel_error_history = self._joint_vel_error_history.roll(1, 1)
#         self._joint_vel_error_history[:, 0] = control_action.joint_velocities - joint_vel

#         self._joint_sin_pos_history = self._joint_sin_pos_history.roll(1, 1)
#         self._joint_sin_pos_history[:, 0] = torch.sin(joint_pos)

#         self._joint_cos_pos_history = self._joint_cos_pos_history.roll(1, 1)
#         self._joint_cos_pos_history[:, 0] = torch.cos(joint_pos)

#         self._joint_vel[:] = joint_vel

#         if self.cfg.input_order == "pos_vel":
#             # compute network inputs
#             pos_input = torch.cat(
#                 [self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
#                 dim=2,
#             )
#             pos_input = pos_input.view(self._num_envs * self.num_joints, -1)

#             vel_input = torch.cat(
#                 [self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
#                 dim=2,
#             )
#             vel_input = vel_input.view(self._num_envs * self.num_joints, -1)

#             network_input = torch.cat([pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale], dim=1)
#         elif self.cfg.input_order == "vel_pos":
#             # compute network inputs
#             vel_error_input = torch.cat(
#                 [self._joint_vel_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
#                 dim=2,
#             )
#             vel_error_input = vel_error_input.view(self._num_envs * self.num_joints, -1)

#             sin_pos_input = torch.cat(
#                 [self._joint_sin_pos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
#                 dim=2,
#             )
#             sin_pos_input = sin_pos_input.view(self._num_envs * self.num_joints, -1)

#             cos_pos_input = torch.cat(
#                 [self._joint_cos_pos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
#                 dim=2,
#             )
#             cos_pos_input = cos_pos_input.view(self._num_envs * self.num_joints, -1)

#             network_input = torch.cat(
#                 [vel_error_input * self.cfg.vel_scale, sin_pos_input, cos_pos_input],
#                 dim=1,
#             )
#         else:
#             raise ValueError(
#                 f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
#             )

#         # run network inference
#         network_input = network_input.to(device)
#         torques = self.network(network_input).view(self._num_envs, self.num_joints)
#         self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale

#         # clip the computed effort based on the motor limits
#         self.applied_effort = self._clip_effort(self.computed_effort)

#         # return torques
#         control_action.joint_efforts = self.applied_effort
#         control_action.joint_positions = None
#         control_action.joint_velocities = None
#         return control_action
