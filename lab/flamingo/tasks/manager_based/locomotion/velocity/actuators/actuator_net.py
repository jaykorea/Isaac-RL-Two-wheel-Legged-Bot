import torch
from collections.abc import Sequence
from isaaclab.utils.assets import read_file

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators.actuator_net import (
    ActuatorNetLSTM as BaseActuatorNetLSTM,
    ActuatorNetMLP as BaseActuatorNetMLP,
)
from isaaclab.actuators.actuator_pd import DCMotor


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

        # run network inference
        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )
        self.computed_effort = torques.reshape(self._num_envs, self.num_joints)

        # clip the computed effort based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # return torques
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

        history_length = max(self.cfg.input_idx) + 1

        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_vel_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_sin_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_cos_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
            self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
            self._joint_vel_history = self._joint_vel_history.roll(1, 1)
            self._joint_vel_history[:, 0] = joint_vel

            pos_input = torch.cat(
                [self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            vel_input = torch.cat(
                [self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            network_input = torch.cat(
                [pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale],
                dim=1
            )
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = self._joint_vel_error_history.roll(1, 1)
            self._joint_sin_history = self._joint_sin_history.roll(1, 1)
            self._joint_cos_history = self._joint_cos_history.roll(1, 1)

            self._joint_vel_error_history[:, 0] = control_action.joint_velocities - joint_vel
            self._joint_sin_history[:, 0] = torch.sin(joint_pos)
            self._joint_cos_history[:, 0] = torch.cos(joint_pos)

            vel_error_input = torch.cat(
                [self._joint_vel_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            sin_input = torch.cat(
                [self._joint_sin_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            cos_input = torch.cat(
                [self._joint_cos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            network_input = torch.cat(
                [vel_error_input * self.cfg.vel_scale, sin_input, cos_input],
                dim=1
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        self._joint_vel[:] = joint_vel

        with torch.inference_mode():
            torques = self.network(network_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale

        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
    

class ActuatorNetKAN(DCMotor):
    """Extended Actuator model based on multi-layer perceptron and joint history."""

    def __init__(self, cfg, *args, **kwargs):
        # Delayed import to avoid circular dependency
        from .actuator_cfg import ActuatorNetKANCfg

        self.cfg: ActuatorNetKANCfg = cfg

        formula_str = read_file(self.cfg.symbolic_formula).getvalue().decode().strip()
        self.kan_symbolic_formula, formula_input_dim = self.parse_formula_to_lambda(formula_str)

        super().__init__(cfg, *args, **kwargs)
        
        history_length = max(self.cfg.input_idx) + 1

        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_vel_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            input_dim = len(self.cfg.input_idx) * 2
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_sin_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_cos_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            input_dim = len(self.cfg.input_idx) * 3
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        assert input_dim >= formula_input_dim, (
            f"Symbolic formula requires at least {formula_input_dim} input dimensions, "
            f"but only {input_dim} are constructed from cfg.input_idx={self.cfg.input_idx}."
        )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
            self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
            self._joint_vel_history = self._joint_vel_history.roll(1, 1)
            self._joint_vel_history[:, 0] = joint_vel

            pos_input = torch.cat(
                [self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            vel_input = torch.cat(
                [self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            x_input = torch.cat(
                [pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale],
                dim=1
            )
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = self._joint_vel_error_history.roll(1, 1)
            self._joint_sin_history = self._joint_sin_history.roll(1, 1)
            self._joint_cos_history = self._joint_cos_history.roll(1, 1)

            self._joint_vel_error_history[:, 0] = control_action.joint_velocities - joint_vel
            self._joint_sin_history[:, 0] = torch.sin(joint_pos)
            self._joint_cos_history[:, 0] = torch.cos(joint_pos)

            vel_error_input = torch.cat(
                [self._joint_vel_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            sin_input = torch.cat(
                [self._joint_sin_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            cos_input = torch.cat(
                [self._joint_cos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            x_input = torch.cat(
                [vel_error_input * self.cfg.vel_scale, sin_input, cos_input],
                dim=1
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        self._joint_vel[:] = joint_vel

        with torch.inference_mode():
            torques = self.kan_symbolic_formula(x_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale

        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def parse_formula_to_lambda(self, formula_str: str):
        import re
        allowed_names = {
            'torch': torch,
            'sin': torch.sin,
            'cos': torch.cos,
            'tan': torch.tan,
            'exp': torch.exp,
            'log': torch.log,
            'sqrt': torch.sqrt,
            'abs': torch.abs,
            'pi': torch.pi,
        }
        print("[KAN] Original formula string:")
        print(formula_str)

        x_indices = [int(i) for i in re.findall(r'x_(\d+)', formula_str)]
        required_input_dim = max(x_indices) if x_indices else 0  # 그대로 사용

        for i in x_indices:
            formula_str = formula_str.replace(f'x_{i}', f'x[:, {i - 1}]')

        formula_func = eval(f'lambda x: {formula_str}', allowed_names)

        print(f"[KAN] Required input dim: {required_input_dim}")
        print("[KAN] Transformed PyTorch-compatible formula:")
        print(f"lambda x: {formula_str}")

        return formula_func, required_input_dim