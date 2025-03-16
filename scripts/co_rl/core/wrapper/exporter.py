# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
import torch.nn as nn


def export_policy_as_jit(actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_srm_as_onnx(srm, srm_fc, device, path, filename="srm.onnx", verbose=False):
    """Export SRM (GRU/LSTM + Fully Connected layer) from actor_critic to ONNX.

    Args:
        actor_critic: The actor-critic module containing the SRM and SRM FC layers.
        path: The directory path to save the ONNX file.
        filename: The name of the ONNX file. Defaults to "srm.onnx".
        device: The device for exporting ("cpu" or "cuda").
        verbose: Whether to print the ONNX export details. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Create the exporter instance
    srm_exporter = _OnnxSRMExporter(srm, srm_fc, device, verbose=verbose)

    # Export the model
    srm_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            try:
                ## For On-Policy Algorithms
                obs = torch.zeros(1, self.actor[0].in_features)
            except:
                ## For Off-Policy Algorithms
                obs = torch.zeros(1, self.actor.input_layer.in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _OnnxSRMExporter(nn.Module):
    """Exporter of SRM (GRU or LSTM) with Fully Connected layer into ONNX, initialized from actor_critic."""

    def __init__(self, srm, srm_fc, device="cpu", verbose=False):
        super().__init__()
        self.verbose = verbose
        self.device = device

        # Extract SRM and SRM FC state_dicts
        srm_state_dict = srm.state_dict()
        srm_fc_state_dict = srm_fc.state_dict()

        self.srm_net = srm.__class__.__name__.lower()  # "gru" , "lstm", "sequential"

        # Extract dimensions from state_dict
        if self.srm_net in ["gru", "lstm"]:
            self.num_layers = srm.num_layers  # Number of layers in SRM
            self.hidden_dim = srm.hidden_size  # Hidden dimension
            self.input_dim = srm.input_size  # Input dimension
            self.output_dim = srm_fc_state_dict["weight"].shape[0]  # SRM FC output dimension
        elif self.srm_net == "sequential":
            # Sequential 모델의 경우 input_dim이 state_dict에 없음 -> 수동 입력
            self.input_dim = 88  # 모델 설계에서 고정된 입력 차원
            self.hidden_dim = 256
            self.output_dim = 7

        # Initialize SRM and FC layers
        if self.srm_net == "gru":
            self.srm = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,  # Assume 1 layer; extendable for multi-layer
                batch_first=True,
            ).to(self.device)
            self.srm_fc = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        elif self.srm_net == "lstm":
            self.srm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,  # Assume 1 layer; extendable for multi-layer
                batch_first=True,
            ).to(self.device)
            self.srm_fc = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        elif self.srm_net == "sequential":
            self.srm = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
                nn.ReLU(),
                nn.Linear(int(self.hidden_dim/2), int(self.hidden_dim/4)),
                nn.ReLU(),
            ).to(self.device)
            self.srm_fc = nn.Linear(int(self.hidden_dim/4), self.output_dim).to(self.device)

        # Load weights into the models
        self.srm.load_state_dict(srm_state_dict)
        self.srm_fc.load_state_dict(srm_fc_state_dict)

    def forward(self, obs, h_in=None):
        # Initialize hidden state if not provided
        if self.srm_net in ["gru", "lstm"]:
            # Initialize hidden state if not provided
            if h_in is None:
                if self.srm_net == "gru":
                    h_in = torch.zeros(self.num_layers, obs.size(0), self.hidden_dim, device=self.device)
                elif self.srm_net == "lstm":
                    h_in = (
                        torch.zeros(self.num_layers, obs.size(0), self.hidden_dim, device=self.device),
                        torch.zeros(self.num_layers, obs.size(0), self.hidden_dim, device=self.device),
                    )
            srm_out, h_out = self.srm(obs.unsqueeze(1), h_in)  # Add time dimension (seq_len=1)
            encoded_features = self.srm_fc(srm_out[:, -1, :])  # Use last output

            return encoded_features, h_out
        
        elif self.srm_net == "sequential":
            srm_out = self.srm(obs)
            encoded_features = self.srm_fc(srm_out)
            return encoded_features

    def export(self, path, filename="srm.onnx"):
        """Export the SRM model to an ONNX file."""
        self.to("cpu")

        # Create dummy inputs for ONNX export
        dummy_obs = torch.zeros(1, self.input_dim, dtype=torch.float32)  # Input observation

        # Initialize dummy hidden state based on SRM type
        if self.srm_net == "gru":
            dummy_h_in = torch.zeros(self.num_layers, 1, self.hidden_dim, dtype=torch.float32)  # GRU hidden state
        elif self.srm_net == "lstm":
            dummy_h_in = (
                torch.zeros(self.num_layers, 1, self.hidden_dim, dtype=torch.float32),  # LSTM hidden state
                torch.zeros(self.num_layers, 1, self.hidden_dim, dtype=torch.float32),  # LSTM cell state
            )
        else:
            dummy_h_in = None  # No hidden state needed for MLP

        # Prepare model inputs for export
        inputs = (dummy_obs,) if dummy_h_in is None else (dummy_obs, dummy_h_in)

        # Export to ONNX
        torch.onnx.export(
            self,
            inputs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"] if dummy_h_in is None else ["obs", "h_in"],
            output_names=["encoded_features", "h_out"] if dummy_h_in is not None else ["encoded_features"],
            dynamic_axes={"obs": {0: "batch_size"}} if dummy_h_in is None else {"obs": {0: "batch_size"}, "h_in": {1: "batch_size"}},
        )
        print(f"SRM ONNX model has been saved to {path}/{filename}")