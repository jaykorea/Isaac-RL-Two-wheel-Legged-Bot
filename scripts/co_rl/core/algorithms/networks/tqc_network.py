import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scripts.co_rl.core.utils.utils import weight_init



class QuantileCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, n_quantiles, activation_fc, device):
        super(QuantileCritic, self).__init__()
        self.activation_fc = activation_fc
        self.device = device
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], n_quantiles)
        self.apply(weight_init)

    def forward(self, state ,action):
        sa = torch.cat([state ,action], dim=1)

        x = self.activation_fc(self.input_layer(sa))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        quantiles = self.output_layer(x)
        return quantiles


class EnsembleQuantileCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dims, n_critics, n_quantiles, activation_fc=F.relu):
        super(EnsembleQuantileCritic, self).__init__()
        self.device = device

        self.critics = []
        for i in range(n_critics):
            one_critic = QuantileCritic(state_dim, action_dim, hidden_dims, n_quantiles, activation_fc, device)
            self.add_module(f'qf{i}', one_critic)
            self.critics.append(one_critic)

    def forward(self, state, action):
        quantiles = torch.stack(tuple(one_critic(state, action) for one_critic in self.critics), dim=1)
        return quantiles


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dims=(256, 256), activation_fc=F.relu, device='cuda'):
        super(GaussianPolicy, self).__init__()
        self.device = device

        self.log_std_min = -20
        self.log_std_max = 2

        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

        self.apply(weight_init)


    def forward(self, state):
        x = self.activation_fc(self.input_layer(state))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())

        unbounded_action = distribution.rsample()
        # [Paper: Appendix C] Enforcing Action Bounds: [a_min, a_max] -> [-1, 1]
        bounded_action = torch.tanh(unbounded_action)
        action = bounded_action * self.action_rescale + self.action_rescale_bias

        # We must recover ture log_prob from true distribution by 'The Change of Variable Formula'.
        log_prob = distribution.log_prob(unbounded_action) - torch.log(self.action_rescale *
                                                                       (1 - bounded_action.pow(2).clamp(0, 1)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_rescale + self.action_rescale_bias

        return action, log_prob, mean




