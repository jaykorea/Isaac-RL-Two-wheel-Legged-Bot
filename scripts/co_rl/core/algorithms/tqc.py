#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from scripts.co_rl.core.algorithms.networks.tqc_network import GaussianPolicy, EnsembleQuantileCritic
from scripts.co_rl.core.modules import ReplayMemory
import torch
import torch.nn as nn

from scripts.co_rl.core.utils.utils import hard_update, soft_update, quantile_huber_loss_f


class TQC:
    def __init__(
        self,
            state_dim,
            action_dim,
            actor_hidden_dims,
            critic_hidden_dims,
            num_envs,
            device
    ):
        # Environment parameters
        self.env_device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = [-1, 1]
        self.num_envs = num_envs
        self.start_step = 10000
        self.update_after = 1000
        self.is_recurrent = False
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.temperature_lr = 0.0003
        self.n_drop_atoms = 2
        self.n_critics = 5
        self.n_quantiles = 25
        self.quantiles_total = self.n_critics * self.n_quantiles
        self.top_quantiles_to_drop = self.n_critics * self.n_drop_atoms

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 256
        self.buffer = ReplayMemory(num_envs, state_dim, action_dim, device=self.env_device, capacity=self.buffer_size)

        # Initialize the actor and critic networks
        self.actor = GaussianPolicy(state_dim, action_dim, self.action_bound, actor_hidden_dims, nn.ReLU(), self.env_device).to(self.env_device)
        self.critic = EnsembleQuantileCritic(state_dim, action_dim, device, critic_hidden_dims, self.n_critics, self.n_quantiles).to(device)
        self.target_critic = EnsembleQuantileCritic(state_dim, action_dim, device, critic_hidden_dims, self.n_critics, self.n_quantiles).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        hard_update(self.critic, self.target_critic)

        self.target_entropy = -torch.prod(torch.Tensor((action_dim,))).to(self.env_device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.env_device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.temperature_lr)

    def act(self, obs, cnt):
        if cnt < self.start_step:
            actions = self.action_bound[0] + (self.action_bound[1] - self.action_bound[0]) * torch.rand(self.num_envs, self.action_dim)
        else:
            actions, _ ,_  = self.actor.sample(obs)
        return actions.detach()

    def act_inference(self, obs):
        with torch.no_grad():
            _, _ , means  = self.actor.sample(obs)
        return means.detach()

    def process_env_step(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push_all(obs, actions, rewards, next_obs, dones)
        return

    def update(self, update_cnt):
        for _ in range(update_cnt):
            # Sample a batch of transitions
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            # Update the critic
            self.critic_optimizer.zero_grad()
            with torch.no_grad():
                next_actions, next_log_pis, _ = self.actor.sample(next_states)
                next_z = self.target_critic(next_states, next_actions)  # batch x nets x quantiles
                sorted_z, _ = torch.sort(next_z.reshape(self.batch_size, -1))
                sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]

                target_z = rewards + (1 - dones) * self.gamma * (sorted_z_part - self.log_alpha.exp() * next_log_pis)

            cur_z = self.critic(states, actions)
            critic_loss = quantile_huber_loss_f(cur_z, target_z, self.env_device)

            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the actor
            self.actor_optimizer.zero_grad()
            actions, log_pis, _ = self.actor.sample(states)
            z = self.critic(states, actions)
            actor_loss = (self.log_alpha.exp().detach() * log_pis - z.mean(2).mean(1, keepdim=True)).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the temperature parameter
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            soft_update(self.critic, self.target_critic, self.tau)

        return


