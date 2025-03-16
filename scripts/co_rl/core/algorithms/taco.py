#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from scripts.co_rl.core.algorithms.networks.taco_network import *
from scripts.co_rl.core.modules import TACOReplayMemory
import torch
import torch.nn as nn
import itertools
from scripts.co_rl.core.utils.utils import hard_update, soft_update, schedule


class TACO:
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
        self.projection_dim = 263
        self.feature_dim = 50
        self.taco_hidden_dim = 128
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = [-1, 1]
        self.num_envs = num_envs
        self.start_step = 10
        self.update_after = 1000
        self.is_recurrent = False
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.encoder_lr = 0.0003
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.taco_model_lr = 0.0003
        self.temperature_lr = 0.0003
        self.stddev_schedule = 'linear(1.0,0.1,1000)'
        self.stddev_clip = 0.3
        self.multistep = 3
        self.reward = True
        self.curl = True
        self.latent_a_dim = int(action_dim*1.25)+1
        
        # Augmentation
        self.aug = RandomShiftsAug(pad=4)
        
        # Replay memory
        self.buffer_size = 200000
        self.batch_size = 256
        self.buffer = TACOReplayMemory(num_envs, state_dim, action_dim, device=self.env_device, capacity=self.buffer_size)

        # Initialize the actor and critic networks
        self.act_tok = ActionEncoding(action_dim, self.taco_hidden_dim, self.latent_a_dim, self.multistep)
        self.encoder = Encoder().to(self.env_device)        
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
        )        
        self.actor = Actor(self.projection_dim, self.feature_dim, action_dim, self.action_bound, actor_hidden_dims, nn.ReLU(), self.env_device).to(self.env_device)
        self.critic = Twin_Q_net(self.projection_dim, self.feature_dim, self.latent_a_dim, self.env_device, critic_hidden_dims).to(self.env_device)
        self.target_critic = Twin_Q_net(self.projection_dim, self.feature_dim, self.latent_a_dim, self.env_device, critic_hidden_dims).to(self.env_device)
        self.taco_model = TACO_model(self.projection_dim, self.feature_dim, self.latent_a_dim, self.taco_hidden_dim, self.act_tok, self.encoder, self.multistep, self.env_device).to(self.env_device)
        hard_update(self.critic, self.target_critic)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.taco_model_opt = torch.optim.Adam(self.taco_model.parameters(), lr=self.taco_model_lr)
        
        self.target_entropy = -torch.prod(torch.Tensor((action_dim,))).to(self.env_device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.env_device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.temperature_lr)

    def act(self, obs, cnt):
        if cnt < self.start_step:
            actions = self.action_bound[0] + (self.action_bound[1] - self.action_bound[0]) * torch.rand(self.num_envs, self.action_dim)
        else:
            # obs = self.encoder(obs)
            stddev = schedule(self.stddev_schedule, cnt)
            dist = self.actor(obs, stddev)
            actions = dist.sample(clip=None)
        return actions.detach()

    def act_inference(self, obs):
        with torch.no_grad():
            means = self.encoder(obs)
        return means.detach()

    def process_env_step(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push_all(obs, actions, rewards, next_obs, dones)
        return

    def update_critic(self, obs, actions, rewards, gamma, next_obs, dones, step):
        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_actions = dist.sample(clip=self.stddev_clip)
            next_q_values_A, next_q_values_B = self.target_critic(next_obs, next_actions, self.act_tok)
            next_q_values = torch.min(next_q_values_A, next_q_values_B)
            target_q_values = rewards + (1 - dones) * gamma * next_q_values

        q_values_A, q_values_B = self.critic(obs, actions, self.act_tok)
        critic_loss = ((q_values_A - target_q_values) ** 2).mean() + ((q_values_B - target_q_values) ** 2).mean()

        self.encoder_opt.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_opt.step()
        
    def update_actor(self, obs, step):
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        actions = dist.sample(clip=self.stddev_clip)
        q_values_A, q_values_B = self.critic(obs, actions, self.act_tok)
        q_values = torch.min(q_values_A, q_values_B)

        self.actor_optimizer.zero_grad()
        actor_loss = (-q_values).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def update_taco(self, obs, actions, action_seq, r_next_obs, rewards):
        obs_anchor = self.aug(obs.float())
        obs_pos = self.aug(obs.float())
        z_a = self.taco_model.encode(obs_anchor)
        z_pos = self.taco_model.encode(obs_pos, ema=True)
        ### Compute CURL loss
        if self.curl:
            logits = self.taco_model.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.env_device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)
        
        ### Compute action encodings
        action_en = self.taco_model.act_tok(actions, seq=False) 
        action_seq_en = self.taco_model.act_tok(action_seq, seq=True)
        
        ### Compute reward prediction loss
        if self.reward:
            reward_pred = self.taco_model.reward(torch.concat([z_a, action_seq_en], dim=-1))
            reward_loss = F.mse_loss(reward_pred, rewards)
        else:
            reward_loss = torch.tensor(0.)
        
        ### Compute TACO loss
        next_z = self.taco_model.encode(self.aug(r_next_obs.float()), ema=True)
        curr_za = self.taco_model.project_sa(z_a, action_seq_en) 
        logits = self.taco_model.compute_logits(curr_za, next_z)
        labels = torch.arange(logits.shape[0]).long().to(self.env_device)
        taco_loss = self.cross_entropy_loss(logits, labels)
            
        self.taco_model_opt.zero_grad()
        (taco_loss + curl_loss + reward_loss).backward()
        self.taco_model_opt.step()
        
    def update(self, update_cnt):
        for _ in range(update_cnt):
            # Sample a batch of transitions
            states, actions, action_seqs, rewards, next_states, r_next_obs, dones = self.buffer.sample(self.batch_size)
            
            states = self.aug(states.float())
            next_states = self.aug(next_states.float())
            
            states_en = self.taco_model.encode(states)
            with torch.no_grad():
                next_states_en = self.taco_model.encode(next_states)

            # Update
            self.update_critic(states_en, actions, rewards, self.gamma, next_states_en, dones, step=update_cnt)
            self.update_actor(states_en.detach(), step=update_cnt)
            self.update_taco(states, actions, action_seqs, r_next_obs, rewards)
            
            soft_update(self.critic, self.target_critic, self.tau)

        return


