#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import huber_loss

from scripts.co_rl.core.modules import ActorCritic
from scripts.co_rl.core.storage import SRMRolloutStorage as RolloutStorage

import torch.nn.functional as F  # Functional module for loss functions and other utilities
import matplotlib.pyplot as plt


class SRMPPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        srm_net="lstm",
        srm_input_dim=32,
        cmd_dim=4,
        srm_hidden_dim=256,
        srm_output_dim=5,
        srm_num_layers=1,
        srm_r_loss_coef=1.0,
        srm_rc_loss_coef=1.0,
        use_acaps=False,
        acaps_lambda_t_coef=1.0,
        acaps_lambda_s_coef=1.0,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # SRM parameters
        self.srm_net = srm_net
        assert not (self.actor_critic.is_recurrent and srm_net == "mlp") # SRM-MLP is not supported for recurrent models

        self.srm_input_dim = srm_input_dim
        self.srm_hidden_dim = srm_hidden_dim
        self.srm_output_dim = srm_output_dim
        self.srm_num_layers = srm_num_layers
        self.srm_r_loss_coef = srm_r_loss_coef
        self.srm_rc_loss_coef = srm_rc_loss_coef
        # SRM env parameters
        self.cmd_dim = cmd_dim
        self.in_dim = self.srm_input_dim - self.cmd_dim
        # Not initialize here
        self.num_stacks = None
        self.stack_frames = None
        # self.stack_frames = self.num_stacks + 1 if self.actor_critic.is_recurrent is False else 0

        # SRM Configuration parameters
        self.cfg = None # not initialize here

        # ACAPS parameters
        self.use_acaps = use_acaps
        self.acaps_lambda_t_coef = acaps_lambda_t_coef
        self.acaps_lambda_s_coef = acaps_lambda_s_coef

        if self.srm_net == "lstm":
            self.srm = nn.LSTM(
                input_size=self.srm_input_dim,
                hidden_size=self.srm_hidden_dim,
                num_layers=self.srm_num_layers,
                batch_first=True,
                device=self.device,
            )
            self.srm_fc = nn.Linear(self.srm_hidden_dim, self.srm_output_dim, device=self.device)
        elif self.srm_net == "gru":
            self.srm = nn.GRU(
                input_size=self.srm_input_dim,
                hidden_size=self.srm_hidden_dim,
                num_layers=self.srm_num_layers,
                batch_first=True,
                device=self.device,
            )
            self.srm_fc = nn.Linear(self.srm_hidden_dim, self.srm_output_dim, device=self.device)
        elif self.srm_net == "mlp":
            layers = [nn.Linear(self.in_dim*self.stack_frames+self.cmd_dim, self.srm_hidden_dim), nn.ReLU(),
                      nn.Linear(self.srm_hidden_dim, int(self.srm_hidden_dim/2)), nn.ReLU(),
                      nn.Linear(int(self.srm_hidden_dim/2), int(self.srm_hidden_dim/4)), nn.ReLU()]
            self.srm = nn.Sequential(*layers).to(self.device)
            self.srm_fc = nn.Linear(int(self.srm_hidden_dim/4), self.srm_output_dim, device=self.device)
        else:
            raise ValueError("Unsupported SRM network type")

        self.srm_optimizer = optim.Adam(self.srm.parameters(), lr=learning_rate)

        print(f"SRM Network: {self.srm}\n")

    def init_storage(self, cfg, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.update_late_cfg(cfg)
        self.storage = RolloutStorage(
            cfg , num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )
    
    def update_late_cfg(self, cfg):
        self.cfg = cfg

        # initialize here to get the 'CoRlPolicyRunnerCfg' cfg
        self.num_stacks = self.cfg["num_policy_stacks"]
        self.stack_frames = self.num_stacks + 1 if self.actor_critic.is_recurrent is False else 0

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()

        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def store_prev_info(self, prev_rewards, prev_actions):
        self.transition.prev_rewards = prev_rewards.clone()
        self.transition.prev_actions = prev_actions.clone()

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_srm_loss = 0
        mean_r_loss = 0
        mean_rc_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            actions_difference_batch,
            rewards_batch,
            prev_rewards_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ** ACAPS loss **
            if self.use_acaps:
                # Use returns_batch as Q(s_t, a_t)
                q_values = returns_batch  # Returns are used as Q(s_t, a_t)

                alpha_t = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)  # Normalize α_t

                # Temporal Smoothness Regularization (L_T)
                temporal_diff = actions_difference_batch.pow(2).mean(dim=-1)  # Directly use actions_difference_batch
                L_t = (alpha_t.squeeze(-1) * temporal_diff).mean()

                # Spatial Smoothness Regularization (L_S)
                noise_obs_batch = obs_batch + torch.normal(mean=0, std=0.1, size=obs_batch.size()).clamp(-1, 1).to(obs_batch.device)
                if self.actor_critic.is_recurrent:
                    # Pass the last observation to act_inference
                    mu_batch_tilde = self.actor_critic.act_inference_rnn(
                        noise_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                    )
                else:
                    mu_batch_tilde = self.actor_critic.act_inference(noise_obs_batch)  # π(s̃)
                spatial_diff = (mu_batch - mu_batch_tilde).pow(2).mean(dim=-1)  # ||π(s_t) - π(s̃)||^2
                L_s = (alpha_t.squeeze(-1) * spatial_diff).mean()

            # ** SRM Training **
            s_o_t = obs_batch.clone().detach().to(self.device)
            s_f_t = critic_obs_batch.clone().detach().to(self.device)

            # Split obs into input x and target y for reconstruction
            if self.actor_critic.is_recurrent: #! No stack here
                x = s_o_t[:, :, : self.srm_input_dim]  # [batch_size, seq_len=1, input_size]
                y = s_f_t[:, :, -5:]  # Target for reconstruction
                srm_hidden_state, _ = self.srm(x)
            else:
                if self.srm_net in ["lstm", "gru"]:
                    if self.stack_frames > 0:
                        x = torch.cat((s_o_t[:, : self.in_dim], s_o_t[:, self.in_dim*self.stack_frames:self.in_dim*self.stack_frames+self.cmd_dim]), 1).unsqueeze(1)   # [batch_size, seq_len=1, input_size]
                    else:
                        x = s_o_t[:, :self.srm_input_dim]
                    srm_hidden_state, _ = self.srm(x)  # [batch_size, seq_len=n, hidden_size]
                elif self.srm_net == "mlp":
                    x = s_o_t[:, : self.in_dim*3 + self.cmd_dim].unsqueeze(1)   # [batch_size, seq_len=1, input_size]
                    srm_hidden_state = self.srm(x)  # [batch_size, seq_len=1, hidden_size]

                y = s_f_t[:, -5:].unsqueeze(1)  # Target for reconstruction

            # Hidden states to fully connected layer
            s_hat_f7_t = self.srm_fc(
                srm_hidden_state
            )  # * reconstructed full state notation: s_hat_f_t. reconstructed 7-dim state: s_hat_f7_t

            # Compute reconstruction loss for other features
            continuous_target = y[:, :, :3]
            continuous_output = s_hat_f7_t[:, :, :3]

            # Compute reconstruction loss
            # continuous_loss = huber_loss(continuous_output, continuous_target, delta=1.0)
            continuous_loss = F.mse_loss(continuous_output, continuous_target)

            # Compute binary cross entropy loss for 4:6 features
            binary_target = y[:, :, 3:5]
            binary_output = torch.sigmoid(s_hat_f7_t[:, :, 3:5])
            binary_loss = 0.01 * (F.binary_cross_entropy(binary_output, binary_target))

            # Combine binary and continuous reconstruction losses
            reconstruction_loss = continuous_loss + binary_loss

            # Compute reward consistency loss
            predicted_reward = s_hat_f7_t[:, :, 2].squeeze(-1)  # 복원된 리워드 (마지막 dimension)
            # reward_consistency_loss = huber_loss(predicted_reward, prev_rewards_batch.squeeze(-1), delta=1.0)  # 실제 리워드와 비교
            reward_consistency_loss = F.mse_loss(predicted_reward, prev_rewards_batch.squeeze(-1))  # 실제 리워드와 비교

            # SRM losses
            r_loss = self.srm_r_loss_coef * reconstruction_loss
            rc_loss = self.srm_rc_loss_coef * reward_consistency_loss
            srm_loss = r_loss + rc_loss

            # Back propagation for SRM
            self.srm_optimizer.zero_grad()
            srm_loss.backward()
            nn.utils.clip_grad_norm_(self.srm.parameters(), self.max_grad_norm)
            self.srm_optimizer.step()

            mean_srm_loss += srm_loss.item()
            mean_r_loss += r_loss.item()
            mean_rc_loss += rc_loss.item()

            # ** PPO Training **
            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            if self.use_acaps:
                loss += self.acaps_lambda_t_coef * L_t + self.acaps_lambda_s_coef * L_s

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_srm_loss /= num_updates
        mean_r_loss /= num_updates
        mean_rc_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_srm_loss, mean_r_loss, mean_rc_loss

    def encode_obs(self, obs):
        obs_ = obs.clone().detach().to(self.device)

        if self.actor_critic.is_recurrent:
            x = obs_[:, : self.srm_input_dim]
        else:
            if self.srm_net in ["lstm", "gru"]:
                if self.stack_frames > 0:
                    x = torch.cat((obs_[:, : self.in_dim], obs_[:, self.in_dim*self.stack_frames:self.in_dim*self.stack_frames+self.cmd_dim]), 1)   # [batch_size, seq_len=1, input_size]
                else:
                    x = obs_[:, :self.srm_input_dim]
            elif self.srm_net == "mlp":
                if self.stack_frames > 0:
                    x = obs_[:, : self.in_dim*self.stack_frames+self.cmd_dim]
                else:
                    x = obs_[:, : self.srm_input_dim]

        with torch.inference_mode():
            if self.srm_net in ["lstm", "gru"]:
                srm_out, _ = self.srm(x.unsqueeze(1))
                encoded_features_actor = self.srm_fc(srm_out[:, -1, :])
            elif self.srm_net == "mlp":
                srm_out = self.srm(x)
                encoded_features_actor = self.srm_fc(srm_out)
            encoded_features_actor[:, 3:5] = torch.round(torch.sigmoid(encoded_features_actor[:, 3:5]))

        obs_[:, -5:] = encoded_features_actor
        return obs_
