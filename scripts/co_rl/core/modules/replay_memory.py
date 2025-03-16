import torch
import numpy as np


class ReplayMemory:
    def __init__(self, num_envs, state_dim, action_dim, device, capacity):
        self.device = device
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.num_envs = num_envs

        self.state_buffer = torch.empty(size=(self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.empty(size=(self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.empty(size=(self.capacity, 1), dtype=torch.float32, device=self.device)
        self.next_state_buffer = torch.empty(size=(self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.done_buffer = torch.empty(size=(self.capacity, 1), dtype=torch.float32, device=self.device)

    def push_all(self, states, actions, rewards, next_states, dones):
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        if self.position + self.num_envs < self.capacity:
            self.size = min(self.size + self.num_envs, self.capacity)
            self.state_buffer[self.position : self.position + self.num_envs, :] = states[: self.num_envs, :]
            self.action_buffer[self.position : self.position + self.num_envs, :] = actions[: self.num_envs, :]
            self.reward_buffer[self.position : self.position + self.num_envs, :] = rewards[: self.num_envs, :]
            self.next_state_buffer[self.position : self.position + self.num_envs, :] = next_states[: self.num_envs, :]
            self.done_buffer[self.position : self.position + self.num_envs, :] = dones[: self.num_envs, :]

            self.position = self.position + self.num_envs
            assert self.position < self.capacity
        else:
            for i in range(self.num_envs):
                self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def push(self, state, action, reward, next_state, done):
        self.size = min(self.size + 1, self.capacity)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = self.state_buffer[idxs]
        actions = self.action_buffer[idxs]
        rewards = self.reward_buffer[idxs]
        next_states = self.next_state_buffer[idxs]
        dones = self.done_buffer[idxs]

        return states, actions, rewards, next_states, dones


class TACOReplayMemory:
    def __init__(self, num_envs, state_dim, action_dim, device, capacity):
        self.device = device
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.num_envs = num_envs
        self._nstep = 3
        self._multistep = 3

        self.state_buffer = torch.empty(size=(self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.empty(size=(self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.empty(size=(self.capacity, 1), dtype=torch.float32, device=self.device)
        self.next_state_buffer = torch.empty(size=(self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.done_buffer = torch.empty(size=(self.capacity, 1), dtype=torch.float32, device=self.device)

    def push_all(self, states, actions, rewards, next_states, dones):
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        if self.position + self.num_envs < self.capacity:
            self.size = min(self.size + self.num_envs, self.capacity)
            self.state_buffer[self.position : self.position + self.num_envs, :] = states[: self.num_envs, :]
            self.action_buffer[self.position : self.position + self.num_envs, :] = actions[: self.num_envs, :]
            self.reward_buffer[self.position : self.position + self.num_envs, :] = rewards[: self.num_envs, :]
            self.next_state_buffer[self.position : self.position + self.num_envs, :] = next_states[: self.num_envs, :]
            self.done_buffer[self.position : self.position + self.num_envs, :] = dones[: self.num_envs, :]

            self.position = self.position + self.num_envs
            assert self.position < self.capacity
        else:
            for i in range(self.num_envs):
                self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def push(self, state, action, reward, next_state, done):
        self.size = min(self.size + 1, self.capacity)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        n_step = max(self._nstep, self._multistep)
        idxs = np.random.randint(0, self.size - n_step, size=batch_size)

        states = self.state_buffer[idxs]
        actions = self.action_buffer[idxs]
        rewards = self.reward_buffer[idxs]
        next_states = self.next_state_buffer[idxs]
        dones = self.done_buffer[idxs]

        action_seqs = torch.stack(
            [
                torch.stack([self.action_buffer[i], self.action_buffer[i + 1], self.action_buffer[i + 2]], dim=0)
                for i in idxs
            ]
        )
        r_next_obs = torch.stack([self.next_state_buffer[i + 3] for i in idxs])

        reward = torch.zeros((256, 1)).to("cuda")
        discount = torch.ones((256, 1)).to("cuda")

        for k in range(self._nstep):
            step_reward = torch.stack([self.reward_buffer[i + k] for i in idxs])
            reward += discount * step_reward
            discount *= 0.99

        return states, actions, action_seqs, rewards, next_states, r_next_obs, dones
