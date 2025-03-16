# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
import torch
from scripts.co_rl.core.env import VecEnv
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from lab.flamingo.isaaclab.isaaclab.envs.manager_based_constraint_rl_env import ManagerBasedConstraintRLEnv
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg
from scripts.co_rl.core.utils.state_handler import StateHandler


class CoRlVecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv, agent_cfg: CoRlPolicyRunnerCfg):
        """
        Args:
            env: The environment to wrap around.
            stack_frames: Number of frames to stack for observations.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv) and not isinstance(env.unwrapped, ManagerBasedConstraintRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv, ManagerBasedConstraintRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # Determine the number of policy and critic stacks
        self.num_policy_stacks = agent_cfg.num_policy_stacks
        self.num_critic_stacks = agent_cfg.num_critic_stacks

        # Determine if constraint RL is used
        self.use_constraint_rl = agent_cfg.use_constraint_rl
        
        if hasattr(self.unwrapped, "observation_manager"):
            group_obs_dim = self.unwrapped.observation_manager.group_obs_dim
            if "stack_policy" not in group_obs_dim:
                raise KeyError('"stack_policy" key is missing in observation_manager.group_obs_dim')
            if "none_stack_policy" not in group_obs_dim:
                raise KeyError('"none_stack_policy" key is missing in observation_manager.group_obs_dim')
            if "stack_critic" not in group_obs_dim:
                raise KeyError('"stack_critic" key is missing in observation_manager.group_obs_dim')
            if "none_stack_critic" not in group_obs_dim:
                raise KeyError('"none_stack_critic" key is missing in observation_manager.group_obs_dim')
    
        # Determine action and observation dimensions
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self.unwrapped.num_actions

        if hasattr(self.unwrapped, "observation_manager"):
            # -- Policy observations
            stack_policy_dim = self.unwrapped.observation_manager.group_obs_dim["stack_policy"][0]
            nonstack_policy_dim = self.unwrapped.observation_manager.group_obs_dim["none_stack_policy"][0]
            self.policy_state_handler = StateHandler(self.num_policy_stacks + 1, stack_policy_dim, nonstack_policy_dim)
            # state handler 내부에서 최종 policy 차원(policy_dim)을 계산합니다.]
            self.unwrapped.observation_manager.group_obs_dim["policy"] = (self.policy_state_handler.num_obs,)
            self.num_obs = self.policy_state_handler.num_obs
        else:
            self.num_obs = self.unwrapped.num_observations

        # -- Privileged observations (Critic)
        if hasattr(self.unwrapped, "observation_manager"):
            stack_critic_dim = self.unwrapped.observation_manager.group_obs_dim["stack_critic"][0]
            nonstack_critic_dim = self.unwrapped.observation_manager.group_obs_dim["none_stack_critic"][0]
            self.critic_state_handler = StateHandler(self.num_critic_stacks + 1, stack_critic_dim, nonstack_critic_dim)
            self.unwrapped.observation_manager.group_obs_dim["critic"] = (self.critic_state_handler.num_obs,)
            self.num_privileged_obs = self.critic_state_handler.num_obs
        elif hasattr(self.unwrapped, "num_states"):
            self.num_privileged_obs = self.unwrapped.num_states
        else:
            self.num_privileged_obs = 0

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        # Policy observations: 항상 state handler를 사용하여 "stack_policy"와 "none_stack_policy"를 합칩니다.
        if hasattr(self, "policy_state_handler"):
            if self.policy_state_handler.stack_buffer is None:
                policy_obs = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
            else:
                policy_obs = self.policy_state_handler.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
        
        #!Todo - should consider DirectEnv case
        else:
            policy_obs = obs_dict["policy"]
            

        obs_dict["policy"] = policy_obs

        # Critic observations: state handler가 있다면 "stack_critic"과 "none_stack_critic"을 합칩니다.
        if hasattr(self, "critic_state_handler"):
            if self.critic_state_handler.stack_buffer is None:
                critic_obs = self.critic_state_handler.reset(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            else:
                critic_obs = self.critic_state_handler.update(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            obs_dict["critic"] = critic_obs

        return policy_obs, {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        # Policy observations reset: state handler가 존재하면 reset() 호출, 없으면 단순 concat
        if hasattr(self, "policy_state_handler") and self.policy_state_handler is not None:
            policy_obs = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
        
        #!Todo - should consider DirectEnv case
        else: 
            policy_obs = obs_dict["policy"]
            
        obs_dict["policy"] = policy_obs

        # Critic observations reset: state handler가 존재하면 reset() 호출, 없으면 단순 concat
        if hasattr(self, "critic_state_handler") and self.critic_state_handler is not None:
            critic_obs = self.critic_state_handler.reset(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            obs_dict["critic"] = critic_obs

        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        if not self.use_constraint_rl:
            dones = (terminated | truncated).to(dtype=torch.long)
        else:
            dones = torch.max(terminated, truncated).to(dtype=torch.float32)
        # Update policy observations via state handler
        
        if hasattr(self, "policy_state_handler"):
            policy_obs = self.policy_state_handler.update(
                obs_dict["stack_policy"], obs_dict["none_stack_policy"]
            )
            obs_dict["policy"] = policy_obs        

        if hasattr(self, "critic_state_handler"):
            critic_obs = self.critic_state_handler.update(
                obs_dict["stack_critic"], obs_dict["none_stack_critic"]
            )
            obs_dict["critic"] = critic_obs

        policy_obs = obs_dict["policy"]
        extras["observations"] = obs_dict

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        
        return policy_obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

