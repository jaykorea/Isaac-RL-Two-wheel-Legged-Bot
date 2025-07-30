# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env,
)

##
# Register Gym environments.
##


#########################################CoRL###################################################
################################################################################################
gym.register(
    id="Isaac-Velocity-Flat-Humanoid-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_walk_cfg.HumanoidFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidFlatPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Humanoid-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_walk_cfg.HumanoidFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidFlatPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_walk_cfg.HumanoidRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidRoughPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_walk_cfg.HumanoidRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidRoughPPORunnerCfg_Stand_Walk,
    },
)