# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.FlamingoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FlamingoFlatPPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.FlamingoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FlamingoFlatPPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.FlamingoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FlamingoRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.FlamingoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FlamingoRoughPPORunnerCfg,
    },
)
