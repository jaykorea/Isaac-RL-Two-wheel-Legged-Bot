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
    id="Isaac-Position-Flat-Flamingo-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Position,
    },
)

gym.register(
    id="Isaac-Position-Flat-Flamingo-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Position,
    },
)


gym.register(
    id="Isaac-Position-Rough-Flamingo-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.stair_env_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoRoughPPORunnerCfg_Position,
    },
)

gym.register(
    id="Isaac-Position-Rough-Flamingo-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.stair_env_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoRoughPPORunnerCfg_Position,
    },
)


##########################################SRM###################################################


#########################################CoRL###################################################
gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v3-sac",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatSACRunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v3-tqc",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatTQCRunnerCfg_Stand_Drive,
    },
)