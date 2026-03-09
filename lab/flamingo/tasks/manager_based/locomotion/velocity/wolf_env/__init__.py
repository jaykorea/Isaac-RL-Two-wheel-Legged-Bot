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
    id="Isaac-Velocity-Flat-Wolf-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.WolfFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.WolfFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Wolf-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.WolfFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.WolfFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Wolf-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.WolfRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.WolfRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Wolf-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.WolfRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.WolfRoughPPORunnerCfg_Stand_Drive,
    },
)