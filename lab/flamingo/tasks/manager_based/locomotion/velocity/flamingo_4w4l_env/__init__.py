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
    id="Isaac-Velocity-Flat-Flamingo4w4l-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.Flamingo4w4lFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.Flamingo4w4lFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo4w4l-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.Flamingo4w4lFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.Flamingo4w4lFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo4w4l-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.Flamingo4w4lRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.Flamingo4w4lRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo4w4l-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.Flamingo4w4lRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.Flamingo4w4lRoughPPORunnerCfg_Stand_Drive,
    },
)