# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import (
    agents,
    rough_env,
    velocity_env_cfg
)

##
# Register Gym environments.
##


#PPO
gym.register(
    id="Isaac-Velocity-Rough-4w4l-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg. Flamingo4w4lRoughPPORunnerCfg_Stand_Drive
    },
)

gym.register(
    id="Isaac-Velocity-Rough-4w4l-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.Flamingo4w4lRoughPPORunnerCfg_Stand_Drive
    },
)
