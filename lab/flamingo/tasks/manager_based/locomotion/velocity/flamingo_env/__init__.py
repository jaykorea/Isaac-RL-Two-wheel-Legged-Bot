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

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoFlatPPORunnerCfg_Stand_Drive,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_play_cfg.FlamingoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoFlatPPORunnerCfg_Stand_Drive,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_play_cfg.FlamingoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v2",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_z_cfg.FlamingoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoFlatPPORunnerCfg_Track_Z,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Play-v2",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_z_play_cfg.FlamingoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoFlatPPORunnerCfg_Track_Z,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v2",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_track_z_cfg.FlamingoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoRoughPPORunnerCfg_Track_Z,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v2",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_track_z_play_cfg.FlamingoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoRoughPPORunnerCfg_Track_Z,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v3",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_walk_cfg.FlamingoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoRoughPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v3",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_walk_play_cfg.FlamingoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FlamingoRoughPPORunnerCfg_Stand_Walk,
    },
)
