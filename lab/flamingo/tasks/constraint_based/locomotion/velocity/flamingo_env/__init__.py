# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
)

##
# Register Gym environments.
##


#########################################CoRL##########################################################
#######################################################################################################

###########################################Track Velocity##############################################
gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v1-ppo-constraint",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Play-v1-ppo-constraint",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Stand_Drive,
    },
)

###########################################Track Velocity##############################################

#############################################Back Flip#################################################
gym.register(
    id="Isaac-Backflip-Flat-Flamingo-v1-ppo-constraint",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_back_flip_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Back_Flip,
    },
)

gym.register(
    id="Isaac-Backflip-Flat-Flamingo-Play-v1-ppo-constraint",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_back_flip_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Back_Flip,
    },
)
#############################################Back Flip################################################

######################################################################################################
############################################CoRL######################################################

#########################################Off Policy###################################################
# gym.register(
#     id="Isaac-Velocity-Flat-Flamingo-v3-sac",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
#         "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatSACRunnerCfg_Stand_Drive,
#     },
# )
# 
# gym.register(
#     id="Isaac-Velocity-Flat-Flamingo-v3-tqc",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
#         "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatTQCRunnerCfg_Stand_Drive,
#     },
# )
######################################################################################################
