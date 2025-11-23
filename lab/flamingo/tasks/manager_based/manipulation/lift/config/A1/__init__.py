# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-Cube-A1-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:A1CubeLiftEnvCfg",
        "co_rl_cfg_entry_point": agents.co_rl_cfg.A1PPORunnerCfg_Lift_Cube,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-A1-ppo-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:A1CubeLiftEnvCfg_PLAY",
        "co_rl_cfg_entry_point": agents.co_rl_cfg.A1PPORunnerCfg_Lift_Cube,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-A1-IK-Abs-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:A1CubeLiftEnvCfg",
        "co_rl_cfg_entry_point": agents.co_rl_cfg.A1PPORunnerCfg_Lift_Cube,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-A1-IK-Abs-v0-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:A1CubeLiftEnvCfg_PLAY",
        "co_rl_cfg_entry_point": agents.co_rl_cfg.A1PPORunnerCfg_Lift_Cube,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Teddy-Bear-A1-IK-Abs-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:A1TeddyBearLiftEnvCfg",
        "co_rl_cfg_entry_point": agents.co_rl_cfg.A1PPORunnerCfg_Lift_Cube,
    },
    disable_env_checker=True,
)
##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-A1-IK-Rel-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:A1LiftEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
