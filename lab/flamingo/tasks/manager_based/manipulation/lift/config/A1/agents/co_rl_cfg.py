# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
)

######################################## [ PPO CONFIG] ########################################


@configclass
class A1PPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "A1-v0"
    experiment_description = "test"
    empirical_normalization = False
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class A1PPORunnerCfg_Lift_Cube(A1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "A1_Lift_Cube-v0"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]