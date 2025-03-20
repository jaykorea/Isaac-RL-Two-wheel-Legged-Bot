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
class FlamingoPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "FlamingoStand-v0"
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FlamingoFlatPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoFlatPPORunnerCfg_Track_Z(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

@configclass
class FlamingoFlatPPORunnerCfg_Track_RP(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Flat_Track_Roll_Pitch"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

@configclass
class FlamingoRoughPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Flamingo_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


###############################################################################################
######################################## [ SAC CONFIG] ########################################


@configclass
class FlamingoSACRunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 200000
    save_interval = 200
    experiment_name = "FlamingoStand-v0"
    algorithm = {"class_name": "SAC"}


@configclass
class FlamingoFlatSACRunnerCfg_Stand_Drive(FlamingoSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoFlatSACRunnerCfg_Track_Z(FlamingoSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoRoughSACRunnerCfg_Stand_Drive(FlamingoSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoRoughSACRunnerCfg_Track_Z(FlamingoSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Rough_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoRoughSACRunnerCfg_Stand_Walk(FlamingoSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Rough_Stand_Walk"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


###############################################################################################
######################################## [ TQC CONFIG] ########################################
@configclass
class FlamingoTQCRunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 200000
    save_interval = 200
    experiment_name = "FlamingoStand-v0"
    algorithm = {"class_name": "TQC"}


@configclass
class FlamingoFlatTQCRunnerCfg_Stand_Drive(FlamingoTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoFlatTQCRunnerCfg_Track_Z(FlamingoTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoRoughTQCRunnerCfg_Stand_Drive(FlamingoTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoRoughTQCRunnerCfg_Track_Z(FlamingoTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Rough_Track_Z"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoRoughTQCRunnerCfg_Stand_Walk(FlamingoTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Flamingo_Rough_Stand_Walk"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]
