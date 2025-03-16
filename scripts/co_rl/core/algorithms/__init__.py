#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .srmppo import SRMPPO
from .sac import SAC
from .tqc import TQC
from .taco import TACO

__all__ = ["PPO", "SRMPPO", "SAC", "TQC", "TACO"]
