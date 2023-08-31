# Isaac_gym_legged_bot
two wheel legged bot for Isaac gym reinforcement learning

# Overview
This repo contains the code for learning an RL policy in [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) for a two wheeled legged robot

1. ```assets```: contains the URDF file of the robot

2. ```isaacgymenvs```: contains the IsaacGym environment and the the task specification/PPO parameters


# Install isaac gym4
## Install dependency
```
sudo apt update
sudo apt install build-essential -y
# verify it worked
gcc — version
```
## Install conda
download Miniconda3 Linux 64-bit for Python 3.8 here
```
bash Miniconda3-<latest>-Linux-x86_64.sh
```
## Check cuda - specific version is required.
plz follow this link 
https://developer.nvidia.com/isaac-gym
```
nvidia-smi
```
## Download IsaacGym4
Be sure to check whether the ubuntu & cuda version is supported!!!
```
https://developer.nvidia.com/isaac-gym
```

## Install IsaacGym4
```
cd isaacgym
./create_conda_env_rlgpu.sh       #takes a while
conda activate rlgpu
```

## Update LD Path
Should source the path before launch the nodes
```
export LD_LIBRARY_PATH=/home/stuart/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH
```

## Install IsaacGymEnv
1. Download [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)

2. Copy the URDF file contained in the assets folder in IsaacGymEnvs/assets

3. Copy ```Postech.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/task

4. Copy ```PostechPPO.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/train

5. Copy ```postech.py```in the assets folder in IsaacGymEnvs/isaacgymenvs/task

6. should import postech task on IsaacGymEnvs/isaacgymenvs/task/__init__.py
```
from .postech import Postech

...

"Postech": Postech,
```

7. run ```python train.py task=Postech```

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# "Balancing a Two-Wheeled Legged Robot using Isaac Gym and Reinforcement Learning"
## Abstract
This paper presents an approach to balance a two-wheeled legged robot using reinforcement learning (RL) with Nvidia's Isaac Gym. We detail the design of the reward and reset functions, which are critical for successful learning, and present experimental results to demonstrate the effectiveness of our approach.

## Introduction
Balancing robots with non-standard configurations, such as those with both wheels and legs, poses unique challenges that traditional control methods struggle to address. Reinforcement learning offers an alternative method that can adapt to complex dynamics and environments. In this work, we use Nvidia's Isaac Gym, a toolkit for RL in robotic simulation, to train a two-wheeled legged robot to maintain balance.
