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

## Methodology
### Robot Configuration
The robot has two legs, each with a hip and shin joint, and two wheels. The robot's task is to maintain an upright position while also being capable of forward and backward motion.

### Simulation Environment
We use Nvidia's Isaac Gym to simulate the robot's dynamics and environment. The Isaac Gym provides a PyTorch-compatible interface, allowing seamless integration with popular RL algorithms.

### State Observations
The robot's state comprises its joint positions and velocities, body position, orientation, and linear and angular velocities. Specifically:

Joint positions for hips and shins: 4 variables
Wheel positions: 2 variables
Body position (x, y, z): 3 variables
Body orientation (roll, pitch, yaw): 3 variables
Body linear velocity (x, y, z): 3 variables
Body angular velocity (roll, pitch, yaw): 3 variables

### Action Space
The robot's actions are the torques applied to the joints and wheels.

### Reward Function Design
The reward function aims to incentivize the robot to maintain an upright orientation while discouraging excessive movements or control efforts.

![1](https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/ebd9d6bd-23a5-4392-8f4e-96cf6491db51)

* OrientationReward: Rewards the robot for maintaining an upright orientation. This is calculated as<br/>
 ![2](https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/985741a5-8797-465f-b65a-f9553cd1427f)

* SmoothnessReward<br/>
![3](https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/f67621d2-f270-4294-87d9-05566af0e3b4)

* ControlEffort: Penalizes large torques or forces applied at the joints and wheels.

### Reset Function Design
The reset function is triggered if:

1. The robot's pitch or roll exceeds a set threshold, indicating that it has fallen over.
2. The height of the robot's base_link drops below a certain value, indicating that it has kneeled or otherwise left a standard operating position.

## Experiments and Results
TBD

## Conclusion
I presented an RL-based approach for balancing a two-wheeled legged robot using Isaac Gym. The reward and reset functions were designed to address the unique challenges posed by this robot configuration. Experimental results demonstrate the effectiveness of our approach in various scenarios.
