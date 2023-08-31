# Isaac_gym_legged_bot
two wheel legged bot for Isaac gym reinforcement learning

## Overview
This repo contains the code for learning an RL policy in [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) for a two wheeled legged robot

1. ```assets```: contains the URDF file of the robot

2. ```isaacgymenvs```: contains the IsaacGym environment and the the task specification/PPO parameters


## Install isaac gym4
# Install dependency
```
sudo apt update
sudo apt install build-essential -y
# verify it worked
gcc — version
```
# Install conda
download Miniconda3 Linux 64-bit for Python 3.8 here
```
bash Miniconda3-<latest>-Linux-x86_64.sh
```
# Check cuda - specific version is required.
plz follow this link 
https://developer.nvidia.com/isaac-gym
```
nvidia-smi
```
# Download IsaacGym4
```
https://developer.nvidia.com/isaac-gym
```

## How to start the learning
1. Download [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)

2. Copy the URDF file contained in the assets folder in IsaacGymEnvs/assets

3. Copy ```Twip.yaml``` and ```FlywheelPendulum.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/task

4. Copy ```TwipPPO.yaml``` and ```FlywheelPendulumPPO.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/train

5. Copy ```twip.py``` and ```flywheel_pendulum.py```in the assets folder in IsaacGymEnvs/isaacgymenvs/task

6. Paste in IsaacGymEnvs/isaacgymenvs/tasks/__init__.py 
```
from .twip import Twip
from .flywheel_pendulum import FlywheelPendulum

....

"Twip": Twip,
"FlywheelPendulum": FlywheelPendulum,
```

7. run ```python train.py task=Twip``` or ```python train.py task=FlywheelPendulum```


