# Isaac LAB for Flamingo

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/Lab-2.0.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Sim2Real - ZeroShot Transfer
<table>
    <td><img src="https://github.com/user-attachments/assets/bb14612c-85c2-43ce-a7df-8b09ee4d3f69" width="800" height="400"/></td>
</table>
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8f9f990d-e8e9-400a-82b2-1131ff73f891" width="385" height="170"/></td>
    <td><img src="https://github.com/user-attachments/assets/93c6b187-4680-435e-800a-9e6d3d570d13" width="385" height="170"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9991ff73-5b3e-4d10-9b63-548197f18e54" width="540" height="170"/></td>
    <td><img src="https://github.com/user-attachments/assets/545fd258-1add-499a-8c62-520e113a951b" width="540" height="170"/></td>
  </tr>
</table>


## Isaac Lab Flamingo
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0037889b-bab7-4686-a9a5-46ea9bbe6ac2" width="385" height="170"/></td>
    <td><img src="https://github.com/user-attachments/assets/16d8d025-7e57-479a-80d4-9cfef2cf9b6b" width="385" height="170"/></td>
  </tr>
</table>

## Sim 2 Sim framework - Lab to MuJoCo
![image](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/assets/95605860/c242590d-b1d4-427e-8f52-4190cafc38e9)

- Simulation to Simulation framework is available on sim2sim_onnx branch (Currently on migration update)
- You can simply inference trained policy (basically export as .onnx from isaac lab)

## Setup
This repo is tested on Ubuntu 20.04, and I recommend you to install 'local install'
### Install Isaac Sim
  ```
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html
  ```
### Install Isaac Lab
  ```
  https://github.com/isaac-sim/IsaacLab
  ```

### Install lab.flamingo package
1. clone repository
   ```
   git clone https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot
   ```
2. install lab.flamingo pip package by running below command
   run it on 'lab.flamingo' root path
   ```
   conda activate env_isaaclab # change to you conda env
   pip install -e .
   ```
4. Unzip assets(usd asset) on folder
   Since git does not correctly upload '.usd' file, you should manually unzip the usd files on assests folder
   ```
    path example: lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_4_1/
   ```

### Launch script
#### Train flamingo
  run it on 'lab.flamingo' root path
  ```
    python scripts/co_rl/train.py --task {task name} --algo ppo --num_envs 4096 --headless --num_policy_stacks {stack number on policy obs} --num_critic_stacks {stack number on critic obs}
  ```
### Train example - track velocity
  ```
    python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  ```
#### play flamingo
  run it on 'lab.flamingo' root path
  ```
    python scripts/co_rl/play.py --task {task name} --algo ppo --num_envs 64 --num_policy_stacks {stack number on policy obs} --num_critic_stacks {stack number on critic obs} --load_run {folder name} --plot False
  ```
  ```
    python scripts/co_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Play-v1-ppo --algo ppo --num_envs 64 --num_policy_stacks {stack number on policy obs} --num_critic_stacks {stack number on critic obs} --load_run 2025-03-16_17-09-35 --plot False
  ```
