# Isaac LAB for Flamingo

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Orbit](https://img.shields.io/badge/Lab-0.3.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Flamingo

https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/assets/95605860/75075512-d2c6-4373-a932-c299567022e6

https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/assets/95605860/a3618385-364a-4817-b817-a64cb9ebd6a9


## Sim 2 Sim framework
![image](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/assets/95605860/c242590d-b1d4-427e-8f52-4190cafc38e9)

- Simulation to Simulation framework is available on sim2sim_onnx branch
- You can simply inference trained policy (basically export as .onnx from isaac lab)

## Setup
### Install Isaac Lab

### Install lab.flamingo package
1. Set the ISAACSIM_PATH environment variable to point to your isaaclab installation directory(register on your environment [.bashrc])
   ```
   export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0"
   export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
   ```
2. replace 'source' folder into your isaaclab 'source' folder
3. install lab.flamingo pip package by running below command
   ```
   ${ISAACLAB_PATH}/orbit.sh -p -m pip install --upgrade pip
   ${ISAACLAB_PATH}/orbit.sh -p -m pip install -e .
   ```
