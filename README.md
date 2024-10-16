# Isaac LAB for Flamingo

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/Lab-1.1.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Sim2Real - ZeroShot Transfer (Indoor)

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8f9f990d-e8e9-400a-82b2-1131ff73f891" width="520" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/93c6b187-4680-435e-800a-9e6d3d570d13" width="520" height="240"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9991ff73-5b3e-4d10-9b63-548197f18e54" width="520" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/545fd258-1add-499a-8c62-520e113a951b" width="520" height="240"/></td>
  </tr>
</table>

## Sim2Real - ZeroShot Transfer (Outdoor)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/548268e2-5919-425c-90b0-045b9368280a" width="520" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/3796372c-2241-49f3-95de-2a0f41276bb0" width="520" height="240"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/103e50bf-6405-4115-a34b-cfea6a31bbee" width="520" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/cb895df4-2b38-4a6f-945f-406fa8502f2c" width="520" height="240"/></td>
  </tr>
</table>

## Isaac Lab Flamingo

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1816afbd-4a18-4285-84a7-3f1f7cc92c8c" width="520" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/4c5cd561-2c4d-479c-90d1-391b3d1158cf" width="520" height="240"/></td>
  </tr>
</table>

## Sim 2 Sim framework - Lab to MuJoCo
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/61778cd7-be18-4a9f-9f1e-633af2f66ce2" width="520" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/5d2fe780-9c15-4a28-8213-78aa9f85e09d" width="520" height="240"/></td>
  </tr>
</table>

- Simulation to Simulation framework is available on sim2sim_onnx branch
- You can simply inference trained policy (basically export as .onnx from isaac lab)

## Setup
### Install Isaac Sim
```
https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html
```
### Install Isaac Lab
```
https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html
```

### Install lab.flamingo package
1. Set the ISAACSIM_PATH environment variable to point to your isaaclab installation directory(register on your environment [.bashrc])
   ```
   export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0"
   export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
   export ISAACLAB_PATH="${HOME}/IsaacLab"
   ```
2. clone repository
   ```
   git clone -b flamingo_isaac_lab_envs
   ```
3. replace 'source' folder into your isaaclab 'source' folder
   ```
   cp ${HOME}/lab.flamingo/modified_source/source ${HOME}/IsaacLab/source
   ```
5. install lab.flamingo pip package by running below command
   ```
   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip
   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .
   ```
### Launch script
#### train flamingo
on lab.flamingo root path, type
```
${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1 --num_envs 4096 --headless
```
#### play flamingo
```
${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Play-v1 --num_envs 32
```
