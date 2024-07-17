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

Depending on the use case defined [above](#overview), follow the instructions to set up your extension template. Start with the [Basic Setup](#basic-setup), which is required for either use case.

### Basic Setup
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
#### Dependencies

This template depends on Isaac Sim and Lab. For detailed instructions on how to install these dependencies, please refer to the [installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).

- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Orbit](https://isaac-orbit.github.io/orbit/)

#### Configuration

Decide on a name for your project or extension. This guide will refer to this name as `<your_extension_name>`.

- Create a new repository based off this template [here](https://github.com/new?owner=isaac-orbit&template_name=orbit.ext_template&template_owner=isaac-orbit). Name your forked repository using the following convention: `"orbit.<your_extension_name>"`.

- Clone your forked repository to a location **outside** the orbit repository.

```bash
git clone <your_repository_url>
```

- To tailor this template to your needs, customize the settings in `config/extension.toml` and `setup.py` by completing the sections marked with TODO.

- Rename your source folder.

```bash
cd orbit.<your_extension_name>
mv orbit/ext_template orbit/<your_extension_name>
```

- Define the following environment variable to specify the path to your Orbit installation:

```bash
# Set the ORBIT_PATH environment variable to point to your Orbit installation directory
export ORBIT_PATH=<your_orbit_path>
```

#### Set Python Interpreter

Although using a virtual environment is optional, we recommend using `conda` (detailed instructions [here](https://isaac-orbit.github.io/orbit/source/setup/installation.html#setting-up-the-environment)). If you decide on using Isaac Sim's bundled Python, you can skip these steps.

- If you haven't already: create and activate your `conda` environment, followed by installing extensions inside Orbit:

```bash
# Create conda environment
${ORBIT_PATH}/orbit.sh --conda

# Activate conda environment
conda activate orbit

# Install all Orbit extensions in orbit/source/extensions
${ORBIT_PATH}/orbit.sh --install
```

- Set your `conda` environment as the default interpreter in VSCode by opening the command palette (`Ctrl+Shift+P`), choosing `Python: Select Interpreter` and selecting your `conda` environment.

Once you are in the virtual environment, you do not need to use `${ORBIT_PATH}/orbit.sh -p` to run python scripts. You can use the default python executable in your environment by running `python` or `python3`. However, for the rest of the documentation, we will assume that you are using `${ORBIT_PATH}/orbit.sh -p` to run python scripts.

#### Set up IDE

To setup the IDE, please follow these instructions:

1. Open the `orbit.<your_extension_template>` directory on Visual Studio Code IDE
2. Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Orbit installation.

If everything executes correctly, it should create a file .python.env in the .vscode directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Project Template

From within this repository, install your extension as a Python package to the Isaac Sim Python executable.

```bash
${ORBIT_PATH}/orbit.sh -p -m pip install --upgrade pip
${ORBIT_PATH}/orbit.sh -p -m pip install -e .
```

### Setup as Omniverse Extension

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the path that goes up to your repository's location without actually including the repository's own directory. For example, if your repository is located at `/home/code/orbit.ext_template`, you should add `/home/code` as the search path.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to your local Orbit directory. For example: `/home/orbit/source/extensions`
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.


### Installing Dependencies

To ensure that your program works as expected, please add your extensions's dependencies to the appropriate configuration file. Below are links for how to specify extension dependencies on ``IsaacSim`` and ``Orbit`` extensions, ``pip`` packages, ``apt`` packages, and [rosdep packages](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html).

- [Extensions](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_advanced.html#dependencies-section)
- [pip packages](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/#install-requires)
- [apt packages](https://isaac-orbit.github.io/orbit/source/setup/developer.html#extension-dependency-management)
- [rosdep packages](https://isaac-orbit.github.io/orbit/source/setup/developer.html#extension-dependency-management)

## Usage

### Project Template

We provide an example for training and playing a policy for ANYmal on flat terrain. Install [RSL_RL](https://github.com/leggedrobotics/rsl_rl) outside of the orbit repository, e.g. `home/code/rsl_rl`.

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
${ORBIT_PATH}/orbit.sh -p -m pip install -e .
```

Train a policy.

```bash
cd <path_to_your_extension>
${ORBIT_PATH}/orbit.sh -p scripts/rsl_rl/train.py --task Template-Velocity-Flat-Anymal-D-v0 --num_envs 4096 --headless
```

Play the trained policy.

```bash
${ORBIT_PATH}/orbit.sh -p scripts/rsl_rl/play.py --task Template-Velocity-Flat-Anymal-D-Play-v0 --num_envs 16
```

### Omniverse Extension

We provide an example UI extension that will load upon enabling your extension defined in `orbit/ext_template/ui_extension_example.py`. For more information on UI extensions, enable and check out the source code of the `omni.isaac.ui_template` extension and refer to the introduction on [Isaac Sim Workflows 1.2.3. GUI](https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html#gui).

## Pre-Commit

Pre-committing involves using a framework to automate the process of enforcing code quality standards before code is actually committed to a version control system, like Git. This process involves setting up hooks that run automated checks, such as code formatting, linting (checking for programming errors, bugs, stylistic errors, and suspicious constructs), and running tests. If these checks pass, the commit is allowed; if not, the commit is blocked until the issues are resolved. This ensures that all code committed to the repository adheres to the defined quality standards, leading to a cleaner, more maintainable codebase. To do so, we use the [pre-commit](https://pre-commit.com/) module. Install the module using:

```bash
pip install pre-commit
```

Run the pre-commit with:

```bash
pre-commit run --all-files
```

## Finalize

You are all set and no longer need the template instructions

- The `orbit/ext_template` and `scripts/rsl_rl` directories act as a reference template for your convenience. Delete them if no longer required.

- When ready, use this `README.md` as a template and customize where appropriate.

## Docker

For docker usage, we require the following dependencies to be set up:

- [Docker and Docker Compose](https://isaac-orbit.github.io/orbit/source/deployment/docker.html#docker-and-docker-compose)

- [Isaac Sim Container](https://isaac-orbit.github.io/orbit/source/deployment/docker.html#obtaining-the-isaac-sim-container)

Clone this template into the `${ORBIT_PATH}/source/extensions` directory, and set it up as described
above in [Configuration](#configuration) (no other steps in Setup section required). Once done, start and enter your 
container with:

```bash
# start container
${ORBIT_PATH}/docker/container.sh start

# enter container
${ORBIT_PATH}/docker/container.sh enter
```

More information on working with Docker in combination with Orbit can be found [here](https://isaac-orbit.github.io/orbit/source/deployment/index.html).

## Troubleshooting

### Docker Container

When running within a docker container, the following error has been encountered: `ModuleNotFoundError: No module named 'orbit'`. To mitigate, please comment out the docker specific environment definitions in `.vscode/launch.json` and run the following:

```bash
echo -e "\nexport PYTHONPATH=\$PYTHONPATH:/workspace/orbit.<your_extension_name>" >> ~/.bashrc
source ~/.bashrc
```

### License

The source code is released under a [BSD 3-Clause license](https://opensource.org/licenses/BSD-3-Clause).
