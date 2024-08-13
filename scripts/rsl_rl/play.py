"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import matplotlib.pyplot as plt


from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    export_policy_as_jit,
    export_policy_as_onnx,
)

#  from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from lab.flamingo.tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import lab.flamingo.tasks  # noqa: F401  TODO: import orbit.<your_extension_name>


def main():
    """Play with RSL-RL agent."""
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    # Initialize lists to store data
    joint_pos_list = []
    target_joint_pos_list = []
    joint_velocity_obs_list = []
    target_joint_velocity_list = []

    obs, _ = env.get_observations()

    # Simulate environment and collect data
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # Extract the relevant slices and convert to numpy
            joint_pos = obs[0, :6].cpu().numpy()
            target_joint_pos = (obs[0, 48:54] * 1.0).cpu().numpy()
            joint_velocity_obs = obs[0, 12:14].cpu().numpy()
            target_joint_velocity = (obs[0, 54:56] * 25.0).cpu().numpy()

            # Store the data
            joint_pos_list.append(joint_pos)
            target_joint_pos_list.append(target_joint_pos)
            joint_velocity_obs_list.append(joint_velocity_obs)
            target_joint_velocity_list.append(target_joint_velocity)

    env.close()

    # Plot the collected data after the simulation ends
    plt.figure(figsize=(14, 16))

    for i in range(6):
        plt.subplot(4, 2, i + 1)
        plt.plot([step[i] for step in joint_pos_list], label=f"Joint Position {i+1}")
        plt.plot([step[i] for step in target_joint_pos_list], label=f"Target Joint Position {i+1}", linestyle="--")
        plt.title(f"Joint Position {i+1} and Target Joint Position", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()

    for i in range(2):
        plt.subplot(4, 2, i + 7)
        plt.plot([step[i] for step in joint_velocity_obs_list], label=f"Observed Joint Velocity {i+1}")
        plt.plot([step[i] for step in target_joint_velocity_list], label=f"Target Joint Velocity {i+1}", linestyle="--")
        plt.title(f"Observed and Target Joint Velocity {i+1}", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    simulation_app.close()
