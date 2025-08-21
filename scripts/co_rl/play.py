# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt

# local imports
import cli_args  # isort: skip
from scripts.co_rl.core.runners import OffPolicyRunner

from scripts.co_rl.core.utils.str2bool import str2bool
from scripts.co_rl.core.utils.analyzer import Analyzer

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CO-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the task.")
parser.add_argument("--stack_frames", type=int, default=None, help="Number of frames to stack.")
parser.add_argument("--plot", type=str2bool, default="False", help="Plot the data.")
parser.add_argument(
    "--analyze",
    type=str,
    nargs="+",  
    default=None,
    help="Specify which data to analyze (e.g., cmd_vel joint_vel torque)."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")

parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")

# append CO-RL cli arguments
cli_args.add_co_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from scripts.co_rl.core.runners import OnPolicyRunner, SRMOnPolicyRunner
from isaaclab.utils.dict import print_dict

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlVecEnvWrapper,
    export_env_as_pdf,
    export_policy_as_jit,
    export_policy_as_onnx,
    export_srm_as_onnx,
)

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


# Import extensions to set up environment tasks
import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv, ManagerBasedConstraintRLEnvCfg

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with CO-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks

    is_off_policy = False if agent_cfg.to_dict()["algorithm"]["class_name"] in ["PPO", "SRMPPO"] else True
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("co_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if isinstance(env.unwrapped, ManagerBasedConstraintRLEnv):
        agent_cfg.use_constraint_rl = True

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    if args_cli.analyze is not None:
        analyze_items = args_cli.analyze[0].split()
        analyzer = Analyzer(env=env, analyze_items=analyze_items, log_dir=log_dir)
    # wrap around environment for co-rl
    env = CoRlVecEnvWrapper(env, agent_cfg)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if is_off_policy:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        if args_cli.algo == "srmppo":
            runner = SRMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif args_cli.algo == "ppo":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)

    # Initialize GRU model and FC layer
    srm = None
    if hasattr(runner.alg, "srm") and hasattr(runner.alg, "srm_fc"):
        # GRU 모델 가져오기
        srm = runner.alg.srm

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if is_off_policy:
        export_policy_as_jit(runner.alg, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    else:
        export_policy_as_jit(
            runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
        if args_cli.algo == "srmppo":
            export_srm_as_onnx(
                runner.alg.srm, runner.alg.srm_fc, device=agent_cfg.device, path=export_model_dir, filename="srm.onnx"
            )
    
    # export environment to pdf
    export_env_as_pdf(yaml_path=os.path.join(log_dir, "params", "env.yaml"), pdf_path=os.path.join(export_model_dir, "env.pdf"))


    # reset environment
    obs, _ = env.get_observations()
    
    timestep = 0
    # Simulate environment and collect data
    while simulation_app.is_running():
        with torch.inference_mode():
            if srm is not None:
                encoded_obs = runner.alg.encode_obs(obs)
                actions = policy(encoded_obs)
            else:
                actions = policy(obs)
            clipped_actions = torch.clamp(actions, -1.0, 1.0)
            obs, _, _, extras = env.step(clipped_actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
                
        # Extract the relevant slices and convert to numpy
        if args_cli.analyze is not None:
            analyzer.append(extras['observations']['obs_info'])
    
    env.close()

    if args_cli.analyze is not None:
        analyzer.export()
        
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()