import argparse
import isaacgymenvs
from flamingo_env import Flamingo, get_cfg
from isaacgymenvs.tasks import isaacgym_task_map
import os
import torch


def main(args):
    seed = args.seed
    num_envs = args.num_envs
    task_name = 'Flamingo'

    cfg = get_cfg(args.cfg_path, num_envs, task_name)
    isaacgym_task_map["Flamingo"] = Flamingo

    envs = isaacgymenvs.make(
        seed=seed,
        task=task_name,
        num_envs=num_envs,
        graphics_device_id=args.graphics_device_id,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        headless=args.headless,
        cfg=cfg
    )

    print("Observation space is", envs.observation_space)
    print("Action space is", envs.action_space)
    envs.render()
    obs = envs.reset()
    for _ in range(100000):
        random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device=args.sim_device) - 1.0
        envs.step(torch.zeros_like(random_actions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flamingo task with IsaacGymEnvs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_envs', type=int, default=5, help='Number of environments')
    parser.add_argument('--task_name', type=str, default='Flamingo', help='Task name')
    parser.add_argument('--cfg_path', type=str, default='./cfg', help='Path to the configuration directory')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics device ID')
    parser.add_argument('--sim_device', type=str, default='cpu', help='Simulation device (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--rl_device', type=str, default='cuda:0', help='RL device (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')

    args = parser.parse_args()
    main(args)