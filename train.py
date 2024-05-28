import argparse
import isaacgymenvs
from flamingo_env import Flamingo, get_cfg
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.train import launch_rlg_hydra

def main(args):
    seed = args.seed
    num_envs = args.num_envs
    task_name = 'Flamingo'

    cfg = get_cfg(args.cfg_path, num_envs, task_name)
    cfg.test = False
    cfg.headless = False
    #cfg.checkpoint = 'runs/Flamingo_28-00-14-59/nn/Flamingo.pth'
    isaacgym_task_map[task_name] = Flamingo

    launch_rlg_hydra(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flamingo task with IsaacGymEnvs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_envs', type=int, default=1024, help='Number of environments')
    parser.add_argument('--task_name', type=str, default='Flamingo', help='Task name')
    parser.add_argument('--cfg_path', type=str, default='./cfg', help='Path to the configuration directory')

    args = parser.parse_args()
    main(args)
