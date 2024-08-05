import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.memory_utils import MemoryUtils


def check_episode_numbers(pkl_file_path):
    # Load the MemoryUtils instance from the pkl file
    with open(pkl_file_path, 'rb') as f:
        mem_utils = pickle.load(f)

    # Ensure mem_utils is an instance of MemoryUtils
    if not isinstance(mem_utils, MemoryUtils):
        print("[Error] Loaded object is not an instance of MemoryUtils")
        return

    # Check the total number of episodes
    total_episodes = mem_utils.total_episodes
    print(f"Total episodes: {total_episodes}")

    # Optionally, inspect the episodes data
    for i, episode in enumerate(mem_utils.episodes):
        print(f"Episode {i+1}:")
        print(f"  States: {episode['states'].shape}")
        print(f"  Actions: {episode['actions'].shape}")
        print(f"  Rewards: {len(episode['rewards'])}")
        print(f"  Next States: {episode['next_states'].shape}")
        print(f"  Terminated: {len(episode['terminated'])}")
        print(f"  Info: {len(episode['info'])}")


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pkl_file_path = os.path.join(root_dir, 'memory1/raw_memory1.pkl')

check_episode_numbers(pkl_file_path)
