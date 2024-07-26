import numpy as np


class MemoryUtils:
    def __init__(self):
        self.total_episodes = 0
        self.total_transition = 0
        self.maximum_episodes = 10000
        self.episodes = []

    def load(self, states: np.ndarray, actions: np.ndarray, rewards: list, next_states: np.ndarray, terminated: list):
        if self.total_episodes < self.maximum_episodes:
            data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states,
                    'terminated': terminated}
            self.episodes.append(data)
        else:
            print("[Error] Maximum number of episodes exceeded!")
            raise MemoryError
        self.total_episodes += 1
        self.total_transition += states.shape[0]