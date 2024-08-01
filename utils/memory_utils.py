import numpy as np


class MemoryUtils:
    def __init__(self):
        self.total_episodes = 0
        self.total_transition = 0
        self.maximum_episodes = 10000
        self.episodes = []

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.terminates = []
        self.infos = []

    def load(self):
        if self.total_episodes < self.maximum_episodes:
            data = {'states': np.array(self.states), 'actions': np.array(self.actions), 'rewards': self.rewards, 'next_states': np.array(self.next_states),
                    'terminated': self.terminates, 'info': self.infos}
            self.episodes.append(data)
        else:
            print("[Error] Maximum number of episodes exceeded!")
            raise MemoryError
        self.total_episodes += 1
        self.total_transition += np.array(self.states).shape[0]

    def store(self, states, actions, rewards, next_states, terminates, info):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.terminates.append(terminates)
        self.infos.append(info)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminates = []
        self.infos = []
