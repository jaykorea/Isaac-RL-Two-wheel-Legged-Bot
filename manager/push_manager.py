import numpy as np


class PushManager:
    def __init__(self):
        self.push_robot = True
        self.push_interval = 0.0

    def reset(self):
        self.push_robot = True

    def apply_push(self, env, push_vel, current_time, push_vel_range=(-0.5, 0.5), time_range=(3.0, 6.0), random=False):
        if current_time == 0.0:
            self.push_interval = np.random.uniform(time_range[0], time_range[1])

        if random:
            push_vel = np.array([0.0, 0.0, 0.0]) + np.random.uniform(push_vel_range[0], push_vel_range[1], 3)

        if self.push_interval <= current_time and self.push_robot:
            env.data.qvel[:3] = push_vel
            self.push_robot = False
