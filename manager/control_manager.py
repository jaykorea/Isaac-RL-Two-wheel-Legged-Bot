import numpy as np


class ControlManager:
    @staticmethod
    def pd_controller(kp, tq, q, kd, td, d):
        return kp * (tq - q) + kd * (td - d)

    def __init__(self):
        self.filtered_action = None

    def low_pass_filter(self, action, alpha=1.0):
        if self.filtered_action is None:
            self.filtered_action = action
        else:
            self.filtered_action = alpha * action + (1.0 - alpha) * self.filtered_action
        return self.filtered_action
