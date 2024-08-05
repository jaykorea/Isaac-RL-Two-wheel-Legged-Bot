import torch


class ExtraUtils:
    def __init__(self):
        self.episode_sums = {
            "track_lin_vel_xy_exp": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "track_ang_vel_z_exp": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "lin_vel_z_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "anv_vel_xy_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_torques_joint_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_torques_wheels_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_acc_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "action_rate_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "undesired_contact": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "flat_orientation_l2": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "base_target_range_height": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_hip": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_align_shoulder": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_align_leg": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_hip": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_shoulder": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_leg": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "error_vel_xy": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "error_vel_yaw": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "is_terminated": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "total_reward": torch.zeros(1, dtype=torch.float, requires_grad=False),
        }

        self.extras = {}

    def reset_episode_sums(self):
        self.extras = {}
        self.episode_sums = {key: torch.zeros(1, dtype=torch.float, requires_grad=False) for key in self.episode_sums}

    def update_episode_sums(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.episode_sums:
                self.episode_sums[key] += value

    def update_extras(self, step_counter):
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = self.episode_sums[key] / step_counter

    def get_episode_sums(self):
        return self.episode_sums

    def get_extras(self):
        return self.extras
