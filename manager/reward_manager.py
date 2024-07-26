import torch
import math


class RewardManager:
    def __init__(self):
        self.init_params()

    def init_params(self):
        self.target_height = 0.35842
        self.min_target_height = 0.325
        self.max_target_height = 0.385
        self.default_joint_pos = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.target_joint_pos = torch.FloatTensor([0.0, 0.0, -0.261799, -0.261799, 0.56810467, 0.56810467])
        self.soft_limit_factor = 0.8
        self.joint_pos_limits = torch.FloatTensor([-1.5708, 1.5708, -1.5708, 1.5708, -0.0872665, 1.5708])
        self.track_lin_vel_xy_std = math.sqrt(0.25)
        self.track_anv_vel_z_std = math.sqrt(0.25)

        # Joint names and their indices
        self.joint_indices = {
            'left_hip': 0,
            'right_hip': 1,
            'left_shoulder': 2,
            'right_shoulder': 3,
            'left_leg': 4,
            'right_leg': 5,
            'left_wheel': 6,
            'right_wheel': 7
        }

    def get_privileged_observations(self, data, is_done):
        self.data = data
        self.current_height = torch.tensor(self.data.qpos[2], requires_grad=False)
        self.joint_acc = torch.tensor(self.data.qacc, requires_grad=False)
        self.contact_forces = torch.tensor(self.data.cfrc_ext[1:10], requires_grad=False)  # External contact forces
        self.is_done = is_done

    def get_observations(self, obs, step_counter, sim_step):
        self.obs_buf = torch.tensor(obs, requires_grad=False)
        self.step_counter = step_counter
        self.sim_step = sim_step

        joint_pos_mean = [((self.joint_pos_limits[2 * i] + self.joint_pos_limits[2 * i + 1]) / 2) for i in range(len(self.joint_pos_limits) // 2)]
        joint_pos_range = [(self.joint_pos_limits[2 * i + 1] - self.joint_pos_limits[2 * i]) for i in range(len(self.joint_pos_limits) // 2)]
        self.soft_joint_pos_limits = [[mean - 0.5 * range_val * self.soft_limit_factor, mean + 0.5 * range_val * self.soft_limit_factor] for mean, range_val in zip(joint_pos_mean, joint_pos_range)]

        self.joint_pos = self.obs_buf[:6]
        self.joint_vel = self.obs_buf[6:14]
        self.base_lin_vel = self.obs_buf[14:17]
        self.base_ang_vel = self.obs_buf[17:20]
        self.base_euler = self.obs_buf[20:23]
        self.actions = self.obs_buf[23:31]
        self.prev_actions = self.obs_buf[54:62]
        self.joint_torques = torch.tensor(self.data.actuator_force[:6], requires_grad=False)
        self.wheel_torques = torch.tensor(self.data.actuator_force[6:8], requires_grad=False)

        self.commands = torch.tensor([-0.1, 0.0, 0.0], requires_grad=False)

    def joint_deviation(self, joint_name, weight=1.0):
        joint_idx = self.joint_indices[joint_name]
        target_pos = self.target_joint_pos[joint_idx]
        return weight * torch.abs(self.joint_pos[joint_idx] - target_pos)

    def joint_align(self, joint_name1, joint_name2, weight=1.0):
        joint_idx1 = self.joint_indices[joint_name1]
        joint_idx2 = self.joint_indices[joint_name2]
        return weight * torch.abs(self.joint_pos[joint_idx1] - self.joint_pos[joint_idx2])

    def dof_pos_limits(self, joint_name1, joint_name2, weight=1.0):
        joint_idx1 = self.joint_indices[joint_name1]
        joint_idx2 = self.joint_indices[joint_name2]
        soft_limit = self.soft_joint_pos_limits[(joint_idx1 // 2)]
        dof_pos_limits = -(self.joint_pos[joint_idx1:joint_idx2+1] - soft_limit[0]).clip(max=0.0)
        dof_pos_limits += (self.joint_pos[joint_idx1:joint_idx2+1] - soft_limit[1]).clip(min=0.0)
        return weight * torch.sum(dof_pos_limits, dim=0)

    def track_lin_vel_xy_exp(self, weight=1.0):
        return weight * torch.exp(-torch.sum(torch.square(self.commands[:2] - self.base_lin_vel[:2]), dim=0) / self.track_lin_vel_xy_std**2)

    def track_ang_vel_z_exp(self, weight=1.0):
        return weight * torch.exp(-torch.square(self.commands[2] - self.base_ang_vel[2]) / self.track_anv_vel_z_std**2)

    def lin_vel_z_l2(self, weight=1.0):
        return weight * torch.square(self.base_lin_vel[2])

    def anv_vel_xy_l2(self, weight=1.0):
        return weight * torch.sum(torch.square(self.base_ang_vel[:2]), dim=0)

    def dof_torques_joint_l2(self, weight=1.0):
        return weight * torch.sum(torch.square(self.joint_torques), dim=0)

    def dof_torques_wheels_l2(self, weight=1.0):
        return weight * torch.sum(torch.square(self.wheel_torques), dim=0)

    def dof_acc_l2(self, weight=1.0):
        return weight * torch.sum(torch.square(self.joint_acc[7:14]), dim=0)

    def action_rate_l2(self, weight=1.0):
        return weight * torch.sum(torch.square(self.actions - self.prev_actions), dim=0)

    def undesired_contact(self, weight=1.0):
        shoulder_contact = torch.sum((torch.max(torch.norm(self.contact_forces[[2, 6]], dim=-1), dim=0)[0] > 1.0), dim=0)
        leg_contact = torch.sum((torch.max(torch.norm(self.contact_forces[[3, 7]], dim=-1), dim=0)[0] > 1.0), dim=0)
        return weight * (shoulder_contact + leg_contact)

    def flat_orientation_l2(self, weight=1.0):
        return weight * torch.sum(torch.square(self.base_euler[:2]), dim=0)

    def base_target_range_height(self, weight=1.0):
        in_range = (self.current_height >= self.min_target_height) & (self.current_height <= self.max_target_height)
        out_of_range_penalty = torch.abs(self.current_height - torch.where(self.current_height < self.min_target_height, self.min_target_height, self.max_target_height))
        return weight * torch.where(in_range, 0.005 * torch.ones_like(self.current_height), -out_of_range_penalty)

    def error_vel_xy(self, weight=1.0):
        return weight * torch.sum(torch.square(self.base_ang_vel[:2]), dim=0)

    def error_vel_yaw(self, weight=1.0):
        return weight * torch.square(self.base_lin_vel[2])

    def is_terminated(self, weight=1.0):
        terminated = self.is_done
        time_outs = torch.tensor([self.step_counter >= self.sim_step], dtype=torch.bool)
        if terminated and not time_outs:
            return weight * torch.tensor([1.0], requires_grad=False)
        return torch.zeros(1, requires_grad=False)