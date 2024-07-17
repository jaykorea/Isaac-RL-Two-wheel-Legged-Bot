from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import mujoco
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
import torch
import math
from prettytable import PrettyTable
import pygame
import random
import time
import pandas as pd
import argparse
import threading

class FLA_STAND(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, env_id='FLA_STAND-v0',
                 model_path='./assets/flamingo_pos_vel.xml',
                 frame_skip=4, render_mode='human'):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.dt_ = 0.005  # 시뮬레이션 주기
        self.sim_duration = 20  # 시뮬레이션 시간 unit: sec
        self.sim_step = (self.sim_duration / self.dt_) / self.frame_skip
        self.id = env_id
        self.obs_dim = 28 * 3 + 3  # 모델에 따라 조정
        self.act_dim = 8  # 모델에 따라 조정
        self.action_scaler = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 25.0, 25.0]
        self.filtered_action = None

        self.previous_states = []  # 이전 상태를 저장할 리스트
        self.step_counter = 0  # 스텝 카운터 초기화
        self.state_clip = 25

        pygame.init()
        self.screen = pygame.display.set_mode((480, 240))
        pygame.display.set_caption('Command Controller')

        self.commands = np.zeros(3)  # Initialize the command array
        self.max_linear_speed = 0.65
        self.max_angular_speed = 1.0
        self.acceleration = 0.1  # Rate at which speed increases
        self.deceleration = 0.05  # Rate at which speed decreases
        self.automation_command = False
        self.automation_command_lateral = True
        self.automation_command_back_and_forth = True
        self.start_key_time = time.time()
        self.key_duration = 3  # Each key press duration in seconds
        self.next_key_time = self.start_key_time
        self.current_key = None

        self.save_data = False
        self.save_trajectory = False
        self.plot_log = True
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
            "base_target_height": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_hip": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_shoulder": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_leg": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_hip": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_shoulder": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_leg": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "error_vel_xy": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "error_vel_yaw": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "alive_bonus": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "total_reward": torch.zeros(1, dtype=torch.float, requires_grad=False),
        }

        self.extras = {}

        self.joint_data = []
        self.wheel_data = []
        self.trajectory_data = []


        utils.EzPickle.__init__(self)

        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            observation_space=Box(low=-self.state_clip,
                                  high=self.state_clip,
                                  shape=(self.obs_dim,),
                                  dtype=np.float32),  # 3 프레임을 쌓음
        )

    def draw_keyboard(self, keys):
        self.screen.fill((0, 0, 0))  # 화면을 검은색으로 초기화

        # 각 키 위치와 크기 정의
        key_positions = {
            pygame.K_UP: (75, 50, 50, 50),
            pygame.K_DOWN: (75, 150, 50, 50),
            pygame.K_LEFT: (25, 100, 50, 50),
            pygame.K_RIGHT: (125, 100, 50, 50)
        }

        for key, pos in key_positions.items():
            if keys[key]:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            # 사각형 그리기
            pygame.draw.rect(self.screen, color, pos)

#        pygame.display.flip()

    def simulate_key_press(self):
        current_time = time.time()
        if current_time >= self.next_key_time:
            self.next_key_time = current_time + self.key_duration
            if self.automation_command_lateral and not self.automation_command_back_and_forth:
                self.current_key = random.choice([pygame.K_LEFT, pygame.K_RIGHT])
            elif self.automation_command_back_and_forth and not self.automation_command_lateral:
                self.current_key = random.choice([pygame.K_UP, pygame.K_DOWN])
            else:
                self.current_key = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT])

        keys = {pygame.K_UP: 0, pygame.K_DOWN: 0, pygame.K_LEFT: 0, pygame.K_RIGHT: 0}
        if self.current_key is not None:
            keys[self.current_key] = 1

        return keys

    def update_commands(self, automation=False):
        pygame.event.pump()  # Pygame 이벤트 큐를 처리

        if automation:
            keys = self.simulate_key_press()
        else:
            keys = pygame.key.get_pressed()

        # 선형 속도 (위, 아래 방향키)
        if keys[pygame.K_UP]:
            self.commands[0] += self.acceleration
            self.commands[0] = min(self.commands[0], self.max_linear_speed)
        elif keys[pygame.K_DOWN]:
            self.commands[0] -= self.acceleration
            self.commands[0] = max(self.commands[0], -self.max_linear_speed)
        else:
            if self.commands[0] > 0:
                self.commands[0] -= self.deceleration
                self.commands[0] = max(self.commands[0], 0)
            elif self.commands[0] < 0:
                self.commands[0] += self.deceleration
                self.commands[0] = min(self.commands[0], 0)

        # 각속도 (왼쪽, 오른쪽 방향키)
        if keys[pygame.K_LEFT]:
            self.commands[2] += self.acceleration
            self.commands[2] = min(self.commands[2], self.max_angular_speed)
        elif keys[pygame.K_RIGHT]:
            self.commands[2] -= self.acceleration
            self.commands[2] = max(self.commands[2], -self.max_angular_speed)
        else:
            if self.commands[2] > 0:
                self.commands[2] -= self.deceleration
                self.commands[2] = max(self.commands[2], 0)
            elif self.commands[2] < 0:
                self.commands[2] += self.deceleration
                self.commands[2] = min(self.commands[2], 0)

        self.draw_keyboard(keys)  # 키보드 그리기 함수 호출

    def quaternion_to_euler_array(self, quat):
        # Ensure quaternion is in the correct format [x, y, z, w]
        x, y, z, w = quat

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        # Returns roll, pitch, yaw in a NumPy array in radians
        return np.array([roll_x, pitch_y, yaw_z])

    def pd_controller(self, kp, tq, q, kd, td, d):
        """Calculates torques from position or velocity commands"""
        return kp * (tq - q) + kd * (td - d)

    def quat_rotate_inverse(self, quaternion, vectors):
        """
        Inverse rotate vectors by quaternion.
        Args:
            quaternion (np.ndarray): Quaternion (w, x, y, z)
            vectors (np.ndarray): Vectors to be rotated

        Returns:
            np.ndarray: Inverse rotated vectors
        """
        q = np.array(quaternion)
        v = np.array(vectors)

        # Calculate the conjugate of the quaternion
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])

        # Perform the inverse rotation
        t = 2 * np.cross(q_conj[1:], v)
        return v + q_conj[0] * t + np.cross(q_conj[1:], t)

    def wrap_to_2pi(self, angle):
        """
        Wraps an angle to the range [-2π, 2π].

        Parameters:
        - angle: The input angle in radians.

        Returns:
        - Wrapped angle in the range [-2π, 2π].
        """
        wrapped_angle = np.fmod(angle, 4 * np.pi)
        if wrapped_angle > 2 * np.pi:
            wrapped_angle -= 4 * np.pi
        elif wrapped_angle < -2 * np.pi:
            wrapped_angle += 4 * np.pi
        return wrapped_angle

    def _get_obs(self, action):
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])

        # 새로운 센서 데이터 사용
        # lin_vel = self.data.sensor('linear-velocity').data[:3].astype(np.double)
        # ang_vel = self.data.sensor('angular-velocity').data[:3].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(self.data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = self.data.sensor('angular-velocity').data.astype(np.double)
        euler = self.quaternion_to_euler_array(quat)
        q = self.data.qpos[[7, 11, 8, 12, 9, 13]]
        # qpos와 qvel의 특정 인덱스에 cos, sin 변환 적용
        # wheel_sin_cos = np.concatenate([
        #     [np.sin(self.data.qpos[10]), np.sin(self.data.qpos[14])],
        #     [np.cos(self.data.qpos[10]), np.cos(self.data.qpos[14])]
        # ])
        qd = self.data.qvel[[6, 10, 7, 11, 8, 12, 9, 13]]

        current_state = np.concatenate([
            q,  # joint pos
            # wheel_sin_cos,
            qd,  # joint vel
            # v,  # base lin vel
            omega,  # base ang vel
            euler,  # base euler
            action,
        ])

        # 이전 상태를 저장하고 새로운 상태를 추가
        if len(self.previous_states) < 2:
            self.previous_states.append(current_state)
            self.previous_states = self.previous_states + [np.zeros_like(current_state)] * (3 - len(self.previous_states))
        else:
            self.previous_states.pop(-1)
            # self.previous_states.append(current_state)
            self.previous_states.insert(0, current_state)

        self.update_commands(automation=self.automation_command)
        # 상태를 결합하여 반환
        return np.clip(np.concatenate([np.concatenate(self.previous_states), self.commands]), -self.state_clip, self.state_clip)

    def seed(self, seed):
        return

    def random_action_sample(self) -> np.ndarray:
        return self.action_space.sample()

    def low_pass_filter(self, action, alpha=1.0):
        if self.filtered_action is None:
            self.filtered_action = action
        else:
            self.filtered_action = alpha * action + (1 - alpha) * self.filtered_action
        return self.filtered_action

    def collect_and_save_data(self, obs, action_scaled, actuator_forces):
        """
        Collects data for joint and wheel datasets.

        Parameters:
        - obs: Tensor of observations
        - action_scaled: Tensor of scaled actions
        - actuator_forces: Tensor of actuator forces
        """
        joint_positions = obs[0:6]  # Current joint positions
        joint_velocities = obs[6:12]  # Current joint velocities
        target_positions = action_scaled[0:6]  # Target joint positions
        torques = actuator_forces[0:6]  # Current joint torques

        # Collect data for each joint
        joint_names = ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_leg", "right_leg"]

        for i, joint_name in enumerate(joint_names):
            self.joint_data.append({
                'joint': joint_name,
                # 'position_error': target_positions[i].item() - joint_positions[i].item(),
                'target_position': target_positions[i].item(),
                'current_position': joint_positions[i].item(),
                'velocity': joint_velocities[i].item(),
                'torque': torques[i].item()
            })

        # Assuming you need to do similar collection for wheels
        wheel_positions = [self.wrap_to_2pi(self.data.qpos[10]), self.wrap_to_2pi(self.data.qpos[13])]  # Current wheel positions (wrapped)
        wheel_velocities = obs[12:14]  # Current wheel velocities
        target_wheel_velocities = action_scaled[6:8]  # Target wheel velocities
        wheel_torques = actuator_forces[6:8]  # Current wheel torques

        # Collect data for each wheel
        wheel_names = ["left_wheel", "right_wheel"]
        for i, wheel_name in enumerate(wheel_names):
            self.wheel_data.append({
                'wheel': wheel_name,
                # 'velocity_error': target_wheel_velocities[i].item() - wheel_velocities[i].item(),
                'target_velocity': target_wheel_velocities[i].item(),
                'current_velocity': wheel_velocities[i].item(),
                'sin_position': np.sin(wheel_positions[i]),
                'cos_position': np.cos(wheel_positions[i]),
                'torque': wheel_torques[i].item()
            })

        # Append necessary data to trajectory_data
        data_entry = {}
        for i, joint_name in enumerate(joint_names):
            data_entry[f'{joint_name}_current_position'] = joint_positions[i].item()
            data_entry[f'{joint_name}_target_position'] = target_positions[i].item()

        for i, wheel_name in enumerate(wheel_names):
            data_entry[f'{wheel_name}_current_velocity'] = wheel_velocities[i].item()
            data_entry[f'{wheel_name}_target_velocity'] = target_wheel_velocities[i].item()

        self.trajectory_data.append(data_entry)

    def save_data_to_csv(self, joint_csv_dir, wheel_csv_dir):
        """
        Saves collected data to CSV files, each joint and wheel data independently.

        Parameters:
        - joint_csv_dir: Directory path to save joint dataset CSVs
        - wheel_csv_dir: Directory path to save wheel dataset CSVs
        """
        # Create directories if they do not exist
        if not os.path.exists(joint_csv_dir):
            os.makedirs(joint_csv_dir)

        if not os.path.exists(wheel_csv_dir):
            os.makedirs(wheel_csv_dir)

        def get_unique_filename(directory, filename):
            """
            Generate a unique filename by adding an index if the file already exists.
            """
            base, ext = os.path.splitext(filename)
            counter = 1
            new_filename = filename
            while os.path.exists(os.path.join(directory, new_filename)):
                new_filename = f"{base}_{counter}{ext}"
                counter += 1
            return os.path.join(directory, new_filename)

        # Save joint data for each joint
        joint_df = pd.DataFrame(self.joint_data)
        joint_names = joint_df['joint'].unique()
        for joint_name in joint_names:
            joint_data = joint_df[joint_df['joint'] == joint_name]
            filename = os.path.join(joint_csv_dir, f"{joint_name}_joint_data.csv")

            if os.path.exists(filename):
                joint_data.to_csv(filename, mode='a', header=False, index=False)
            else:
                joint_data.to_csv(filename, mode='w', header=True, index=False)

        # Save wheel data for each wheel
        wheel_df = pd.DataFrame(self.wheel_data)
        wheel_names = wheel_df['wheel'].unique()
        for wheel_name in wheel_names:
            wheel_data = wheel_df[wheel_df['wheel'] == wheel_name]
            filename = os.path.join(wheel_csv_dir, f"{wheel_name}_joint_data.csv")

            if os.path.exists(filename):
                wheel_data.to_csv(filename, mode='a', header=False, index=False)
            else:
                wheel_data.to_csv(filename, mode='w', header=True, index=False)

    def save_trajectory_to_csv(self, trajectory_csv_dir):
        """
        Saves collected data to CSV files, each joint and wheel data independently.

        Parameters:
        - trajectory_csv_dir: Directory path to save trajectory dataset CSV
        """
        # Create directory if it does not exist
        if not os.path.exists(trajectory_csv_dir):
            os.makedirs(trajectory_csv_dir)

        def get_unique_filename(directory, base_filename):
            """
            Generate a unique filename by adding an index if the file already exists.
            """
            base, ext = os.path.splitext(base_filename)
            counter = 1
            new_filename = base_filename
            while os.path.exists(os.path.join(directory, new_filename)):
                new_filename = f"{base}_{counter}{ext}"
                counter += 1
            return os.path.join(directory, new_filename)

        trajectory_df = pd.DataFrame(self.trajectory_data)
        base_filename = "trajectory.csv"
        filename = os.path.join(trajectory_csv_dir, base_filename)

        if os.path.exists(filename):
            filename = get_unique_filename(trajectory_csv_dir, base_filename)

        trajectory_df.to_csv(filename, mode='w', header=True, index=False)

    def step(self, action):
        obs = self._get_obs(action)
        # Apply low-pass filter to the action
        filtered_action = self.low_pass_filter(action)
        action_scaled = filtered_action * self.action_scaler

        # Apply the actions to the simulation
        self.do_simulation(action_scaled, self.frame_skip)
        self.step_counter += 1

        # Get new observations, rewards, done, and info
        obs = self._get_obs(filtered_action)

        # Collect data and save it
        if self.save_data or self.save_trajectory:
            self.collect_and_save_data(obs, action_scaled, self.data.actuator_force)

        reward = self._get_reward(obs)
        done = self._is_done()
        term = self.step_counter >= self.sim_step
        return obs, reward, done, term, {}

    def _get_reward(self, obs):
        obs_buf = torch.tensor(obs)
        with torch.no_grad():
            """Params"""
            target_height = 0.35842
            default_joint_pos = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            target_joint_pos = torch.FloatTensor([0.0, 0.0, -0.261799, -0.261799, 0.56810467, 0.56810467])
            soft_limit_factor = 0.8
            joint_pos_limits = torch.FloatTensor([-1.5708, 1.5708, -1.5708,
                                                  1.5708, -0.0872665, 1.5708])
            track_lin_vel_xy_std = math.sqrt(0.25)
            track_anv_vel_z_std = math.sqrt(0.25)

            """ Privileged Observations """
            current_height = torch.tensor(self.data.qpos[2])
            joint_acc = torch.tensor(self.data.qacc)
            self.contact_forces = self.data.cfrc_ext[1:10]  # External contact forces
            contact_forces = torch.tensor(self.contact_forces)

            """ Observations """
            self.commands = [-0.1, 0.0, 0.0]
            commands = torch.tensor(self.commands)
            # Calculate joint_pos_mean for pairs [0, 1], [2, 3], [4, 5]
            joint_pos_mean = [((joint_pos_limits[2 * i] + joint_pos_limits[2 * i + 1]) / 2) for i in
                              range(len(joint_pos_limits) // 2)]
            joint_pos_range = [(joint_pos_limits[2 * i + 1] - joint_pos_limits[2 * i]) for i in
                               range(len(joint_pos_limits) // 2)]
            # Calculate soft_joint_pos_limits
            soft_joint_pos_limits = [
                [mean - 0.5 * range_val * soft_limit_factor, mean + 0.5 * range_val * soft_limit_factor]
                for mean, range_val in zip(joint_pos_mean, joint_pos_range)
            ]

            joint_pos = obs_buf[:6]
            joint_vel = obs_buf[6:14]
            base_lin_vel = obs_buf[14:17]
            base_ang_vel = obs_buf[17:20]
            base_euler = obs_buf[20:23]
            actions = obs_buf[23:31]
            prev_actions = obs_buf[54:62]
            joint_torques = torch.tensor(self.data.actuator_force[:6])
            wheel_torques = torch.tensor(self.data.actuator_force[6:8])

            track_lin_vel_xy_exp = 1.5 * torch.exp(-torch.sum(torch.square(commands[:2] - base_lin_vel[:2]), dim=0) / track_lin_vel_xy_std**2)
            track_ang_vel_z_exp = 0.75 * torch.exp(-torch.square(commands[2] - base_ang_vel[2]) / track_anv_vel_z_std**2)

            lin_vel_z_l2 = -2.0 * torch.square(base_lin_vel[2])
            anv_vel_xy_l2 = -0.05 * torch.sum(torch.square(base_ang_vel[:2]), dim=0)

            dof_torques_joint_l2 = -1.0e-5 * torch.sum(torch.square(joint_torques), dim=0)
            dof_torques_wheels_l2 = -1.0e-5 * torch.sum(torch.square(wheel_torques), dim=0)

            dof_acc_l2 = -3.75e-7 * torch.sum(torch.square(joint_acc[7:14]), dim=0) # -3.75e-7

            action_rate_l2 = -0.015 * torch.sum(torch.square(actions - prev_actions), dim=0)

            shoulder_contact = torch.sum((torch.max(torch.norm(contact_forces[[2, 6]], dim=-1), dim=0)[0] > 1.0), dim=0)
            leg_contact = torch.sum((torch.max(torch.norm(contact_forces[[3, 7]], dim=-1), dim=0)[0] > 1.0), dim=0)
            undesired_contact = -1.0 * (shoulder_contact + leg_contact)

            flat_orientation_l2 = -0.5 * torch.sum(torch.square(base_euler[:2]), dim=0)

            base_target_height = -100.0 * torch.square(current_height - target_height)

            joint_deviation_hip = -5.0 * torch.sum(torch.abs((joint_pos[:2] - target_joint_pos[:2])), dim=0)
            joint_deviation_shoulder = -0.5 * torch.sum(torch.abs((joint_pos[2:4] - target_joint_pos[2:4])), dim=0)
            joint_deviation_leg = -0.5 * torch.sum(torch.abs((joint_pos[4:6] - target_joint_pos[4:6])), dim=0)

            dof_pos_limits_hip = -(joint_pos[:2] - soft_joint_pos_limits[0][0]).clip(max=0.0)
            dof_pos_limits_hip += (joint_pos[:2] - soft_joint_pos_limits[0][1]).clip(min=0.0)
            dof_pos_limits_hip = -2.0 * torch.sum(dof_pos_limits_hip, dim=0)
            dof_pos_limits_shoulder = -(joint_pos[2:4] - soft_joint_pos_limits[1][0]).clip(max=0.0)
            dof_pos_limits_shoulder += (joint_pos[2:4] - soft_joint_pos_limits[1][1]).clip(min=0.0)
            dof_pos_limits_shoulder = -10.0 * torch.sum(dof_pos_limits_shoulder, dim=0)
            dof_pos_limits_leg = -(joint_pos[4:6] - soft_joint_pos_limits[2][0]).clip(max=0.0)
            dof_pos_limits_leg += (joint_pos[4:6] - soft_joint_pos_limits[2][1]).clip(min=0.0)
            dof_pos_limits_leg = -2.0 * torch.sum(dof_pos_limits_leg, dim=0)
            error_vel_xy = -0.05 * torch.sum(torch.square(base_ang_vel[:2]), dim=0)
            error_vel_yaw = -2.0 * torch.square(base_lin_vel[2])

            alive_bonus = 2.0

            total_reward = ((track_lin_vel_xy_exp) + (track_ang_vel_z_exp) +
                            (lin_vel_z_l2) + (anv_vel_xy_l2) +
                            (dof_torques_joint_l2) + (dof_torques_wheels_l2) +
                            (dof_acc_l2) + (action_rate_l2) +
                            (undesired_contact) +
                            (flat_orientation_l2) + (base_target_height) +
                            (joint_deviation_hip) + (joint_deviation_shoulder) + (joint_deviation_leg) +
                            (dof_pos_limits_hip) + (dof_pos_limits_shoulder) + (dof_pos_limits_leg) +
                            (error_vel_xy) + (error_vel_yaw) + (alive_bonus)
            )

            self.episode_sums["track_lin_vel_xy_exp"] += track_lin_vel_xy_exp
            self.episode_sums["track_ang_vel_z_exp"] += track_ang_vel_z_exp
            self.episode_sums["lin_vel_z_l2"] += lin_vel_z_l2
            self.episode_sums["anv_vel_xy_l2"] += anv_vel_xy_l2
            self.episode_sums["dof_torques_joint_l2"] += dof_torques_joint_l2
            self.episode_sums["dof_torques_wheels_l2"] += dof_torques_wheels_l2
            self.episode_sums["dof_acc_l2"] += dof_acc_l2
            self.episode_sums["action_rate_l2"] += action_rate_l2
            self.episode_sums["undesired_contact"] += undesired_contact
            self.episode_sums["flat_orientation_l2"] += flat_orientation_l2
            self.episode_sums["base_target_height"] += base_target_height
            self.episode_sums["joint_deviation_hip"] += joint_deviation_hip
            self.episode_sums["joint_deviation_shoulder"] += joint_deviation_shoulder
            self.episode_sums["joint_deviation_leg"] += joint_deviation_leg
            self.episode_sums["dof_pos_limits_hip"] += dof_pos_limits_hip
            self.episode_sums["dof_pos_limits_shoulder"] += dof_pos_limits_shoulder
            self.episode_sums["dof_pos_limits_leg"] += dof_pos_limits_leg
            self.episode_sums["error_vel_xy"] += error_vel_xy
            self.episode_sums["error_vel_yaw"] += error_vel_yaw
            self.episode_sums["alive_bonus"] += alive_bonus
            self.episode_sums["total_reward"] += total_reward

            # reward = torch.clamp_min(total_reward, 0.0)

        return total_reward.item()

    def _is_done(self):
        contact_forces = self.data.cfrc_ext[1:10]  # External contact forces
        base_contact = contact_forces[0] > 1.0
        shoulder_l_contact = contact_forces[2] > 1.0
        shoulder_r_contact = contact_forces[6] > 1.0
        leg_l_contact = contact_forces[3] > 1.0
        leg_r_contact = contact_forces[7] > 1.0
        contact = shoulder_l_contact.any() or shoulder_r_contact.any() or leg_l_contact.any() or leg_r_contact.any()

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = self.episode_sums[key] / self.step_counter

        # 데이터를 표 형식으로 출력합니다.
        table = PrettyTable()
        table.field_names = ["Index", "Metric", "Value"]
        # 'Epochs'를 전체 행으로 표시
        step_info = f"Steps: {self.step_counter}/{self.sim_step}"
        table.add_row(["", step_info, ""])

        # 구분선 추가
        table.add_row(["-----", "-" * 50, "-----------"])

        index = 0
        for idx, (key, value) in enumerate(self.extras["episode"].items(), start=index + 1):
            table.add_row([idx, key, f"{value.item():.4f}"])

        # 테이블 출력
        print(table)

        return contact

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.previous_states = []
        self.step_counter = 0
        self.action = [0, 0, 0, 0, 0, 0, 0, 0]

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
            "base_target_height": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_hip": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_shoulder": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "joint_deviation_leg": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_hip": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_shoulder": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "dof_pos_limits_leg": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "error_vel_xy": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "error_vel_yaw": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "alive_bonus": torch.zeros(1, dtype=torch.float, requires_grad=False),
            "total_reward": torch.zeros(1, dtype=torch.float, requires_grad=False),
        }

        return self._get_obs(self.action)

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.2561942  # default: 0.2561942
        qpos[3:7] = np.array([1, 0, 0, 0])
        return qpos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # Plotting function
    def plot_logged_data(self, log_data):
        time = log_data['time']

        # Joint names for labeling
        joint_names = [
            'LHJ', 'RHJ', 'LSJ', 'RSJ', 'LLG', 'RLJ'
        ]
        wheel_names = ['LWJ', 'RWJ']

        # Plot desired vs actual joint positions and wheel velocities
        plt.figure("RL/KD Debugger", figsize=(16, 10))

        # Plot joint positions
        for i in range(6):  # Assuming 6 joints
            plt.subplot(4, 2, i + 1)
            plt.plot(time, np.array(log_data['desired_joint_positions'])[:, i],
                     label=f'Des {joint_names[i]} Pos')
            plt.plot(time, np.array(log_data['actual_joint_positions'])[:, i],
                     label=f'Act {joint_names[i]} Pos')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (rad)')
            ticks = np.linspace(-1.0478, 1.0478, num=10)
            plt.ylim(-1.0478, 1.0478)
            plt.yticks(ticks, [f"{tick:.2f}" for tick in ticks])  # Set y-ticks and format them to 2 decimal places
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add a light grid
            plt.legend()

        # Plot wheel velocities
        for i in range(2):  # Assuming 2 wheels
            plt.subplot(4, 2, 6 + i + 1)
            plt.plot(time, np.array(log_data['desired_wheel_velocities'])[:, i],
                     label=f'Des {wheel_names[i]} Vel')
            plt.plot(time, np.array(log_data['actual_wheel_velocities'])[:, i],
                     label=f'Act {wheel_names[i]} Vel')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (rad/s)')
            plt.ylim(-30, 30)  # Fix y-axis limits
            plt.yticks(np.arange(-30, 31, 5))  # Set y-axis ticks with a step of 1
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add a light grid
            plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot joint torques
        plt.figure("Joint Torques", figsize=(16, 10))

        # Plot joint torques
        for i in range(6):  # Assuming 6 joints
            plt.subplot(4, 2, i + 1)
            plt.plot(time, np.array(log_data['joint_torques'])[:, i], label=f'{joint_names[i]} Torque')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.ylim(-25, 25)  # Fix y-axis limits
            plt.yticks(np.arange(-25, 26, 5))  # Set y-axis ticks with a step of 1
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add a light grid
            plt.legend()

        for i in range(2):  # Assuming 6 joints
            plt.subplot(4, 2, 6 + i + 1)
            plt.plot(time, np.array(log_data['wheel_torques'])[:, i], label=f'{wheel_names[i]} Torque')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.ylim(-25, 25)  # Fix y-axis limits
            plt.yticks(np.arange(-25, 26, 5))  # Set y-axis ticks with a step of 1
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add a light grid
            plt.legend()

        plt.tight_layout()
        plt.show()

    def run_mujoco(self, session, input_name, env, sim_hz=200, decimation=4):
        """
        Run the Mujoco simulation at specified frequencies.

        Args:
            session: The ONNX runtime session to run the policy.
            input_name: The input name for the ONNX model.
            env: The environment object.
            sim_hz: The simulation frequency in Hz.
            decimation: The number of simulation steps between each action.
            sim_duration: Duration of the simulation in seconds.

        Returns:
            None
        """
        # Logging structure
        log_data = defaultdict(list)
        dt = (1.0 / sim_hz) * self.frame_skip  # simulation timestep
        push_robot = False  # Flag to check if base velocity has been set
        push_interval = 6  # unit: seconds

        while True:
            obs, _ = env.reset()
            actions = []

            for step in tqdm(range(int(self.sim_step)), desc="Inferencing...", disable=True):
                current_time = step * dt
                # if step % decimation == 0:
                obs_tensor = obs.astype(np.float32)
                # obs_tensor = np.full_like(obs_tensor, -0.1)
                # print("obs tensor:", obs_tensor)
                action = session.run(None, {input_name: obs_tensor[np.newaxis, :]})[0][0]
                action_clipped = np.clip(action, -1, 1)
                actions.append(action_clipped)
                # print("Model action:", action)

                # Set base velocity once within the specified time window
                if push_interval <= current_time and push_robot:
                    base_velocity = np.array([1.0, 0.5, 0.0])
                    env.data.qvel[:3] = base_velocity
                    push_robot = False

                obs, rewards, dones, term, info = env.step(action_clipped)
                env.render()
                # Log the desired and actual values
                log_data['time'].append(current_time)
                log_data['desired_joint_positions'].append(action_clipped[:6] * 1.0)
                log_data['desired_wheel_velocities'].append(action_clipped[6:8] * 25.0)
                log_data['actual_joint_positions'].append(env.data.qpos[[7, 11, 8, 12, 9, 13]].copy())
                log_data['actual_wheel_velocities'].append(env.data.qvel[[9, 13]].copy())
                log_data['joint_torques'].append(env.data.actuator_force[:6].copy())
                log_data['wheel_torques'].append(env.data.actuator_force[6:8].copy())

                if dones or term:
                    break

            if self.plot_log:
                self.plot_logged_data(log_data)
            log_data = defaultdict(list)

            if self.save_data:
                self.save_data_to_csv('dataset/joints/', 'dataset/wheels/')
            if self.save_trajectory:
                self.save_trajectory_to_csv('dataset/trajectory/')
            env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default="policy.onnx", required=True,
                        help='Path to the ONNX model to load.')
    args = parser.parse_args()

    # ONNX 모델 로드
    session = ort.InferenceSession(args.load_model)
    input_name = session.get_inputs()[0].name

    env = FLA_STAND(env_id="FLA_STAND-v0")
    env.render_mode = "human"
    mujoco_thread = threading.Thread(target=env.run_mujoco, args=(session, input_name, env, 200, 4))
    mujoco_thread.start()

    try:
        pygame.init()
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    mujoco_thread.join()
                    exit()
            pygame.display.flip()
            clock.tick(100)

    except KeyboardInterrupt:
        pygame.quit()
        mujoco_thread.join()
