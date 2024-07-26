import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from collections import defaultdict
import os

from .math_utils import MathUtils


class LogUtils:
    def __init__(self):
        self.joint_data = []
        self.wheel_data = []
        self.trajectory_data = []
        self.log_data = defaultdict(list)
        self.math_utils = MathUtils()

    def collect_and_save_data(self, obs, action_scaled, actuator_forces):
        joint_positions = obs[0:6]
        joint_velocities = obs[6:12]
        target_positions = action_scaled[0:6]
        torques = actuator_forces[0:6]

        joint_names = ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_leg", "right_leg"]
        for i, joint_name in enumerate(joint_names):
            self.joint_data.append({
                'joint': joint_name,
                'target_position': target_positions[i].item(),
                'current_position': joint_positions[i].item(),
                'velocity': joint_velocities[i].item(),
                'torque': torques[i].item()
            })

        wheel_positions = [self.math_utils.wrap_to_2pi(obs[10]), self.math_utils.wrap_to_2pi(obs[13])]
        wheel_velocities = obs[12:14]
        target_wheel_velocities = action_scaled[6:8]
        wheel_torques = actuator_forces[6:8]

        wheel_names = ["left_wheel", "right_wheel"]
        for i, wheel_name in enumerate(wheel_names):
            self.wheel_data.append({
                'wheel': wheel_name,
                'target_velocity': target_wheel_velocities[i].item(),
                'current_velocity': wheel_velocities[i].item(),
                'sin_position': np.sin(wheel_positions[i]),
                'cos_position': np.cos(wheel_positions[i]),
                'torque': wheel_torques[i].item()
            })

        data_entry = {}
        for i, joint_name in enumerate(joint_names):
            data_entry[f'{joint_name}_current_position'] = joint_positions[i].item()
            data_entry[f'{joint_name}_target_position'] = target_positions[i].item()

        for i, wheel_name in enumerate(wheel_names):
            data_entry[f'{wheel_name}_current_velocity'] = wheel_velocities[i].item()
            data_entry[f'{wheel_name}_target_velocity'] = target_wheel_velocities[i].item()

        self.trajectory_data.append(data_entry)

    def save_data_to_csv(self, joint_csv_dir, wheel_csv_dir):
        if not os.path.exists(joint_csv_dir):
            os.makedirs(joint_csv_dir)

        if not os.path.exists(wheel_csv_dir):
            os.makedirs(wheel_csv_dir)

        def get_unique_filename(directory, filename):
            base, ext = os.path.splitext(filename)
            counter = 1
            new_filename = filename
            while os.path.exists(os.path.join(directory, new_filename)):
                new_filename = f"{base}_{counter}{ext}"
                counter += 1
            return os.path.join(directory, new_filename)

        joint_df = pd.DataFrame(self.joint_data)
        joint_names = joint_df['joint'].unique()
        for joint_name in joint_names:
            joint_data = joint_df[joint_df['joint'] == joint_name]
            filename = os.path.join(joint_csv_dir, f"{joint_name}_joint_data.csv")
            joint_data.to_csv(filename, mode='a' if os.path.exists(filename) else 'w', header=not os.path.exists(filename), index=False)

        wheel_df = pd.DataFrame(self.wheel_data)
        wheel_names = wheel_df['wheel'].unique()
        for wheel_name in wheel_names:
            wheel_data = wheel_df[wheel_df['wheel'] == wheel_name]
            filename = os.path.join(wheel_csv_dir, f"{wheel_name}_joint_data.csv")
            wheel_data.to_csv(filename, mode='a' if os.path.exists(filename) else 'w', header=not os.path.exists(filename), index=False)

    def save_trajectory_to_csv(self, trajectory_csv_dir):
        if not os.path.exists(trajectory_csv_dir):
            os.makedirs(trajectory_csv_dir)

        def get_unique_filename(directory, base_filename):
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
        filename = get_unique_filename(trajectory_csv_dir, base_filename) if os.path.exists(filename) else filename

        trajectory_df.to_csv(filename, mode='w', header=True, index=False)

    def plot_table(self, episode_sums, step_counter, sim_step):
        extras = {"episode": {}}
        for key in episode_sums.keys():
            extras["episode"]['rew_' + key] = episode_sums[key] / step_counter

        table = PrettyTable()
        table.field_names = ["Index", "Metric", "Value"]
        step_info = f"Steps: {step_counter}/{sim_step}"
        table.add_row(["", step_info, ""])
        table.add_row(["-----", "-" * 50, "-----------"])

        index = 0
        for idx, (key, value) in enumerate(extras["episode"].items(), start=index + 1):
            table.add_row([idx, key, f"{value.item():.4f}"])

        print(table)

    def plot_logged_data(self):
        time = self.log_data['time']
        joint_names = ['LHJ', 'RHJ', 'LSJ', 'RSJ', 'LLG', 'RLJ']
        wheel_names = ['LWJ', 'RWJ']

        plt.figure("RL/KD Debugger", figsize=(16, 10))
        for i in range(6):
            plt.subplot(4, 2, i + 1)
            plt.plot(time, np.array(self.log_data['desired_joint_positions'])[:, i], label=f'Des {joint_names[i]} Pos')
            plt.plot(time, np.array(self.log_data['actual_joint_positions'])[:, i], label=f'Act {joint_names[i]} Pos')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (rad)')
            ticks = np.linspace(-1.0478, 1.0478, num=10)
            plt.ylim(-1.0478, 1.0478)
            plt.yticks(ticks, [f"{tick:.2f}" for tick in ticks])
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            plt.legend()

        for i in range(2):
            plt.subplot(4, 2, 6 + i + 1)
            plt.plot(time, np.array(self.log_data['desired_wheel_velocities'])[:, i], label=f'Des {wheel_names[i]} Vel')
            plt.plot(time, np.array(self.log_data['actual_wheel_velocities'])[:, i], label=f'Act {wheel_names[i]} Vel')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (rad/s)')
            plt.ylim(-30, 30)
            plt.yticks(np.arange(-30, 31, 5))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            plt.legend()

        plt.tight_layout()
        plt.show()

        plt.figure("Joint Torques", figsize=(16, 10))
        for i in range(6):
            plt.subplot(4, 2, i + 1)
            plt.plot(time, np.array(self.log_data['joint_torques'])[:, i], label=f'{joint_names[i]} Torque')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.ylim(-25, 25)
            plt.yticks(np.arange(-25, 26, 5))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            plt.legend()

        for i in range(2):
            plt.subplot(4, 2, 6 + i + 1)
            plt.plot(time, np.array(self.log_data['wheel_torques'])[:, i], label=f'{wheel_names[i]} Torque')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.ylim(-25, 25)
            plt.yticks(np.arange(-25, 26, 5))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def log_step_data(self, current_time, action_clipped, env):
        self.log_data['time'].append(current_time)
        self.log_data['desired_joint_positions'].append(action_clipped[:6] * 1.0)
        self.log_data['desired_wheel_velocities'].append(action_clipped[6:8] * 25.0)
        self.log_data['actual_joint_positions'].append(env.data.qpos[[7, 11, 8, 12, 9, 13]].copy())
        self.log_data['actual_wheel_velocities'].append(env.data.qvel[[9, 13]].copy())
        self.log_data['joint_torques'].append(env.data.actuator_force[:6].copy())
        self.log_data['wheel_torques'].append(env.data.actuator_force[6:8].copy())

    def reset_log_data(self):
        self.log_data = defaultdict(list)
