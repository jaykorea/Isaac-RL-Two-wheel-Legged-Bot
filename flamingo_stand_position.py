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
# import torch

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
                 frame_skip=1, render_mode='human'):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.total_steps = 1500  # 최대 스텝 수
        self.id = env_id
        self.obs_dim = 31 * 3 + 3  # 모델에 따라 조정
        self.act_dim = 8  # 모델에 따라 조정
        self.action_scaler = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 25.0, 25.0]
        self.filtered_action = None

        self.previous_states = []  # 이전 상태를 저장할 리스트
        self.step_counter = 0  # 스텝 카운터 초기화
        self.state_clip = 25
        self.commands = np.zeros(3)
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
            v,  # base lin vel
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

        # 상태를 결합하여 반환
        return np.clip(np.concatenate([np.concatenate(self.previous_states), self.commands]), -self.state_clip, self.state_clip)

    def seed(self, seed):
        return

    def random_action_sample(self) -> np.ndarray:
        return self.action_space.sample()

    def low_pass_filter(self, action, alpha=0.05):
        if self.filtered_action is None:
            self.filtered_action = action
        else:
            self.filtered_action = alpha * action + (1 - alpha) * self.filtered_action
        return self.filtered_action

    def step(self, action):
        obs = self._get_obs(action)
        # Apply low-pass filter to the action
        filtered_action = self.low_pass_filter(action)
        action_scaled = filtered_action * self.action_scaler

        # # Extract joint positions and velocities from observation
        # pos_hip = obs[0:2]
        # pos_shoulder = obs[2:4]
        # pos_leg = obs[4:6]
        # pos_wheels = obs[6:8]

        # vel_hip = obs[8:10]
        # vel_shoulder = obs[10:12]
        # vel_leg = obs[12:14]
        # vel_wheels = obs[14:16]

        # # PD control parameters from Isaac Orbit configuration
        # kp_j = 35.0
        # kd_j = 0.0
        # kp_w = 0.0
        # kd_w = 1.0

        # hip_action_scaled = action[0:2] * self.action_scaler[0:2]
        # shoulder_action_scaled = action[2:4] * self.action_scaler[2:4]
        # leg_action_scaled = action[4:6] * self.action_scaler[4:6]
        # wheel_action_scaled = action[6:8] * self.action_scaler[6:8]

        # # Calculate torques for each joint using PD controller
        # hip_action = np.clip(self.pd_controller(kp_j, hip_action_scaled, pos_hip, kd_j, 0.0, vel_hip), -23.0, 23.0)
        # shoulder_action = np.clip(self.pd_controller(kp_j, shoulder_action_scaled, pos_shoulder, kd_j, 0.0, vel_shoulder), -23.0, 23.0)
        # leg_action = np.clip(self.pd_controller(kp_j, leg_action_scaled, pos_leg, kd_j, 0.0, vel_leg), -23.0, 23.0)
        # wheel_action = np.clip(self.pd_controller(kp_w, 0.0, pos_wheels, kd_w, wheel_action_scaled, vel_wheels), -6.0, 6.0)

        # # Combine all actions
        # action_scaled = np.concatenate([hip_action, shoulder_action, leg_action, wheel_action])

        # Apply the actions to the simulation
        self.do_simulation(action_scaled, self.frame_skip)
        self.step_counter += 1

        # Get new observations, rewards, done, and info
        obs = self._get_obs(action)
        height = self.data.qpos[2]
        reward = self._get_reward()
        done = self._is_done()
        term = self.step_counter >= self.total_steps
        return obs, reward, done, term, {}

    def _get_reward(self):
        height = self.data.qpos[2]
        roll, pitch, yaw = self.previous_states[0][-9:-6]
        lin_vel = self.previous_states[0][-6:-3]
        ang_vel = self.previous_states[0][-3:]

        height_target = 0.4
        height_reward = (height - height_target)

        velocity_reward = -np.linalg.norm(lin_vel)
        orientation_reward = -np.linalg.norm([yaw])
        ang_velocity_reward = -np.linalg.norm(ang_vel)

        upright_temp = 0.1; vel_temp = 0.1
        upright_reward_transformed = np.where(height_reward > 0,
                                                 np.exp(- height_reward / upright_temp),-3)
        command_x_reward_transformed = np.exp(velocity_reward / vel_temp)
        command_y_reward_transformed = np.exp(orientation_reward / vel_temp)
        command_angvel_reward_transformed = np.exp(ang_velocity_reward / vel_temp)
        total_reward = (upright_reward_transformed
                        + command_x_reward_transformed
                        + command_y_reward_transformed
                        + command_angvel_reward_transformed)
        total_reward = np.clip(total_reward, 0, 1e6)
        total_reward += 0.1
        total_reward /= (4+0.1)

        return total_reward

    def _is_done(self):
        height = self.data.qpos[2]
        return height < 0.1

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.previous_states = []
        self.step_counter = 0
        self.action = [0,0,0,0,0,0,0,0]
        return self._get_obs(self.action)

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.2471942
        qpos[3:7] = np.array([1, 0, 0, 0])
        return qpos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # def run_mujoco(self, session, input_name, env):
    #     while True:
    #         obs, _ = env.reset()
    #         actions = []
    #         for _ in range(1000):
    #             try:
    #                 obs_tensor = obs.astype(np.float32)
    #                 action = session.run(None, {input_name: obs_tensor[np.newaxis, :]})[0][0]
    #                 # print("Model action:", action)
    #             except Exception as e:
    #                 print(f'Error: {e}')
    #                 print('Random Action!')
    #                 action = env.action_space.sample()
    #                 raise
    #             action_clipped = np.clip(action, -1, 1)
    #             # print("Clipped and reordered action:", action)

    #             obs, rewards, dones, term, info = env.step(action_clipped)
    #             # print("Step results - Observation:", obs, "Rewards:", rewards, "Dones:", dones, "Term:", term)
    #             env.render()

    #             actions.append(action_clipped)
    #             if dones or term:
    #                 break

    #         env.reset()

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
            plt.legend()

        plt.tight_layout()
        plt.show()

    def run_mujoco(self, session, input_name, env, sim_hz=200, decimation=2):
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
        dt = 1.0 / sim_hz  # simulation timestep

        while True:
            obs, _ = env.reset()
            actions = []

            for step in tqdm(range(self.total_steps), desc="Inferencing..."):

                if step % decimation == 0:
                    obs_tensor = obs.astype(np.float32)
                    action = session.run(None, {input_name: obs_tensor[np.newaxis, :]})[0][0]
                    action_clipped = np.clip(action, -1, 1)
                    actions.append(action_clipped)

                obs, rewards, dones, term, info = env.step(action_clipped)
                env.render()

                # Log the desired and actual values
                log_data['time'].append(step * dt)
                log_data['desired_joint_positions'].append(action_clipped[:6] * 2.0)
                log_data['desired_wheel_velocities'].append(action_clipped[6:8] * 25.0)
                log_data['actual_joint_positions'].append(env.data.qpos[[7, 11, 8, 12, 9, 13]].copy())
                log_data['actual_wheel_velocities'].append(env.data.qvel[[9, 13]].copy())

                if dones or term:
                    break

            self.plot_logged_data(log_data)
            log_data = defaultdict(list)
            env.reset()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default="policy.onnx", required=True,
                        help='Path to the ONNX model to load.')
    args = parser.parse_args()

    # ONNX 모델 로드
    session = ort.InferenceSession(args.load_model)
    input_name = session.get_inputs()[0].name

    env = FLA_STAND(env_id="FLA_STAND-v0")
    env.render_mode = "human"
    # env.run_mujoco(session, input_name, env)
    env.run_mujoco(session, input_name, env, sim_hz=200, decimation=2)
