from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import os
import torch
import mujoco
import threading
import argparse
import onnxruntime as ort

from utils.memory_utils import MemoryUtils
from utils.pygame_utils import PygameUtils
from utils.math_utils import MathUtils
from utils.log_utils import LogUtils
from utils.extra_utils import ExtraUtils

from manager.noise_manager import AdditiveNoiseManager
from manager.inertia_manager import InertiaManager
from manager.reward_manager import RewardManager
from manager.push_manager import PushManager
from manager.control_manager import ControlManager


class FLA_STAND(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(self, env_id='FLA_STAND-v0', model_path='./assets/flamingo_pos_vel.xml', frame_skip=4, render_mode='human'):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None
        self.dt_ = 0.005
        self.sim_duration = 60.0
        self.sim_step = (self.sim_duration / self.dt_) / self.frame_skip
        self.id = env_id
        self.obs_dim = 28 * 3 + 4
        self.act_dim = 8
        self.previous_states = []
        self.state_clip = np.inf
        self.action_scaler = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 25.0, 25.0]

        self.step_counter = 0

        self.save_data = False
        self.save_trajectory = False
        self.plot_log = True
        self.mem_save = False

        self.randomize_sensor = False
        self.randomize_inertia = False
        self.randomize_initial_state = True
        self.push_robot = False

        self.Unoise = AdditiveNoiseManager(noise_type="uniform")
        self.Gnoise = AdditiveNoiseManager(noise_type="gaussian")

        self.inertia_manager = InertiaManager(self.model_path)
        self.specific_bodies_noise = {
            "base_link": {"mass_mean": 0.0, "mass_std": 0.5, "inertia_mean": 0.0, "inertia_std": 0.0},
            "left_hip_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "right_hip_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "left_shoulder_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "right_shoulder_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "left_leg_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "right_leg_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "left_wheel_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
            "right_wheel_link": {"mass_mean": 0.0, "mass_std": 0.1, "inertia_mean": 0.0, "inertia_std": 0.0},
        }

        self.reward_manager = RewardManager()
        self.control_manager = ControlManager()
        self.push_manager = PushManager()

        self.mem_utils = MemoryUtils()
        self.pygame_utils = PygameUtils()
        self.log_utils = LogUtils()
        self.extra_utils = ExtraUtils()

        utils.EzPickle.__init__(self)

        if self.render_mode is None:
            MujocoEnv.__init__(
                self,
                model_path=self.model_path,
                frame_skip=self.frame_skip,
                observation_space=Box(
                    low=-self.state_clip,
                    high=self.state_clip,
                    shape=(self.obs_dim,),
                    dtype=np.float32,
                ),  # 3 프레임을 쌓음
            )
        else:
            MujocoEnv.__init__(
                self,
                model_path=self.model_path,
                frame_skip=self.frame_skip,
                observation_space=Box(
                    low=-self.state_clip,
                    high=self.state_clip,
                    shape=(self.obs_dim,),
                    dtype=np.float32,
                ),
                render_mode=self.render_mode,
            )

    def _get_obs(self, action):
        q = self.data.qpos[[7, 11, 8, 12, 9, 13]]
        qd = self.data.qvel[[6, 10, 7, 11, 8, 12, 9, 13]]
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])

        omega = self.data.sensor('angular-velocity').data.astype(np.double)
        euler = MathUtils.quaternion_to_euler_array(quat)

        if self.randomize_sensor:
            q = self.Unoise.apply(q, low=-0.04, high=0.04)
            qd = self.Unoise.apply(qd, low=-1.5, high=1.5)
            omega = self.Unoise.apply(omega, low=-0.2, high=0.2)
            euler = self.Unoise.apply(euler, low=-0.1, high=0.1)

        current_state = np.concatenate([q, qd, omega, euler, action])

        if len(self.previous_states) < 2:
            self.previous_states.append(current_state)
            self.previous_states = self.previous_states + [np.zeros_like(current_state)] * (3 - len(self.previous_states))
        else:
            self.previous_states.pop(-1)
            self.previous_states.insert(0, current_state)

        self.pygame_utils.update_commands(automation=self.pygame_utils.automation_command)

        return np.clip(np.concatenate([np.concatenate(self.previous_states), self.pygame_utils.commands]), -self.state_clip, self.state_clip)

    def seed(self, seed):
        return

    def random_action_sample(self) -> np.ndarray:
        return self.action_space.sample()

    def step(self, action):
        obs = self._get_obs(action)
        filtered_action = self.control_manager.low_pass_filter(action)
        action_scaled = filtered_action * self.action_scaler
        self.do_simulation(action_scaled, self.frame_skip)
        self.step_counter += 1
        obs = self._get_obs(filtered_action)

        if self.save_data or self.save_trajectory:
            self.log_utils.collect_and_save_data(obs, action_scaled, self.data.actuator_force)

        reward, info = self._get_reward(obs)

        term = self._is_done()
        trun = self.step_counter >= self.sim_step

        self.log_utils.plot_table(self.extra_utils.get_episode_sums(), self.step_counter, self.sim_step)

        return obs, reward, term, trun, info

    def _get_reward(self, obs):
        """_summary_
            obs_buf = info['obs']  # torch.Tensor
            current_height = info['current_height']  # torch.Tensor
            joint_acc = info['joint_acc']  # torch.Tensor
            contact_forces = info['contact_forces']  # torch.Tensor
            actuator_force = info['actuator_force']  # torch.Tensor
        """

        self.reward_manager.get_privileged_observations(self.data, self._is_done())
        self.reward_manager.get_observations(obs, self.step_counter, self.sim_step)

        track_lin_vel_xy_exp = self.reward_manager.track_lin_vel_xy_exp(weight=2.0)
        track_ang_vel_z_exp = self.reward_manager.track_ang_vel_z_exp(weight=1.0)
        lin_vel_z_l2 = self.reward_manager.lin_vel_z_l2(weight=-2.0)
        anv_vel_xy_l2 = self.reward_manager.anv_vel_xy_l2(weight=-0.05)
        dof_torques_joint_l2 = self.reward_manager.dof_torques_joint_l2(weight=-1.0e-5)
        dof_torques_wheels_l2 = self.reward_manager.dof_torques_wheels_l2(weight=-1.0e-5)
        dof_acc_l2 = self.reward_manager.dof_acc_l2(weight=-3.75e-7)
        action_rate_l2 = self.reward_manager.action_rate_l2(weight=-0.015)
        undesired_contact = self.reward_manager.undesired_contact(weight=-1.0)
        flat_orientation_l2 = self.reward_manager.flat_orientation_l2(weight=-0.5)
        base_target_range_height = self.reward_manager.base_target_range_height(weight=15.0)
        joint_deviation_hip = self.reward_manager.joint_deviation('left_hip', weight=-5.0) + self.reward_manager.joint_deviation('right_hip', weight=-5.0)
        joint_align_shoulder = self.reward_manager.joint_align('left_shoulder', 'right_shoulder', weight=-2.0)
        joint_align_leg = self.reward_manager.joint_align('left_leg', 'right_leg', weight=-2.0)
        dof_pos_limits_hip = self.reward_manager.dof_pos_limits('left_hip', 'right_hip', weight=-2.0)
        dof_pos_limits_shoulder = self.reward_manager.dof_pos_limits('left_shoulder', 'right_shoulder', weight=-10.0)
        dof_pos_limits_leg = self.reward_manager.dof_pos_limits('left_leg', 'right_leg', weight=-2.0)
        error_vel_xy = self.reward_manager.error_vel_xy(weight=-0.05)
        error_vel_yaw = self.reward_manager.error_vel_yaw(weight=-2.0)
        is_terminated = self.reward_manager.is_terminated(weight=-200.0)

        total_reward = (track_lin_vel_xy_exp + track_ang_vel_z_exp +
                        lin_vel_z_l2 + anv_vel_xy_l2 +
                        dof_torques_joint_l2 + dof_torques_wheels_l2 +
                        dof_acc_l2 + action_rate_l2 +
                        undesired_contact +
                        flat_orientation_l2 + base_target_range_height +
                        joint_deviation_hip + joint_align_shoulder + joint_align_leg +
                        dof_pos_limits_hip + dof_pos_limits_shoulder + dof_pos_limits_leg +
                        error_vel_xy + error_vel_yaw + is_terminated)

        self.extra_utils.update_episode_sums(
            track_lin_vel_xy_exp=track_lin_vel_xy_exp,
            track_ang_vel_z_exp=track_ang_vel_z_exp,
            lin_vel_z_l2=lin_vel_z_l2,
            anv_vel_xy_l2=anv_vel_xy_l2,
            dof_torques_joint_l2=dof_torques_joint_l2,
            dof_torques_wheels_l2=dof_torques_wheels_l2,
            dof_acc_l2=dof_acc_l2,
            action_rate_l2=action_rate_l2,
            undesired_contact=undesired_contact,
            flat_orientation_l2=flat_orientation_l2,
            base_target_range_height=base_target_range_height,
            joint_deviation_hip=joint_deviation_hip,
            joint_align_shoulder=joint_align_shoulder,
            joint_align_leg=joint_align_leg,
            dof_pos_limits_hip=dof_pos_limits_hip,
            dof_pos_limits_shoulder=dof_pos_limits_shoulder,
            dof_pos_limits_leg=dof_pos_limits_leg,
            error_vel_xy=error_vel_xy,
            error_vel_yaw=error_vel_yaw,
            is_terminated=is_terminated,
            total_reward=total_reward
        )

        info = {}
        priv_obs = self.reward_manager.req_privileged_observations()
        info['obs'] = priv_obs['obs_buf']
        info['current_height'] = priv_obs['current_height']
        info['joint_acc'] = priv_obs['joint_acc']
        info['contact_forces'] = priv_obs['contact_forces']
        info['actuator_forces'] = priv_obs['actuator_forces']
        # reward = torch.clamp_min(total_reward, 0.0)

        return total_reward.item(), info

    def _is_done(self):
        contact_forces = self.data.cfrc_ext[1:10]  # External contact forces
        base_contact = contact_forces[0] > 1.0
        hip_l_contact = contact_forces[1] > 1.0
        hip_r_contact = contact_forces[5] > 1.0
        shoulder_l_contact = contact_forces[2] > 1.0
        shoulder_r_contact = contact_forces[6] > 1.0
        leg_l_contact = contact_forces[3] > 1.0
        leg_r_contact = contact_forces[7] > 1.0
        contact = base_contact.any() or hip_l_contact.any() or hip_r_contact.any() or shoulder_l_contact.any() or shoulder_r_contact.any() or leg_l_contact.any() or leg_r_contact.any()

        return contact

    def reset_model(self):
        if self.randomize_inertia:
            randomized_model_path = self.inertia_manager.randomize_inertial(self.specific_bodies_noise)
            print(f"Loading model from {randomized_model_path}")
            self.fullpath = randomized_model_path
            self.model = mujoco.MjModel.from_xml_path(self.fullpath)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_resetData(self.model, self.data)

            # Reinitialize the MujocoRenderer with the new model
            # self.mujoco_renderer.close()
            # self.mujoco_renderer = MujocoRenderer(self.model, self.data)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.previous_states = []
        self.step_counter = 0
        self.action = [0, 0, 0, 0, 0, 0, 0, 0]

        self.extra_utils.reset_episode_sums()

        return self._get_obs(self.action)

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.2561942
        qpos[3:7] = np.array([1, 0, 0, 0])
        qpos[7:14] = np.array([0, 0, 0, 0, 0, 0, 0])
        if self.randomize_initial_state:
            qpos[7:14] = self.Unoise.apply(qpos[7:14], low=-0.2, high=0.2)
        return qpos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def run_mujoco(self, session, input_name, env, sim_hz=200):
        dt = (1.0 / sim_hz) * self.frame_skip

        while True:
            obs, _ = env.reset()
            self.push_manager.reset()
            for step in range(int(self.sim_step)):
                current_time = step * dt
                obs_tensor = obs.astype(np.float32)
                action = session.run(None, {input_name: obs_tensor[np.newaxis, :]})[0][0]
                action_clipped = np.clip(action, -1, 1)

                if self.push_robot:
                    self.push_manager.apply_push(env, np.array([0.5, 0.5, 0.0]), current_time, push_vel_range=(-1.0, 1.0), time_range=(5.0, 15.0), random=True)

                prev_obs = obs
                obs, rewards, term, trun, info = env.step(action_clipped)

                if self.mem_save:
                    self.mem_utils.store(prev_obs, action_clipped, rewards, obs, term, info)

                if self.plot_log or self.save_data or self.save_trajectory:
                    self.log_utils.log_step_data(current_time, action_clipped, env)

                env.render()

                if term or trun:
                    break

            if self.plot_log:
                self.log_utils.plot_logged_data()
                self.log_utils.reset_log_data()

            if self.save_data:
                self.log_utils.save_data_to_csv('dataset/joints/', 'dataset/wheels/')
            if self.save_trajectory:
                self.log_utils.save_trajectory_to_csv('dataset/trajectory/')

            if self.mem_save:
                self.mem_utils.load()
                self.mem_utils.reset()

                try:
                    with open("raw_memory20.pkl", "wb") as fw:
                        pickle.dump(self.mem_utils, fw)
                except KeyboardInterrupt:
                    print("Process interrupted. File not saved.")

            env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default="policy.onnx", required=True, help='Path to the ONNX model to load.')
    args = parser.parse_args()

    session = ort.InferenceSession(args.load_model)
    input_name = session.get_inputs()[0].name

    env = FLA_STAND(env_id="FLA_STAND-v0")
    env.render_mode = "human"
    mujoco_thread = threading.Thread(target=env.run_mujoco, args=(session, input_name, env, 200))
    mujoco_thread.start()

    try:
        env.pygame_utils.run_gui(mujoco_thread)
    except KeyboardInterrupt:
        env.pygame_utils.close_gui()
        mujoco_thread.join()
