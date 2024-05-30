from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import os
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# from co_gym.utils.utils import fast_clip


class FLA_STAND(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, env_id='FLA_STAND-v0',
                 model_path='assets/flamingo.xml',
                 frame_skip=2):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.frame_skip = frame_skip
        self.id = env_id
        self.obs_dim = 27*3+3  # 모델에 따라 조정
        self.act_dim = 8  # 모델에 따라 조정
        self.action_bound = [-1.0, 1.0]  # 예시: [-1, 1]
        self.max_torques = [18, 18, 18, 3, 18, 18, 18, 3]
        # self.max_torques = [10, 10, 10, 3, 10, 10, 10, 3]

        self.previous_states = []  # 이전 상태를 저장할 리스트
        self.step_counter = 0  # 스텝 카운터 초기화
        self.state_clip = 5

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

    def _get_obs(self):
        # 쿼터니언을 이용한 회전
        quaternion = self.data.qpos[3:7]
        if np.all(quaternion == 0):
            quaternion = np.array([1, 0, 0, 0])  # 유효한 쿼터니언으로 설정

        # 선형 속도와 각속도를 쿼터니언의 역으로 회전
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        lin_vel_rotated = self.quat_rotate_inverse(quaternion, lin_vel)
        ang_vel_rotated = self.quat_rotate_inverse(quaternion, ang_vel)

        # qpos와 qvel의 특정 인덱스에 cos, sin 변환 적용
        qpos_cos_sin = np.concatenate([
            self.data.qpos[[7, 11, 8, 12, 9, 13]],  # self.data.qpos의 특정 인덱스 값들
            [np.sin(self.data.qpos[10]), np.sin(self.data.qpos[14])],  # sin, cos 변환
            [np.cos(self.data.qpos[10]), np.cos(self.data.qpos[14])]  # sin, cos 변환
            # self.data.qpos[7:10],
            # [np.sin(self.data.qpos[10]), np.cos(self.data.qpos[10])],
            # self.data.qpos[11:14],
            # [np.sin(self.data.qpos[14]), np.cos(self.data.qpos[14])],
        ])
        qvel_zigzag = np.concatenate([
            self.data.qpos[[7, 11, 8, 12, 9, 13, 10, 14]],
        ])

        # 현재 상태 관측 값
        r = R.from_quat(quaternion)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        current_state = np.concatenate([
            qpos_cos_sin,  # 관절 위치 (cos, sin 변환 포함)
            qvel_zigzag, # self.data.qvel[6:],  # 관절 속도 (cos, sin 변환 포함)
            lin_vel_rotated,  # base_link의 회전된 x, y, z 속도
            ang_vel_rotated,  # base_link의 회전된 각속도
            [roll, pitch, yaw],  # base_link의 롤, 피치, 요
            self.action,
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
        return np.clip(np.concatenate([np.concatenate(self.previous_states),np.zeros(3)]),-5,5)

    def seed(self, seed):
        return

    def random_action_sample(self) -> np.ndarray:
        return self.action_space.sample()

    def step(self, action):
        # action = fast_clip(action, -1.0, 1.0)
        action = action * self.max_torques  # 조인트별로 최대 토크를 설정
        self.action = action
        self.do_simulation(action, self.frame_skip)
        self.step_counter += 1  # 스텝 카운터 증가
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()
        term = self.step_counter >= 2000  # 1000 스텝 초과 시 done
        return obs, reward, done, term, {}  # done을 두 번 반환하여 정보와 종료 조건을 분리

    def _get_reward(self):
        # 상태 정보 추출
        height = self.data.qpos[2]
        roll, pitch, yaw = self.previous_states[0][-9:-6]  # 각도 정보 추출 (마지막 상태에서)
        lin_vel = self.previous_states[0][-6:-3]  # 선형 속도 정보 추출 (마지막 상태에서)
        ang_vel = self.previous_states[0][-3:]  # 각속도 정보 추출 (마지막 상태에서)

        # 보상 계산
        height_target = 0.4
        height_reward = (height - height_target)  # 높이가 목표 높이에 가까울수록 보상

        # 속도와 각도의 목표값이 0에 가까울수록 보상
        velocity_reward = -np.linalg.norm(lin_vel)  # 선형 속도 보상
        orientation_reward = -np.linalg.norm([yaw])  # 각도 보상
        ang_velocity_reward = -np.linalg.norm(ang_vel)  # 각속도 보상

        # 총 보상 계산
        upright_temp = 0.1;  vel_temp = 0.1;
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
        total_reward += 0.1 # 살아있는 보상
        total_reward /= (4+0.1)

        return total_reward

    def _is_done(self):
        height = self.data.qpos[2]
        return height < 0.18

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.previous_states = []  # 상태 초기화
        self.step_counter = 0  # 스텝 카운터 초기화
        self.action = [0,0,0,0,0,0,0,0]
        return self._get_obs()

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.2461  # 초기 높이 설정
        qpos[3:7] = np.array([1, 0, 0, 0])  # 유효한 초기 쿼터니언 설정
        return qpos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# 강화학습 실행 코드
if __name__ == "__main__":
    # 환경 초기화
    env = FLA_STAND(env_id="FLA_STAND-v0")
    env.render_mode = 'human'
    import torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class CustomNetwork(nn.Module):
        def __init__(self, input_dim, output_dim, input_mean, input_var):
            super(CustomNetwork, self).__init__()
            self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
            self.input_var = torch.tensor(input_var, dtype=torch.float32)

            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, output_dim)

            self.elu = nn.ELU()

        def forward(self, x):
            # Normalize input
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            # 입력 텐서와 동일한 디바이스로 이동
            device = x.device
            input_mean = self.input_mean.to(device)
            input_var = self.input_var.to(device)
            x = (x - input_mean) / torch.sqrt(input_var + 1e-8)

            x = self.elu(self.fc1(x))
            x = self.elu(self.fc2(x))
            x = self.elu(self.fc3(x))
            x = self.fc4(x)
            return x
    checkpoint = torch.load('Flamingo.pth')
    checkpoint_keys = checkpoint['model'].keys()
    weights = []; biass = []; input_mean = None; input_var = None
    for k in checkpoint_keys:
        v = checkpoint['model'][k]
        if k.find('running_mean_std.running_mean') == 0:
            input_mean = v
        elif k.find('running_mean_std.running_var') == 0:
            input_var = v
        elif k.find('a2c_network.actor_mlp.0.weight') == 0:
            weights.append(v)
        elif k.find('a2c_network.actor_mlp.2.weight') == 0:
            weights.append(v)
        elif k.find('a2c_network.actor_mlp.4.weight') == 0:
            weights.append(v)
        elif k.find('a2c_network.mu.weight') == 0:
            weights.append(v)
        elif k.find('a2c_network.actor_mlp.0.bias') == 0:
            biass.append(v)
        elif k.find('a2c_network.actor_mlp.2.bias') == 0:
            biass.append(v)
        elif k.find('a2c_network.actor_mlp.4.bias') == 0:
            biass.append(v)
        elif k.find('a2c_network.mu.bias') == 0:
            biass.append(v)
        else:
            pass

    input_dim = 84
    model = CustomNetwork(input_dim, 8, input_mean, input_var)
    with torch.no_grad():
        model.fc1.weight.copy_(weights[0])
        model.fc1.bias.copy_(biass[0])
        model.fc2.weight.copy_(weights[1])
        model.fc2.bias.copy_(biass[1])
        model.fc3.weight.copy_(weights[2])
        model.fc3.bias.copy_(biass[2])
        model.fc4.weight.copy_(weights[3])
        model.fc4.bias.copy_(biass[3])

    # 모델을 평가 모드로 설정 (선택 사항, 특히 테스트 시)
    model.eval()
    input_data = torch.randn(1, input_dim)  #
    output = model(input_data)
    print(output)

    # 정책 실행 (선택 사항)
    while True:
        obs, _ = env.reset()
        actions = []
        for _ in range(1000):
            try:
                action = model(obs).detach().numpy()
            except:
                print('Random Action!')
                action = env.action_space.sample()
            action = np.clip(action, -1, 1)
            obs, rewards, dones, term, info = env.step(action)
            env.render()

            ##
            actions.append(action)
            if dones or term:
                break
        # actions = np.array(actions)
        # time = np.arange(actions.shape[0])
        # fig, axs = plt.subplots(4, 2, figsize=(15, 10))
        # for i in range(actions.shape[1]):
        #     row = i // 2
        #     col = i % 2
        #     axs[row, col].plot(time, actions[:, i])
        #     axs[row, col].set_title(f'Column {i + 1}')
        # plt.show()

        env.reset()