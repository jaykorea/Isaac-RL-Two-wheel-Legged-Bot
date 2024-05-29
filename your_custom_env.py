from co_gym.envs.base import BaseEnv
import numpy as np
from typing import (Tuple, SupportsFloat)


class FLA_OTHER(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 model_path='./cocel_customs/assets/urdf/postech/urdf/flamingo.xml',
                 frame_skip=4):
        super(YourCustomEnv, self).__init__()

        # *** Set essential properties below ***
        self.id = "FlamingoStand-v0"  # 예시: Ant-v4
        self.obs_dim = 18  # 모델에 따라 조정
        self.act_dim = 8  # 모델에 따라 조정
        self.action_bound = [-1.0, 1.0]  # 예시: [-1, 1]

        self.model_path = model_path
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(self.model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_bound[0], high=self.action_bound[1], shape=(self.act_dim,), dtype=np.float32)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self) -> Tuple[np.ndarray, dict]:
        self.sim.reset()
        self.sim.data.qpos[:] = self.initial_qpos()
        self.sim.data.qvel[:] = 0
        observation = self._get_obs()
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        action = action * [10, 10, 10, 6, 10, 10, 10, 6]  # 조인트별로 최대 토크를 설정
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, False, info

    def random_action_sample(self) -> np.ndarray:
        return self.action_space.sample()

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 1.0  # 초기 높이 설정
        return qpos

    def _get_reward(self):
        # 바닥에 충돌하지 않고 일어선 상태를 유지하는 보상 함수
        height = self.sim.data.qpos[2]
        upright_reward = height  # 높이에 비례한 보상
        fall_penalty = -1 if height < 0.5 else 0
        return upright_reward + fall_penalty

    def _is_done(self):
        # 바닥에 충돌하거나 일정 범위를 벗어나면 에피소드 종료
        height = self.sim.data.qpos[2]
        return height < 0.5
