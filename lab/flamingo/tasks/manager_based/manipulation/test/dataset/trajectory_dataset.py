from typing import Dict, Tuple, List, Union
from tqdm import tqdm

import torch
import zarr
import numpy as np

# 레퍼런스 코드 및 기존 코드의 유틸리티들을 가져왔다고 가정합니다.
# 실제 사용 시에는 이 클래스들이 import 가능한 상태여야 합니다.
from .base_dataset import BaseDataset
from .utils import GaussianNormalizer, dict_apply, WelfordNormalizer

import wandb

# ======================= 수정된 최종 데이터셋 클래스 =======================
class SingeHorizonTrajectoryDataset(BaseDataset): # <-- 이 부분을 요청대로 수정했습니다.
    """
    Zarr 기반의 시퀀스 데이터셋.
    - D4RLMuJoCoDataset의 시퀀스 샘플링 로직과
    - 기존 TrajectoryDataset의 Zarr 직접 접근 및 메모리 효율적 Normalizer를 결합.
    - 최상위 BaseDataset을 직접 상속합니다.
    """
    def __init__(self,
                 zarr_path: str,
                 horizon: int,
                 discount: float = 0.99,
                 terminal_penalty: float = -100.0,
                 max_episode_lengths: int = 1000
                 ):
        super().__init__()
        
        self.horizon = horizon
        self.store = zarr.open(zarr_path, 'r')

        observations = self.store['observations']
        actions = self.store['actions']
        rewards = self.store['rewards']
        terminals = self.store['dones']
        timeouts = self.store.get('timeouts', np.zeros_like(terminals, dtype=bool))

        self.normalizer = GaussianNormalizer(observations[:])
        
        self.o_dim = observations.shape[1]
        self.a_dim = actions.shape[1]
        
        # 전체 에피소드의 개수 계산
        total_episodes = np.sum(np.logical_or(terminals, timeouts)) + 1
        # (총 에피소드수, 최대 에피소드 길이, 데이터 차원) 배열을 가지는 메모리 생성.
        self.seq_obs = np.zeros((total_episodes, max_episode_lengths, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((total_episodes, max_episode_lengths, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((total_episodes, max_episode_lengths, 1), dtype=np.float32)
        self.seq_val = np.zeros((total_episodes, max_episode_lengths, 1), dtype=np.float32)

        self.indices = []
        path_idx, ptr = 0, 0
        for i in range(len(terminals)):
            # Zarr의 원본 데이터를 처음부터 끝까지 순회하며 에피소드의 경계(terminals 또는 timeouts) 찾기.
            if terminals[i] or timeouts[i] or i == len(terminals) - 1:
                path_len = i - ptr + 1
                
                # 찾은 에피소드의 obs, action, reward 저장.
                ep_obs = observations[ptr:i + 1]
                ep_act = actions[ptr:i + 1]
                ep_rew = rewards[ptr:i + 1].copy()
                
                # 에피소드의 마지막이 terminal 이면 패널티
                if terminals[i] and not timeouts[i]:
                    ep_rew[-1] = terminal_penalty
                
                # 에피소드 데이터 정규화 
                self.seq_obs[path_idx, :path_len] = self.normalizer.normalize(ep_obs)
                self.seq_act[path_idx, :path_len] = ep_act
                self.seq_rew[path_idx, :path_len] = ep_rew[:, None]

                # 샘플링 가능한 모든 시퀸스 인덱스 미리 계산.
                max_start = min(path_len - 1, max_episode_lengths - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1
        
        # 누적 미래 보상 계산.
        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in range(max_episode_lengths - 2, -1, -1):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i + 1]
            
    def get_normalizer(self) -> GaussianNormalizer:
        """ `BaseDataset`의 추상 메소드를 구체적으로 구현합니다. """
        return self.normalizer

    def __len__(self) -> int:
        """ `BaseDataset`의 추상 메소드를 구체적으로 구현합니다. """
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """ `BaseDataset`의 추상 메소드를 구체적으로 구현합니다. """
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': self.seq_val[path_idx, start].copy(),
        }

        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['val'] = torch_data['val'].squeeze(-1)
        
        return torch_data
    
class TrajectoryTDDataset(BaseDataset):
    """
    Zarr 기반의 Transition 데이터셋. (메모리 최적화 버전)
    - IQL과 같은 TD 학습 알고리즘을 위해 (s, a, r, s', done) 데이터를 제공합니다.
    - Zarr 파일에서 직접 데이터를 읽어오며, 메모리 효율성을 극대화합니다.
    """
    def __init__(self, zarr_path: str):
        super().__init__()
        
        self.store = zarr.open(zarr_path, 'r')

        # Zarr 데이터 배열 참조 (실제 데이터 로드 X)
        observations = self.store['observations']
        actions = self.store['actions']
        rewards = self.store['rewards']
        terminals = self.store['dones']
        # timeout 데이터가 없을 경우를 대비해 기본값 제공
        timeouts = self.store.get('timeouts', np.zeros_like(terminals, dtype=bool))
        
        # 데이터 차원 저장
        self.o_dim = observations.shape[1]
        self.a_dim = actions.shape[1]
        
        # 🧠 최적화 1: Normalizer를 전체 데이터가 아닌 일부 샘플로 계산
        print("Calculating normalizer of data...")
        self.normalizer = WelfordNormalizer(observations, data_shape=(self.o_dim,))
        
        # 🧠 최적화 2: 에피소드 경계선만 스캔하여 저장
        print("Scanning episode boundaries...")
        self.episode_slices = []
        # 에피소드의 끝을 나타내는 인덱스를 찾음 (terminal 또는 timeout)
        done_indices = np.where(np.logical_or(terminals[:], timeouts[:]))[0]
        ptr = 0
        for end_idx in done_indices:
            self.episode_slices.append((ptr, end_idx + 1))
            ptr = end_idx + 1
        # 마지막 에피소드 추가
        if ptr < len(terminals):
            self.episode_slices.append((ptr, len(terminals)))

        # 🧠 최적화 3: 유효 Transition 개수를 계산하고 누적 합계 저장
        # valid_indices 리스트를 생성하는 대신, 개수만 파악하여 메모리 사용량 최소화
        print("Calculating virtual indices for transitions...")
        # 각 에피소드는 (길이 - 1) 만큼의 transition을 가짐
        transitions_per_episode = [
            max(0, (ep_end - ep_start) - 1)
            for ep_start, ep_end in self.episode_slices
        ]
        self.size = sum(transitions_per_episode)
        
        # __getitem__에서 idx를 실제 에피소드 위치로 변환하기 위한 누적 합계
        self.cumulative_transitions_per_episode = np.cumsum([0] + transitions_per_episode)

    def get_normalizer(self) -> WelfordNormalizer:
        return self.normalizer

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """
        하나의 Transition (s, a, r, s', d) 데이터를 Zarr에서 실시간으로 로드합니다.
        """
        # 💾 On-the-fly 인덱싱: 저장된 인덱스 리스트 대신 실시간으로 위치 계산
        # 1. idx가 어떤 에피소드에 속하는지 찾기
        episode_idx = np.searchsorted(self.cumulative_transitions_per_episode, idx, side='right') - 1
        
        # 2. 해당 에피소드 내에서의 시작 위치 오프셋 계산
        start_in_episode = idx - self.cumulative_transitions_per_episode[episode_idx]
        
        # 3. 에피소드의 전체 데이터 내 절대 시작 위치
        ep_start, _ = self.episode_slices[episode_idx]
        
        # 4. 최종 데이터의 실제 인덱스 계산
        true_idx = ep_start + start_in_episode
        
        # Zarr 스토어에서 필요한 데이터 조각만 읽어오기
        obs = self.store['observations'][true_idx]
        act = self.store['actions'][true_idx]
        rew = self.store['rewards'][true_idx]
        next_obs = self.store['observations'][true_idx + 1] # 다음 상태
        tml = self.store['dones'][true_idx]
        
        # 실시간으로 정규화 적용
        normed_obs = self.normalizer.normalize(obs)
        normed_next_obs = self.normalizer.normalize(next_obs)
        
        # 데이터를 텐서로 변환
        data = {
            'obs': {
                'state': torch.from_numpy(normed_obs).float(),
            },
            'next_obs': {
                'state': torch.from_numpy(normed_next_obs).float(),
            },
            'act': torch.from_numpy(act).float(),
            'rew': torch.tensor(rew, dtype=torch.float32).view(1),
            'tml': torch.tensor(tml, dtype=torch.float32).view(1), 
        }

        return data
    
class MultiHorizonTrajectoryDataset(BaseDataset):
    """
    Zarr 기반의 Multi-Horizon 시퀀스 데이터셋.
    - DiffuserLite의 계층적 모델 학습을 위해 여러 길이의 시퀀스를
      동시에 샘플링할 수 있습니다.
    """
    def __init__(self,
                 zarr_path: str,
                 horizons: Union[List[int], Tuple[int, ...]],       # Union : List 또는 Tuple 로 이루어진 자료형 둘 다 가능하다는 의미.
                 single_obs_dim : int,
                 num_stacks : int, 
                 discount: float = 0.99,
                 terminal_penalty: float = -100.0,
                 max_episode_lengths: int = 1000,
                 is_normalize: bool = True,
                 ):
        super().__init__()
        
        self.horizons = horizons
        self.store = zarr.open(zarr_path, 'r')
        self.max_episode_lengths = max_episode_lengths
        self.single_obs_dim = single_obs_dim
        self.num_stacks = num_stacks
        self.stacked_obs_dim = single_obs_dim * num_stacks
        
        observations = self.store['observations']
        terminals = self.store['dones']
        timeouts = self.store.get('timeouts', np.zeros_like(terminals, dtype=bool))                # 기본 정보 저장
    
        self.o_dim = observations.shape[1]
        self.a_dim = self.store['actions'].shape[1]
        self.context_dim = self.o_dim - self.single_obs_dim
        
        # discount는 __getitem__에서 직접 계산하도록 변경
        self.discount_factor = discount
        self.terminal_penalty = terminal_penalty

        self.is_normalize = is_normalize
        #self.normalizer = WelfordNormalizer(observations, data_shape=(self.o_dim,))
        if is_normalize:
            print("======================================================")
            print(f"Calculating normalizer for single_obs (dim={self.single_obs_dim})...")
            single_obs_data = observations[:, :self.single_obs_dim]
            #self.normalizer = WelfordNormalizer(single_obs_data, data_shape=(self.single_obs_dim,))
            self.normalizer = GaussianNormalizer(single_obs_data)
            print(f"Finshed Calculating normalizer, total data : {self.store['observations'].shape[0]}")
            print(f"Mean : {self.normalizer.mean}")
            print(f"Std : {self.normalizer.std}")
        
        
         # 🧠 최적화 2: 에피소드 경계선만 계산하고 저장 (실제 데이터 로드 X)
        print("======================================================")
        print("Scanning episode boundaries...")
        self.episode_slices = []
        done_indices = np.where(np.logical_or(terminals[:], timeouts[:]))[0]
        ptr = 0
        
        for end_idx in done_indices:
            self.episode_slices.append((ptr, end_idx + 1))
            ptr = end_idx + 1
        if ptr < len(terminals):
            self.episode_slices.append((ptr, len(terminals)))

        # 🧠 최적화 3: 모든 인덱스를 저장하는 대신, 샘플링 가능한 '개수'만 계산
        # self.indices 리스트를 완전히 제거합니다.
        self.len_each_horizon = []
        self.cumulative_samples_per_horizon = []
        
        print("======================================================")
        print("Calculating virtual indices for each horizon...") # => 에피소드의 경계를 넘어가는 샘플은 배제할 수 있다.
        for horizon in self.horizons:
            # 각 에피소드가 현재 horizon에 대해 몇 개의 샘플을 만들 수 있는지 계산
            # 만약 에피소드의 길이가 1000이고, horizon 이 129 라면 한 에피소드에서 872 개의 궤적을 샘플 할 수 있다.
            samples_per_episode = [
                max(0, (ep_end - ep_start) - horizon + 1)
                for ep_start, ep_end in self.episode_slices
            ]
            # 한 호라이즈에서 샘플할 수 있는 궤적의 갯수를 저장
            self.len_each_horizon.append(sum(samples_per_episode))
            # __getitem__에서 idx를 실제 에피소드 위치로 변환하기 위한 누적 합계
            self.cumulative_samples_per_horizon.append(np.cumsum([0] + samples_per_episode))

    def get_normalizer(self) -> WelfordNormalizer: # WelfordNormalizer:
        return self.normalizer

    def __len__(self) -> int:
        # 원본 로직과 동일하게, 가장 많은 샘플을 가진 horizon을 기준으로 길이를 정함
        return max(self.len_each_horizon) if self.len_each_horizon else 0
    
    def _find_sample_location(self, horizon_idx: int, virtual_idx: int) -> tuple[int, int]:
        """
        가상 인덱스(virtual_idx)를 실제 Zarr 파일의 절대적인 start, end 인덱스로 변환합니다.
        """
        # [1단계] 이 가상 인덱스가 어느 '에피소드'에 속하는지 찾기
        # "1번 책장까지 누적 100권, 2번까지 250권..." 과 같은 누적 합계표를 사용합니다.
        cumulative_counts = self.cumulative_samples_per_horizon[horizon_idx]
        episode_idx = np.searchsorted(cumulative_counts, virtual_idx, side='right') - 1
    
        # [2단계] 해당 에피소드 내에서 몇 번째 '시작점'인지 계산하기
        # (내 번호표) - (이전 책장까지의 누적 책 수)
        start_offset_in_episode = virtual_idx - cumulative_counts[episode_idx]
        
        # [3단계] 에피소드의 실제 시작/끝 위치 가져오기
        # "2번 책장은 파일의 1001번 라인부터 1998번 라인까지다"
        episode_start_line, episode_end_line = self.episode_slices[episode_idx]
        
        # [4단계] 최종 절대 위치 계산 및 반환
        # (책장의 시작 라인) + (책장 내에서의 상대적 위치)
        absolute_start = episode_start_line + start_offset_in_episode
        
        # 샘플이 에피소드 경계를 넘어가지 않도록 안전하게 끝 위치 조정
        horizon_len = self.horizons[horizon_idx]
        absolute_end = min(absolute_start + horizon_len, episode_end_line)
        
        return absolute_start, absolute_end
    
    # for batch in loop_dataloader(dataloader): 에서 batch size 만큼 호출됨.
    # dataloader 는 RandomSampler 를 사용해 idx를 랜덤하게 생성함.
    def __getitem__(self, idx: int) -> List[Dict]:  
        torch_datas = []
    
        for i, horizon in enumerate(self.horizons):
            if self.len_each_horizon[i] == 0:
                continue
            
            # data 를 균등하게 sample 하기 위해 가상 인덱스 계산
            # idx 는 전체 데이터셋 길이 기준으로 주어지므로, 각 horizon의 길이에 맞게 조정합니다.     
            # self.len_each_horizon[i] 는 해당 horizon에서 샘플링 가능한 최대 개수입니다.
            max_len = max(self.len_each_horizon)
            scaled_idx = int(self.len_each_horizon[i] * (idx / max_len))
            virtual_idx = min(scaled_idx, self.len_each_horizon[i] - 1)         # self.len_each_horizon[i] - 1 는 index 범위를 벗어나지 않도록 보장합니다.
        
            # 2. 실제 파일 위치 찾기 (헬퍼 메소드 호출)
            start, end = self._find_sample_location(
            horizon_idx=i, virtual_idx=virtual_idx)
            
            # 3. 데이터 로딩
            obs = self.store['observations'][start:end, :self.single_obs_dim]          
            act = self.store['actions'][start:end]
            rewards = self.store['rewards'][start:end]
    
            # ✨ [수정 2] 각 시점별 Return-to-Go (RTG) 시퀀스 계산
            rtgs = np.zeros_like(rewards, dtype=np.float32)
            rtgs[-1] = rewards[-1]
            for t in reversed(range(len(rewards) - 1)):
                rtgs[t] = rewards[t] + self.discount_factor * rtgs[t + 1]
            
            # context 벡터 생성    
            total_obs = self.store['observations'][start]
            history_obs_flat = np.array(total_obs[self.single_obs_dim: self.stacked_obs_dim])
            history_obs = history_obs_flat.reshape(
                self.num_stacks - 1, # history_len
                self.single_obs_dim # s_dim + a_dim
            )
            # s_{t-n}, ..., s_{t-2}, s_{t-1}
            history_obs = np.flip(history_obs, axis=0)
            command = np.array(total_obs[self.stacked_obs_dim : ]).astype(np.float32)
            
            if self.is_normalize:
                normed_obs = self.normalizer.normalize(obs).astype(np.float32)
                normed_history_obs = self.normalizer.normalize(history_obs).astype(np.float32)
            else:
                normed_obs = obs.astype(np.float32)
                normed_history_obs = history_obs.astype(np.float32)
        
            act = act.astype(np.float32)
            data = {
                'obs': {'state': normed_obs},   # 현재 step 에 대한 상태와 이전 step 에서의 action 저장,
                'act': act,                     # 현재 step 에 대한 action.
                'val': rtgs,
                'history_obs': normed_history_obs,  # ✨ 생성된 context 벡터를 데이터에 추가
                'command' : command,
            }
            
            # NumPy to Tensor 변환
            torch_data = dict_apply(data, torch.from_numpy)
            torch_datas.append({"horizon": horizon, "data": torch_data})
    
        return torch_datas
    
    def get_obs_sample(self, size=1000):
        # 검증을 위해 history_obs 샘플을 추출하는 메서드
        samples = []
        indices = np.random.choice(self.store['observations'].shape[0], size, replace=False)
        for i in indices:
            obs = self.store['observations'][i]
            # ✨ history_obs 추출 로직 (Dataset의 __getitem__과 동일해야 함)
            samples.append(obs[:self.stacked_obs_dim])
        return np.array(samples)
    
    def get_full_obs_sample(self, size=1000):
        # 검증을 위해 history_obs 샘플을 추출하는 메서드
        samples = []
        indices = np.random.choice(self.store['observations'].shape[0], size, replace=False)
        for i in indices:
            obs = self.store['observations'][i]
            # ✨ history_obs 추출 로직 (Dataset의 __getitem__과 동일해야 함)
            samples.append(obs)
        return np.array(samples)
    
    def get_rtg_sample(self, size=1000):
        # 검증을 위해 RTG 샘플을 추출하는 메서드
        samples = []
        indices = np.random.choice(self.store['rewards'].shape[0], size, replace=False)
        for i in indices:
            rew = self.store['rewards'][i]
            # ✨ RTG 계산 로직 (Dataset의 __getitem__과 동일해야 함)
            rtg = np.zeros_like(rew, dtype=np.float32)
            rtg[-1] = rew[-1]
            for t in reversed(range(len(rew) - 1)):
                rtg[t] = rew[t] + self.discount_factor * rtg[t + 1]
            samples.append(rtg)
        return np.array(samples)
    
    def compute_rtg_scales(self):
        rewards = self.store['rewards'][:]
        max_reward = np.max(rewards)
        print("max reward : ", max_reward)
        
        rtg_scales = []
        print("Calculating max RTG for each horizon...")
        for h in self.horizons:
            scale = max_reward * (1 - self.discount_factor**h) / (1 - self.discount_factor)
            rtg_scales.append(scale)
            print(f"Horizon {h}: Max RTG = {scale}")
            
        return rtg_scales
    
    def log_sanity_checks(self, logger, args, first_batch):

        print("Performing data sanity check and logging to WandB...")

       # --- 1단계: 원본 데이터 샘플링 ---
        raw_obs_sample = self.get_full_obs_sample(size=10000)[:, :args.task.single_obs]
        raw_actions_sample = self.store['actions'][np.random.choice(len(self.store['actions']), 10000)]
        raw_rewards_sample = self.store['rewards'][np.random.choice(len(self.store['rewards']), 10000)]

       # --- 2단계: WandB Table 생성 및 데이터 추가 ---
       # 원본 데이터 분포를 위한 테이블
        raw_obs_table = wandb.Table(columns=["raw_observations"])
        raw_act_table = wandb.Table(columns=["raw_actions"])
        raw_rew_table = wandb.Table(columns=["raw_rewards"])

        for i in range(raw_obs_sample.shape[0]):
           # .flatten()으로 다차원 샘플도 추가 가능
            raw_obs_table.add_data(raw_obs_sample[i].flatten())
            raw_act_table.add_data(raw_actions_sample[i].flatten())
            raw_rew_table.add_data(raw_rewards_sample[i].flatten())

        log_dict = {
           "data_distribution_check/1_raw_obs_table": raw_obs_table,
           "data_distribution_check/1_raw_action_table": raw_act_table,
           "data_distribution_check/1_raw_reward_table": raw_rew_table,
        }

       # --- 3단계: 정규화(Normalization) 검증 ---
        if self.is_normalize:
           # 정규화 통계는 요약(Summary) 정보로 적합
            logger._wandb.summary["data_distribution_check/normalizer_mean"] = self.normalizer.mean.mean() # 대표값으로 평균
            logger._wandb.summary["data_distribution_check/normalizer_std"] = self.normalizer.std.mean()  # 대표값으로 평균

           # 정규화된 데이터 분포는 테이블로
            normed_obs_sample = first_batch[0]['data']['obs']['state'].numpy()
            normed_obs_table = wandb.Table(columns=["normalized_observations"])
            for i in range(normed_obs_sample.shape[0]):
                normed_obs_table.add_data(normed_obs_sample[i].flatten())

            log_dict["data_distribution_check/2_normalized_obs_table"] = normed_obs_table

        
        logger._wandb.log(log_dict, commit=True)
        print("Data sanity check complete.")
