# utils/state_handler.py
import torch

class StateHandler:
    """
    StateHandler는 주어진 stacking 프레임 수(total_frames; 현재 프레임 포함)만큼
    stack_obs를 버퍼에 저장한 뒤, 버퍼에 저장된 관측값들을 concat하고, 
    non_stack_obs와 이어붙여 최종 관측 벡터를 반환합니다.
    또한, 최종 관측 벡터의 차원(num_obs)을 계산하여 관리합니다.
    """
    def __init__(self, total_frames: int, stack_dim: int, nonstack_dim: int):
        """
        Args:
            total_frames (int): stacking할 총 프레임 수 (현재 프레임 포함).
                                예를 들어, num_stacks가 2이면 total_frames는 3.
            stack_dim (int): stack_obs의 차원.
            nonstack_dim (int): non_stack_obs의 차원.
        """
        self.total_frames = total_frames
        self.stack_dim = stack_dim
        self.nonstack_dim = nonstack_dim
        self.num_obs = stack_dim * total_frames + nonstack_dim
        self.stack_buffer = None

    def reset(self, stack_obs: torch.Tensor, nonstack_obs: torch.Tensor) -> torch.Tensor:
        """
        버퍼를 초기화하고, 모든 프레임을 stack_obs로 채운 후, stacking된 관측값과 nonstack_obs를 concat하여 최종 관측값을 반환합니다.
        
        Args:
            stack_obs (torch.Tensor): shape (num_envs, stack_dim)
            nonstack_obs (torch.Tensor): shape (num_envs, nonstack_dim)
            
        Returns:
            torch.Tensor: 최종 관측값, shape (num_envs, num_obs)
        """
        self.stack_buffer = [stack_obs.clone() for _ in range(self.total_frames)]
        stacked = self.get_stacked()
        return torch.cat([stacked, nonstack_obs], dim=-1)

    def update(self, stack_obs: torch.Tensor, nonstack_obs: torch.Tensor) -> torch.Tensor:
        """
        새로운 stack_obs를 버퍼에 업데이트하고, stacking된 관측값과 nonstack_obs를 concat하여 최종 관측값을 반환합니다.
        
        Args:
            stack_obs (torch.Tensor): 최신 stack_obs, shape (num_envs, stack_dim)
            nonstack_obs (torch.Tensor): 최신 non_stack_obs, shape (num_envs, nonstack_dim)
            
        Returns:
            torch.Tensor: 최종 관측값, shape (num_envs, num_obs)
        """
        if self.stack_buffer is None:
            return self.reset(stack_obs, nonstack_obs)
        self.stack_buffer = [stack_obs.clone()] + self.stack_buffer[:-1]
        stacked = self.get_stacked()
        return torch.cat([stacked, nonstack_obs], dim=-1)

    def get_stacked(self) -> torch.Tensor:
        """
        버퍼에 저장된 stack_obs들을 concat하여 stacked 관측값을 생성합니다.
        
        Returns:
            torch.Tensor: shape (num_envs, stack_dim * total_frames)
        """
        if self.total_frames == 1:
            return self.stack_buffer[0]
        return torch.cat(self.stack_buffer, dim=-1)
