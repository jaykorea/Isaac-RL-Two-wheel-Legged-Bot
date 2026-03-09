import math
import os
import random
from typing import Union, Dict, Callable

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """ Set seed for reproducibility """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (np.ndarray, list, tuple, int, float)):
        return torch.tensor(x, device=device)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)


# ================ Discretization ==================
def uniform_discretization(T: int = 1000, eps: float = 1e-3):
    return torch.linspace(eps, 1.0, T)


SUPPORTED_DISCRETIZATIONS = {
    "uniform": uniform_discretization,
}


# ================= Noise schedules =================
def linear_noise_schedule(
    t_diffusion: torch.Tensor, beta0: float = 0.1, beta1: float = 20.0
):
    log_alpha = -(beta1 - beta0) / 4.0 * (t_diffusion**2) - beta0 / 2.0 * t_diffusion
    alpha = log_alpha.exp()
    sigma = (1.0 - alpha**2).sqrt()
    return alpha, sigma


def inverse_linear_noise_schedule(
    alpha: torch.Tensor = None,
    sigma: torch.Tensor = None,
    logSNR: torch.Tensor = None,
    beta0: float = 0.1,
    beta1: float = 20.0,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (2 * (1 + (-2 * lmbda).exp()).log() /
                   (beta0 + (beta0**2 + 2 * (beta1 - beta0) * (1 + (-2 * lmbda).exp()).log())))
    return t_diffusion


def cosine_noise_schedule(t_diffusion: torch.Tensor, s: float = 0.008):
    t_diffusion[-1] = 0.9946
    alpha = (np.pi / 2.0 * (t_diffusion + s) / (1 + s)).cos() / np.cos(
        np.pi / 2.0 * s / (1 + s))
    sigma = (1.0 - alpha**2).sqrt()
    return alpha, sigma


def inverse_cosine_noise_schedule(
    alpha: torch.Tensor = None,
    sigma: torch.Tensor = None,
    logSNR: torch.Tensor = None,
    s: float = 0.008,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (
        2 * (1 + s) / np.pi * torch.arccos((
            -0.5 * (1 + (-2 * lmbda).exp()).log()
            + np.log(np.cos(np.pi * s / 2 / (s + 1)))).exp()) - s)
    return t_diffusion


SUPPORTED_NOISE_SCHEDULES = {
    "linear": {
        "forward": linear_noise_schedule,
        "reverse": inverse_linear_noise_schedule,
    },
    "cosine": {
        "forward": cosine_noise_schedule,
        "reverse": inverse_cosine_noise_schedule,
    },
}


# ================= Sampling step schedule ===============
def uniform_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10):
    return torch.linspace(0, T - 1, sampling_steps + 1, dtype=torch.long)


def uniform_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10):
    if trange is None:
        trange = [1e-3, 1.0]
    return torch.linspace(trange[0], trange[1], sampling_steps + 1, dtype=torch.float32)


def quad_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10, n: int = 1.5):
    schedule = (T - 1) * (
        torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n)
    return schedule.to(torch.long)


def quad_sampling_step_schedule_continuous(
    trange=None, sampling_steps: int = 10, n: int = 1.5
):
    if trange is None:
        trange = [1e-3, 1.0]
    schedule = (trange[1] - trange[0]) * (
        torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n
    ) + trange[0]
    return schedule


def cat_cos_sampling_step_schedule(
    T: int = 1000, sampling_steps: int = 10, n: int = 2.0
):
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = (0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n) + 0.5)
    schedule = (T - 1) * idx
    return schedule.to(torch.long)


def cat_cos_sampling_step_schedule_continuous(
    trange=None, sampling_steps: int = 10, n: int = 2.0
):
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = (0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n) + 0.5)
    schedule = (trange[1] - trange[0]) * idx + trange[0]
    return schedule


def quad_cos_sampling_step_schedule(
    T: int = 1000, sampling_steps: int = 10, n: int = 2.0
):
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    schedule = (T - 1) * idx
    return schedule.to(torch.long)


def quad_cos_sampling_step_schedule_continuous(
    trange=None, sampling_steps: int = 10, n: int = 2.0
):
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    schedule = (trange[1] - trange[0]) * idx + trange[0]
    return schedule


SUPPORTED_SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
    "quad": quad_sampling_step_schedule,
    "quad_continuous": quad_sampling_step_schedule_continuous,
    "cat_cos": cat_cos_sampling_step_schedule,
    "cat_cos_continuous": cat_cos_sampling_step_schedule_continuous,
    "quad_cos": quad_cos_sampling_step_schedule,
    "quad_cos_continuous": quad_cos_sampling_step_schedule_continuous,
}


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ema_update(model: nn.Module, model_ema: nn.Module, ema_rate: float):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures,
# from https://github.com/NVlabs/edm/blob/main/training/networks.py#L269
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions              # sinusoidal positional embedding 에서의 10000 에 대응
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs                # 1 / max_positions ** (i / (dim/2 - 1)), i=0, 1, ..., dim/2-1
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class UntrainablePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.einsum('...i,j->...ij', x, freqs.to(x.dtype))
        # x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    """
        Calculate sinusoidal positional embeddings for a given input tensor.
        PE(t, 2i) = sin(t / 10000^(2i/dim))
        PE(t, 2i+1) = cos(t / 10000^(2i/dim))
        where t is the time step, i is the dimension index, and dim is the total dimension of the embedding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim                                                  # 출력할 임베딩 벡터의 차원

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)                          # 수치적 안정성을 위해 log(10000) 사용
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)   # exp(-log(10000) * i / (dim/2 - 1)),  (i=0, 1, ..., dim/2-1)
        emb = torch.einsum('...i,j->...ij', x, emb.to(x.dtype))         # 아래의 주석과 동일한 계산으로, Batched outer product 를 계산한다.
        # emb = x[:, None] * emb[None, :]                               # x의 shape 를 [batch_size, 1] 으로 확장하고, emb의 shape 를 [1, dim/2] 으로 확장하여 outer product 계산
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)                 # 짝수 인덱스는 sin, 홀수 인덱스는 cos 함수를 적용하여 임베딩 벡터 생성
        return emb


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return emb


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,                          # 사인과 코사인 함수를 이용해서 다른 주파수를 가지는 주기 함수를 만들어 t 를 고차원 벡터로 임베딩
    "fourier": FourierEmbedding,                                # t 가 0~1 사이의 연속적인 시간 값을 가질 때 Fourier feature 를 이용해서 임베딩 -> cos(2*pi* B @ T) 의 형태로, B 는 가우시안 분포에서 샘플린된 후 고정. 푸리에 피처 벡터 위의 학습 가능한 MLP 추가?
    "untrainable_fourier": UntrainableFourierEmbedding,         # FourierEmbedding 에서 학습 가능한 레이어 제거
    "untrainable_positional": UntrainablePositionalEmbedding,   # PositionalEmbedding 에서 학습 가능한 레이어 제거
}


# -----------------------------------------------------------
# Beautiful model size visualization from https://github.com/jannerm/diffuser/tree/main


def _to_str(num):
    if num >= 1e6:
        return f"{(num / 1e6):.2f} M"
    else:
        return f"{(num / 1e3):.2f} k"


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    n_parameters = sum(counts.values())
    print(f"Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters


# ----------------------- DD return scale

# discount = 0.997
DD_RETURN_SCALE = {
    "base_velocity" : 0.06
}


# ---------------------- Freeze and Unfreeze


class FreezeModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_grad_status = {}

    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                self.original_grad_status[id(param)] = param.requires_grad
                param.requires_grad = False

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = self.original_grad_status[id(param)]


class UnfreezeModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_grad_status = {}

    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                self.original_grad_status[id(param)] = param.requires_grad
                param.requires_grad = True

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = self.original_grad_status[id(param)]


class EvalModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_status = {}

    def __enter__(self):
        for module in self.modules:
            self.original_status[id(module)] = module.training
            module.eval()

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            module.train(self.original_status[id(module)])


class TrainModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_status = {}

    def __enter__(self):
        for module in self.modules:
            self.original_status[id(module)] = module.training
            module.train()

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            module.train(self.original_status[id(module)])


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b
