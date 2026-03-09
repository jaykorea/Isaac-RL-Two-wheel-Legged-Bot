from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

# Zarr 관련 유틸 (경로가 들어왔을 때만 사용)
from .zarr_utils import open_zarr_group, get_transitions_dataset

class CbfDataset(Dataset):
    """
    Hybrid Dataset:
    1. source가 'str' (파일 경로) -> Zarr를 열어서 Disk I/O 모드로 동작 (RAM 절약, 속도 느림)
    2. source가 'ndarray' (데이터) -> RAM에 있는 데이터를 바로 사용 (RAM 사용, 속도 빠름)
    """

    def __init__(
        self,
        source: Union[str, np.ndarray, Any], # 경로(str) 혹은 데이터 객체(Array)
        indices: np.ndarray,
        depth_clip: Optional[Tuple[float, float]] = (0.0, 4.0),
        depth_scale: float = 4.0,
        depth_downsample: int = 1,
    ):
        super().__init__()
        self.indices = indices.astype(np.int64)

        # ---------------------------------------------------------
        # [핵심] 입력 타입에 따른 유연한 처리
        # ---------------------------------------------------------
        if isinstance(source, str):
            # Case 1: 경로가 들어옴 -> Zarr 직접 열기 (Disk Mode)
            # print(f"[Dataset] Loading from disk (Zarr): {source}")
            self.g = open_zarr_group(source)
            self.ds = get_transitions_dataset(self.g)
        else:
            # Case 2: 배열(Numpy or Zarr Array)이 들어옴 -> 그대로 사용 (RAM Mode)
            # print(f"[Dataset] Using pre-loaded data object.")
            self.ds = source

        # ---------------------------------------------------------
        # 데이터 검증 (필드 및 Shape 확인)
        # ---------------------------------------------------------
        # Numpy Structured Array와 Zarr Array 모두 dtype.names를 가짐
        names = self.ds.dtype.names
        for req in ["depth", "ee_pos", "cbf"]:
            if req not in names:
                raise KeyError(f"Dataset missing field '{req}'. Available: {names}")

        # 첫 샘플로 차원 확인 (데이터가 비어있지 않은 경우)
        if len(self.indices) > 0:
            # self.ds[idx]는 Zarr든 Numpy든 동일하게 동작
            sample_idx = int(self.indices[0])
            sample = self.ds[sample_idx] 
            
            dep = sample["depth"]
            if dep.ndim == 2: # (H,W) -> (H,W,1) 보정
                dep = dep[..., None]
            
            if dep.ndim != 3 or dep.shape[-1] != 1:
                raise ValueError(f"Expected depth shape (H,W,1), got {dep.shape}")
            
            self.H, self.W, _ = dep.shape
        else:
            # Fallback (데이터가 없는 경우, 예외 방지용)
            self.H, self.W = 240, 320

        # ---------------------------------------------------------
        # 설정 저장
        # ---------------------------------------------------------
        self.depth_clip = depth_clip
        self.depth_scale = float(depth_scale) if depth_scale is not None else 1.0

        self.depth_downsample = int(depth_downsample)
        if self.depth_downsample < 1:
            self.depth_downsample = 1

        if self.H % self.depth_downsample != 0 or self.W % self.depth_downsample != 0:
            raise ValueError(
                f"depth_downsample={self.depth_downsample} must divide H,W. got H={self.H}, W={self.W}"
            )

        self.H_ds = self.H // self.depth_downsample
        self.W_ds = self.W // self.depth_downsample

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        
        # [공통 인터페이스] 
        # Numpy Array면 메모리 접근, Zarr Array면 디스크 접근
        # 둘 다 Dictionary-like 구조화된 데이터를 반환함
        row = self.ds[idx]

        depth = row["depth"]
        if depth.ndim == 2:
            depth = depth[..., None]
            
        ee = row["ee_pos"]
        y = row["cbf"]

        # 타입 변환 및 전처리
        depth = depth.astype(np.float32)
        ee = ee.astype(np.float32)
        y = y.astype(np.float32)

        # (H,W,1) -> (1,H,W)
        depth = np.transpose(depth, (2, 0, 1))

        # Downsample
        if self.depth_downsample > 1:
            depth = depth[:, :: self.depth_downsample, :: self.depth_downsample]

        # Clip & Scale
        if self.depth_clip is not None:
            lo, hi = float(self.depth_clip[0]), float(self.depth_clip[1])
            depth = np.clip(depth, lo, hi)
        
        if self.depth_scale is not None and self.depth_scale > 0:
            depth = depth / self.depth_scale

        return (
            torch.from_numpy(depth),
            torch.from_numpy(ee),
            torch.from_numpy(y),
        )
    


class MultiModalDataset(Dataset):
    """
    Zarr 속성(camera_config)을 기반으로 동적으로 이미지 키를 로드하는 데이터셋
    반환값: (images_dict, ee, cbf)
      - images_dict: {"tv_cam_rgb": [3, H, W], "tv_cam_depth": [1, H, W], ...}
    """

    def __init__(
        self,
        source: Union[str, np.ndarray, Any],
        indices: np.ndarray,
        camera_config: Optional[Dict[str, Any]] = None,
        depth_clip: Optional[Tuple[float, float]] = None,
        depth_scale: float = 1.0,
        depth_downsample: int = 1,  # [Fix] 이 인자가 꼭 있어야 합니다
    ):
        super().__init__()
        self.indices = indices.astype(np.int64)
        
        # ---------------------------------------------------------
        # 1. Source 로드 & Config 감지
        # ---------------------------------------------------------
        self.is_zarr_disk = False
        if isinstance(source, str):
            # [Disk Mode]
            self.g = open_zarr_group(source)
            self.ds = get_transitions_dataset(self.g)
            
            if camera_config is None and "camera_config" in self.g.attrs:
                self.camera_config = self.g.attrs["camera_config"]
            else:
                self.camera_config = camera_config
            
            self.is_zarr_disk = True
        else:
            # [RAM Mode]
            self.ds = source
            self.camera_config = camera_config if camera_config else {}

        # ---------------------------------------------------------
        # 2. 로드할 키(Key) 분석
        # ---------------------------------------------------------
        if not self.camera_config:
            self.camera_config = {"default": {"depth": True}}
            self._legacy_mode = True
        else:
            self._legacy_mode = False

        self.load_keys = []
        for cam_name, cfg in self.camera_config.items():
            # RGB
            if cfg.get("rgb"):
                key = "rgb" if self._legacy_mode else f"{cam_name}_rgb"
                self.load_keys.append((key, "rgb"))
            
            # Depth
            if cfg.get("depth"):
                key = "depth" if self._legacy_mode else f"{cam_name}_depth"
                self.load_keys.append((key, "depth"))

        # ---------------------------------------------------------
        # 3. 전처리 파라미터 저장
        # ---------------------------------------------------------
        self.depth_clip = depth_clip
        self.depth_scale = float(depth_scale) if depth_scale is not None else 1.0
        
        # [Fix] 다운샘플링 설정 저장
        self.depth_downsample = int(depth_downsample)
        if self.depth_downsample < 1:
            self.depth_downsample = 1

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        row = self.ds[idx]

        # ---------------------------------------------------------
        # 1. Images Dict 생성
        # ---------------------------------------------------------
        images = {}
        
        for key, modality in self.load_keys:
            if key not in row.dtype.names:
                continue
                
            data = row[key] # (H, W, C) or (H, W)
            
            # (H, W) -> (H, W, 1)
            if data.ndim == 2:
                data = data[..., None]
            
            # 타입 변환 및 전처리
            if modality == "rgb":
                # RGB: uint8 -> float32 (0~1)
                data = data.astype(np.float32) / 255.0
            
            elif modality == "depth":
                # Depth: float16/32 -> float32
                data = data.astype(np.float32)
                
                # Clip
                if self.depth_clip is not None:
                    lo, hi = self.depth_clip
                    data = np.clip(data, lo, hi)
                
                # Scale
                if self.depth_scale > 0:
                    data = data / self.depth_scale

            # Channel First: (H, W, C) -> (C, H, W)
            data = np.transpose(data, (2, 0, 1))
            
            # [Fix] Depth Downsample 적용
            if modality == "depth" and self.depth_downsample > 1:
                # data[:, ::ds, ::ds] 형태로 H, W 차원만 줄임
                data = data[:, ::self.depth_downsample, ::self.depth_downsample]
            
            # 메모리 연속성 확보 (PyTorch 변환 전 필수)
            data = np.ascontiguousarray(data)
            
            images[key] = torch.from_numpy(data)

        # ---------------------------------------------------------
        # 2. Vector Data
        # ---------------------------------------------------------
        ee = row["ee_pos"].astype(np.float32)
        cbf = row["cbf"].astype(np.float32)

        return images, torch.from_numpy(ee), torch.from_numpy(cbf)