import os
import time
import numpy as np
import torch
import zarr
from typing import Dict, Optional, Tuple, List, Any

try:
    from numcodecs import Blosc
except Exception:
    Blosc = None


class SequenceDataStorage:
    
    """
    Buffered Option B (step-major 유지):
    - root["transitions"] 단일 dataset (structured dtype)
    - add_step_data(): 메모리 버퍼에만 push
    - (dones|timeouts)로 끝난 episode 수 누적이 flush_every_episodes 이상이면 flush
    - flush_to_zarr(): append + (NEW) elapsed time / throughput 출력
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        zarr_path: str,
        camera_config : Optional[Dict[str, Any]] = None, 
        include_ee: bool = False,
        include_cbf: bool = False,
        chunk_size: int = 1024,
        depth_dtype: str = "float16",
        compress: bool = True,
        flush_every_episodes: int = 50,   # NEW: 이 값만으로 flush 제어
        auto_flush: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.zarr_path = zarr_path

        self.camera_config = camera_config if camera_config is not None else {}

        self.include_ee = bool(include_ee)
        self.include_cbf = bool(include_cbf)
        self.chunk_size = int(chunk_size)

        self.depth_dtype = str(depth_dtype)
        self.compress = bool(compress)

        self.flush_every_episodes = int(flush_every_episodes)
        self.auto_flush = bool(auto_flush)

        # per-env episode tracker
        self._ep_id = np.zeros((self.num_envs,), dtype=np.int64)
        self._t_in_ep = np.zeros((self.num_envs,), dtype=np.int32)

        # buffering state
        self._pending: List[np.ndarray] = []   # each: (N,)
        self._pending_ended_episodes = 0

        # =========================
        # NEW: flush timing stats
        # =========================
        self._t_flush_window_start = time.perf_counter()  # "since last flush"
        self._total_flush_count = 0

        self._init_zarr_store()

    def _compressor(self):
        if not self.compress or Blosc is None:
            return None
        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    def _make_transition_dtype(self) -> np.dtype:
        fields = [
            ("observations",      np.float32, (self.obs_dim,)),
            ("actions",           np.float32, (self.action_dim,)),
            ("rewards",           np.float32, (1,)),
            ("dones",             np.bool_,   (1,)),
            ("timeouts",          np.bool_,   (1,)),
            ("next_observations", np.float32, (self.obs_dim,)),
            ("env_id",            np.int16,   (1,)),
            ("episode_id",        np.int64,   (1,)),
            ("t_in_episode",      np.int32,   (1,)),
        ]
        if self.include_cbf:
            fields.append(("cbf", np.float32, (1,)))
        if self.include_ee:
            fields.append(("ee_pos", np.float32, (7,)))
        
        for cam_name, cfg in self.camera_config.items():
            shape_hw = cfg.get("shape", (128, 128)) # (H, W)
            
            # RGB 필드
            if cfg.get("rgb", False):
                # shape: (H, W, 3), dtype: uint8
                rgb_shape = tuple(shape_hw) + (3,)
                fields.append((f"{cam_name}_rgb", np.uint8, rgb_shape))
            
            # Depth 필드
            if cfg.get("depth", False):
                # shape: (H, W, 1), dtype: 설정값 따름
                depth_shape = tuple(shape_hw) + (1,)
                dtype_str = cfg.get("depth_dtype", "float16")
                ddt = np.float16 if dtype_str == "float16" else np.float32
                fields.append((f"{cam_name}_depth", ddt, depth_shape))

        return np.dtype(fields)

    def _init_zarr_store(self):
        parent = os.path.dirname(self.zarr_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.root = zarr.open_group(self.zarr_path, mode="a")
        comp = self._compressor()
        self._dtype = self._make_transition_dtype()

        if "transitions" not in self.root:
            self.root.create_dataset(
                "transitions",
                shape=(0,),
                maxshape=(None,),
                chunks=(self.chunk_size,),
                dtype=self._dtype,
                compressor=comp,
            )
            a = self.root.attrs
            a["schema_version"] = self.SCHEMA_VERSION
            a["num_envs"] = self.num_envs
            a["obs_dim"] = self.obs_dim
            a["action_dim"] = self.action_dim
            a["include_ee"] = self.include_ee
            a["include_cbf"] = self.include_cbf
            a["camera_config"] = self.camera_config
        else:
            a = self.root.attrs
            if int(a.get("obs_dim", -1)) != self.obs_dim or int(a.get("action_dim", -1)) != self.action_dim:
                raise ValueError("existing zarr schema mismatch: obs_dim/action_dim differs")
            if bool(a.get("include_ee", False)) != self.include_ee:
                raise ValueError("existing zarr schema mismatch: include_ee differs")
            if bool(a.get("include_cbf", False)) != self.include_cbf:
                raise ValueError("existing zarr schema mismatch: include_cbf differs")
            if dict(a.get("camera_config", {})) != self.camera_config:
                raise ValueError("existing zarr schema mismatch: camera_config differs")
        self.ds = self.root["transitions"]

    def _normalize_depth(self, dep: torch.Tensor) -> torch.Tensor:
        if dep.dim() < 3:
            raise ValueError(f"depth tensor must have at least 3 dims, got {tuple(dep.shape)}")

        tail = tuple(dep.shape[1:])
        target = tuple(self.depth_shape)

        if tail == target:
            return dep
        if len(target) == 3 and target[-1] == 1 and tail == target[:2]:
            return dep.unsqueeze(-1)
        if len(target) == 3 and tail == (1, target[0], target[1]):
            return dep.permute(0, 2, 3, 1).contiguous()

        raise ValueError(f"depth shape mismatch: got {tail}, expected {target}")

    @torch.no_grad()
    def add_step_data(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        timeouts: torch.Tensor,
        next_obs: torch.Tensor,
        cbf: Optional[torch.Tensor] = None,
        camera_data: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        ee_pos: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Isaac Lab의 (num_envs, ...) 형태 데이터를 받아서 버퍼에 저장.
        num_envs=1 인 경우에도 첫 차원은 유지되어야 함.
        """
        
        # 1. 환경 개수(N) 파악 (obs: [1, obs_dim] -> N=1)
        N = int(obs.shape[0])
        if N != self.num_envs:
            raise ValueError(f"num_envs mismatch: got {N}, expected {self.num_envs}")

        # ----------------------------------------------------------------------
        # A. 기본 데이터 변환 (CPU 이동 및 numpy 변환)
        # ----------------------------------------------------------------------
        obs_np  = obs.detach().cpu().to(torch.float32).numpy()       # (N, obs_dim)
        act_np  = actions.detach().cpu().to(torch.float32).numpy()   # (N, action_dim)
        nxt_np  = next_obs.detach().cpu().to(torch.float32).numpy()  # (N, obs_dim)
        
        # [중요] Scalar 값들도 (N, 1) 형태로 확실하게 View (입력이 [N]이든 [N,1]이든 상관없이)
        rew_np  = rewards.detach().cpu().view(N, 1).to(torch.float32).numpy()
        done_np = dones.detach().cpu().view(N, 1).to(torch.bool).numpy()
        tout_np = timeouts.detach().cpu().view(N, 1).to(torch.bool).numpy()

        # ----------------------------------------------------------------------
        # B. 옵션 데이터 변환 (CBF, EE, Depth)
        # ----------------------------------------------------------------------
        # CBF 처리
        if self.include_cbf:
            if cbf is None:
                raise ValueError("include_cbf=True but cbf is None")
            # 입력이 [1] (스칼라) 이거나 [1, 1] 이거나 모두 [1, 1]로 통일
            cbf_np = cbf.detach().cpu().view(N, 1).to(torch.float32).numpy()

        # EE Position 처리
        if self.include_ee:
            if ee_pos is None:
                raise ValueError("include_ee=True but ee_pos is None")
            # 입력: [1, 7] -> 변환: [1, 7] (안전함)
            ee_np = ee_pos.detach().cpu().view(N, -1).to(torch.float32).numpy()

        # ----------------------------------------------------------------------
        # C. 메타 데이터 및 배치 생성
        # ----------------------------------------------------------------------
        env_id_np = np.arange(N, dtype=np.int16).reshape(N, 1)
        ep_id_np  = self._ep_id.astype(np.int64).reshape(N, 1)
        t_np      = self._t_in_ep.astype(np.int32).reshape(N, 1)

        # Structured Array 생성 (size=N)
        batch = np.empty((N,), dtype=self._dtype)
        
        batch["observations"]      = obs_np
        batch["actions"]           = act_np
        batch["rewards"]           = rew_np
        batch["dones"]             = done_np
        batch["timeouts"]          = tout_np
        batch["next_observations"] = nxt_np
        batch["env_id"]            = env_id_np
        batch["episode_id"]        = ep_id_np
        batch["t_in_episode"]      = t_np

        # 옵션 필드 채우기
        if self.include_cbf:
            # batch["cbf"]는 (N, 1)을 기대 -> cbf_np는 (N, 1). OK.
            batch["cbf"] = cbf_np  
        
        if self.include_ee:
            # batch["ee_pos"]는 (N, 7)을 기대 -> ee_np는 (N, 7). OK.
            batch["ee_pos"] = ee_np
            
        if camera_data:
            for cam_name, cam_cfg in self.camera_config.items():
                if cam_name not in camera_data:
                    continue # 데이터가 안 들어왔으면 스킵 (혹은 에러 처리)

                data_dict = camera_data[cam_name] # {'rgb': Tensor, 'depth': Tensor}
                shape_hw = cam_cfg["shape"]

                # RGB 처리
                if cam_cfg.get("rgb") and "rgb" in data_dict:
                    # key name: "tv_cam_rgb"
                    key = f"{cam_name}_rgb"
                    # shape: (H, W, 3)
                    target_shape = tuple(shape_hw) + (3,)
                    processed = self._process_image(data_dict["rgb"], target_shape, is_depth=False)
                    batch[key] = processed.astype(np.uint8)

                # Depth 처리
                if cam_cfg.get("depth") and "depth" in data_dict:
                    key = f"{cam_name}_depth"
                    target_shape = tuple(shape_hw) + (1,)
                    processed = self._process_image(data_dict["depth"], target_shape, is_depth=True)
                    
                    dtype_str = cam_cfg.get("depth_dtype", "float16")
                    ddt = np.float16 if dtype_str == "float16" else np.float32
                    batch[key] = processed.astype(ddt)


        # print(f"prrint batch {batch}")
        self._pending.append(batch)

        # ----------------------------------------------------------------------
        # D. 에피소드 관리 및 Flush
        # ----------------------------------------------------------------------
        ended = (done_np[:, 0] | tout_np[:, 0])
        ended_eps = int(np.sum(ended))
        self._pending_ended_episodes += ended_eps

        # Tracker 업데이트
        if np.any(ended):
            idx = np.nonzero(ended)[0]
            self._ep_id[idx] += 1
            self._t_in_ep[idx] = 0
        self._t_in_ep[~ended] += 1

        flushed_now = False
        if self.auto_flush and self.flush_every_episodes > 0:
            if self._pending_ended_episodes >= self.flush_every_episodes:
                self.flush_to_zarr()
                flushed_now = True

        return flushed_now

    def _process_image(self, tensor: torch.Tensor, target_shape: Tuple, is_depth: bool) -> np.ndarray:
        """
        입력 텐서를 타겟 shape에 맞게 변환 (차원 추가/순서 변경 등)
        """
        # (N, ...) -> CPU numpy
        arr = tensor.detach().cpu().numpy() # float32 or whatever
        
        # 여기서 차원 검사 및 unsqueeze 로직은 기존 _normalize_depth 처럼 구현
        # 예: Depth (N, H, W) -> (N, H, W, 1)
        if is_depth and len(target_shape) == 3 and target_shape[-1] == 1:
             if arr.ndim == 3: # (N, H, W)
                 arr = arr[..., None]
        
        # RGB (N, C, H, W) -> (N, H, W, C) Isaac Lab은 보통 Channel First일 수 있음
        if not is_depth and arr.ndim == 4 and arr.shape[1] == 3:
            arr = np.transpose(arr, (0, 2, 3, 1))

        return arr

    def flush_to_zarr(self):
        if not self._pending:
            return

        # =========================
        # NEW: timing window end
        # =========================
        t0 = self._t_flush_window_start
        t1 = time.perf_counter()
        elapsed = max(0.0, t1 - t0)

        pending_steps = len(self._pending)                 # flush 구간 step 수
        pending_transitions = pending_steps * self.num_envs
        pending_episodes = int(self._pending_ended_episodes)

        big = np.concatenate(self._pending, axis=0)  # (pending_steps * N,)
        self.ds.append(big)

        self._total_flush_count += 1

        # throughput (optional display)
        steps_per_s = (pending_steps / elapsed) if elapsed > 0 else float("inf")
        trans_per_s = (pending_transitions / elapsed) if elapsed > 0 else float("inf")

        print(
            "[INFO] Flush done | "
            f"flush#{self._total_flush_count} | "
            f"elapsed={elapsed:.3f}s | "
            f"pending_steps={pending_steps} | "
            f"pending_transitions={pending_transitions} | "
            f"ended_episodes={pending_episodes} | "
            f"throughput={steps_per_s:.1f} steps/s, {trans_per_s:.1f} trans/s | "
            f"total_transitions={len(self.ds)}"
        )

        self._pending.clear()
        self._pending_ended_episodes = 0

        try:
            self.root.store.flush()
        except Exception:
            pass

        # =========================
        # NEW: reset timing window
        # =========================
        self._t_flush_window_start = time.perf_counter()


    def get_total_flush_count(self) -> int:
        return self._total_flush_count