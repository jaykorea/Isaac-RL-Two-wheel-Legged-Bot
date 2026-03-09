#!/usr/bin/env python3
# train.py
import argparse
import os
import time
import gc  # [Added] 메모리 관리를 위한 가비지 컬렉터
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from network.utils.dataset_loader import MultiModalDataset
from network.model import MultiModalTransformer
from network.utils.zarr_utils import open_zarr_group, get_transitions_dataset


# -----------------------------
# Utilities
# -----------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _squeeze_last(x: np.ndarray) -> np.ndarray:
    if x.ndim >= 1 and x.shape[-1] == 1:
        return x[..., 0]
    return x

def split_indices(
    ds,
    seed: int,
    val_ratio: float = 0.1,
    by_episode: bool = True,
    read_batch: int = 10_000,  # [메모리 최적화] 배치 사이즈 대폭 감소
):
    T = int(ds.shape[0])
    if T <= 0:
        raise RuntimeError("empty transitions")

    rng = np.random.default_rng(seed)
    val_ratio = float(val_ratio)
    val_ratio = min(max(val_ratio, 1e-6), 0.999999)

    # 단순 랜덤 스플릿 (by_episode=False)
    if not by_episode:
        idx_all = np.arange(T, dtype=np.int64)
        rng.shuffle(idx_all)
        n_val = max(1, int(val_ratio * T))
        return idx_all[n_val:], idx_all[:n_val]

    # Zarr array access optimization
    is_ram = isinstance(ds, np.ndarray)

    if is_ram:
        # RAM에 이미 다 올라와 있는 경우 (빠름)
        env_ids = _squeeze_last(ds["env_id"]).astype(np.int64)
        ep_ids = _squeeze_last(ds["episode_id"]).astype(np.int64)
    else:
        # Disk 모드인 경우 (메모리 절약 필수)
        env_ids = np.empty((T,), dtype=np.int64)
        ep_ids = np.empty((T,), dtype=np.int64)

        print(f"[INFO] Scanning episode IDs for split (Batch: {read_batch})...")
        
        for s in tqdm(range(0, T, read_batch), desc="Splitting indices"):
            e = min(T, s + read_batch)
            
            # [메모리 최적화] 컬럼별 접근 시도
            try:
                e_chunk = ds["env_id"][s:e]
                ep_chunk = ds["episode_id"][s:e]
            except (TypeError, KeyError, IndexError):
                # 컬럼 접근 불가 시 row 전체 로드 후 즉시 해제
                b = ds[s:e]
                e_chunk = b["env_id"]
                ep_chunk = b["episode_id"]
                del b # [중요] 임시 객체 명시적 삭제
            
            env_ids[s:e] = _squeeze_last(e_chunk).astype(np.int64)
            ep_ids[s:e] = _squeeze_last(ep_chunk).astype(np.int64)
        
        # [중요] 스캔 후 메모리 정리
        gc.collect()

    keys = env_ids * 1_000_000_000_000 + ep_ids
    uniq_keys = np.unique(keys)
    rng.shuffle(uniq_keys)

    n_val_keys = max(1, int(val_ratio * len(uniq_keys)))
    val_keys = uniq_keys[:n_val_keys]

    is_val = np.isin(keys, val_keys)
    
    idx_all = np.arange(T, dtype=np.int64)
    train_idx = idx_all[~is_val]
    val_idx = idx_all[is_val]

    print(f"[INFO] Split done. Train: {len(train_idx)}, Val: {len(val_idx)} (by episodes)")
    return train_idx, val_idx


# -----------------------------
# Train config
# -----------------------------
@dataclass
class TrainConfig:
    zarr_path: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    amp: bool
    preload_ram: int
    partitions: int

    patch: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    mlp_ratio: float

    depth_clip_lo: float
    depth_clip_hi: float
    depth_scale: float
    depth_downsample: int

    num_workers: int
    log_every: int

    split_by_episode: bool
    val_ratio: float
    es_patience: int
    es_min_delta: float

    camera_config : Dict[str, Any]


# -----------------------------
# Training Logic Helper
# -----------------------------
def run_train_phase(
    loader: DataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: str,
    amp_enabled: bool,
    log_every: int,
    epoch_desc: str = ""
) -> Tuple[float, float, int]:
    
    model.train()
    running_mse = 0.0
    running_mae = 0.0
    seen = 0
    t0 = time.perf_counter()

    tbar = tqdm(loader, desc=epoch_desc, dynamic_ncols=True, leave=False)

    for it, (images, ee, y) in enumerate(tbar, start=1):
        images = {k: v.to(device, non_blocking=True) for k, v in images.items()}
        ee = ee.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(images, ee)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            diff = pred - y
            mse = float((diff * diff).mean().item())
            mae = float(diff.abs().mean().item())
            running_mse += mse * y.shape[0]
            running_mae += mae * y.shape[0]
            seen += int(y.shape[0])

        if log_every > 0 and it % log_every == 0:
            elapsed = max(1e-9, time.perf_counter() - t0)
            steps_per_s = it / elapsed
            tbar.set_postfix(
                mse=f"{(running_mse/max(1,seen)):.4f}",
                mae=f"{(running_mae/max(1,seen)):.4f}",
                fps=f"{steps_per_s:.1f}",
            )

    return running_mse, running_mae, seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", type=str, default="/home/jay/Flamingo_RL/isaaclab_flamingo/lab/flamingo/tasks/manager_based/manipulation/test/dataset/logs/lift_pick_and_lift_sm_20260215_230551", help="Path to the Zarr dataset")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=30)
    p.add_argument("--batch", "--batch_size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", "--weight_decay", dest="weight_decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")

    # [Memory Option]
    p.add_argument("--preload_ram", type=int, default=1, 
                   help="0: Disk Mode, 1: Hybrid Mode (Chunked RAM), 2: Full RAM Mode")
    p.add_argument("--partitions", type=int, default=4, 
                   help="Number of chunks for Hybrid Mode (higher = less RAM usage)")

    # Model
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", dest="n_layers", type=int, default=4)
    p.add_argument("--heads", dest="n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    # Preprocessing
    p.add_argument("--depth_clip_lo", type=float, default=None)
    p.add_argument("--depth_clip_hi", type=float, default=None)
    p.add_argument("--depth_scale", type=float, default=4.0)
    p.add_argument("--depth_downsample", type=int, default=1)

    # System
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=25)

    # Split & ES
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_by_episode", action="store_true")
    p.add_argument("--es_patience", type=int, default=5)
    p.add_argument("--es_min_delta", type=float, default=1e-7)

    args = p.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    
    base = os.path.basename(os.path.normpath(args.zarr_path))
    
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(script_dir, "logs", base)
        
    ensure_dir(out_dir)
    print(f"[INFO] out_dir={out_dir}")

    # 1. Open Zarr Handle
    g = open_zarr_group(args.zarr_path)
    ds_disk = get_transitions_dataset(g)
    T = int(ds_disk.shape[0])
    if T <= 0:
        raise RuntimeError("empty transitions")

    # 2. Split Indices (메모리 최적화 적용됨)
    idx_train_all, idx_val = split_indices(
        ds_disk,
        seed=args.seed,
        val_ratio=args.val_ratio,
        by_episode=bool(args.split_by_episode),
    )
    print(
        f"[INFO] samples: total={T}, train={len(idx_train_all)}, val={len(idx_val)} "
        f"(mode={args.preload_ram}, partitions={args.partitions if args.preload_ram==1 else 'N/A'})"
    )

    camera_config = g.attrs.get("camera_config", {})
    if not camera_config:
        print("[WARN] No camera_config in zarr attrs. Assuming legacy single depth.")
        camera_config = {"default": {"depth": True, "shape": (240, 320)}}

    print(f"[INFO] Camera Config Detected: {list(camera_config.keys())}")
    
    model_input_configs = {}
    for cam_name, cfg in camera_config.items():
        shape_hw = cfg.get("shape", (128, 128))
        ds_factor = args.depth_downsample if args.depth_downsample > 0 else 1
        
        if cfg.get("rgb"):
            key = f"{cam_name}_rgb"
            model_input_configs[key] = {"shape": shape_hw, "ch": 3}
            
        if cfg.get("depth"):
            key = f"{cam_name}_depth"
            orig_h, orig_w = shape_hw
            new_h = (orig_h + ds_factor - 1) // ds_factor
            new_w = (orig_w + ds_factor - 1) // ds_factor
            model_input_configs[key] = {"shape": (new_h, new_w), "ch": 1}
    
    model = MultiModalTransformer(
        input_configs=model_input_configs,
        d_model=args.d_model,
        patch=args.patch,
        n_layers=args.n_layers, 
        n_heads=args.n_heads,
        dropout=0.1
    ).to(device)
    
    print(f"[INFO] Model initialized with inputs: {list(model_input_configs.keys())}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # -------------------------------------------------------------------------
    # 4. Validation Loader 설정
    # - preload_ram == 1 or 2 : val을 RAM으로 1회 preload (속도 우선)
    # - preload_ram == 0      : val은 DISK 유지 (RAM 최소)
    # -------------------------------------------------------------------------
    if args.preload_ram in (1, 2):
        print(f"[INFO] Validation set: Preloading to RAM (once) (mode={args.preload_ram})")
        val_data_source = ds_disk[idx_val]                      # 1회 RAM 로드
        val_indices = np.arange(len(idx_val), dtype=np.int64)   # local index                                # RAM이면 보통 0이 빠름
    else:
        print(f"[INFO] Validation set: Keeping on DISK (mode={args.preload_ram})")
        val_data_source = ds_disk
        val_indices = idx_val

    # Clip Args
    clip_args = None
    if args.depth_clip_lo is not None and args.depth_clip_hi is not None:
        clip_args = (args.depth_clip_lo, args.depth_clip_hi)

    val_set = MultiModalDataset(
        val_data_source,
        val_indices,
        camera_config=camera_config,
        depth_clip=clip_args,
        depth_scale=args.depth_scale,
        depth_downsample=args.depth_downsample
    )

    # RAM 기반 val이면 num_workers=0이 보통 더 빠름(IPC/피클링 없음)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        persistent_workers=False
    )

    # Save Config
    cfg = TrainConfig(
        zarr_path=args.zarr_path,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        amp=args.amp,
        preload_ram=args.preload_ram,
        partitions=args.partitions,
        patch=args.patch,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        depth_clip_lo=args.depth_clip_lo,
        depth_clip_hi=args.depth_clip_hi,
        depth_scale=args.depth_scale,
        depth_downsample=args.depth_downsample,
        num_workers=args.num_workers,
        log_every=args.log_every,
        split_by_episode=bool(args.split_by_episode),
        val_ratio=float(args.val_ratio),
        es_patience=int(args.es_patience),
        es_min_delta=float(args.es_min_delta),
        camera_config=camera_config,
    )
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")

    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    def save_ckpt(path: str, epoch: int, best_val_mse: float):
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val_mse": best_val_mse,
                "config": asdict(cfg),
            },
            path,
        )

    def run_val() -> tuple[float, float]:
        model.eval()
        v_mse_sum = 0.0
        v_mae_sum = 0.0
        v_seen = 0

        vbar = tqdm(val_loader, desc="val", dynamic_ncols=True, leave=False)
        with torch.inference_mode():
            for images, ee, y in vbar:
                images = {k: v.to(device, non_blocking=True) for k, v in images.items()}
                ee = ee.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(images, ee)
                diff = pred - y
                mse = float((diff * diff).mean().item())
                mae = float(diff.abs().mean().item())

                v_mse_sum += mse * y.shape[0]
                v_mae_sum += mae * y.shape[0]
                v_seen += int(y.shape[0])

        return (v_mse_sum / max(1, v_seen), v_mae_sum / max(1, v_seen))

    # --- Training Loop Setup ---
    ds_full_ram = None
    if args.preload_ram == 2:
        print(f"[INFO] Preloading FULL dataset into RAM ({len(idx_train_all)} samples included)...")
        t0 = time.time()
        ds_full_ram = ds_disk[:] 
        print(f"[INFO] Full RAM load done in {time.time() - t0:.2f}s")

    es_pat = int(args.es_patience)
    es_delta = float(args.es_min_delta)
    es_bad = 0

    try:
        for ep in range(1, args.epochs + 1):
            epoch_mse_sum = 0.0
            epoch_mae_sum = 0.0
            epoch_seen = 0
            amp_enabled = bool(args.amp and device == "cuda")

            if args.preload_ram == 1:
                # [Hybrid Mode]
                shuffled_idx = np.random.permutation(idx_train_all)
                chunks = np.array_split(shuffled_idx, args.partitions)
                
                print(f"[EPOCH {ep:03d}] Hybrid: Processing {len(chunks)} chunks...")
                
                for i, chunk_idx in enumerate(chunks):
                    if len(chunk_idx) == 0: continue
                    read_idx = np.sort(chunk_idx)
                    
                    # 1. Load Chunk
                    block_data = ds_disk[read_idx] 
                    
                    # 2. Dataset
                    chunk_ds = MultiModalDataset(
                        block_data, 
                        np.arange(len(block_data)),
                        camera_config=camera_config,
                        depth_clip=clip_args,
                        depth_scale=args.depth_scale,
                        depth_downsample=args.depth_downsample
                    )
                    
                    chunk_loader = DataLoader(
                        chunk_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device == "cuda"),
                        drop_last=True
                    )
                    
                    # 3. Train
                    c_mse, c_mae, c_n = run_train_phase(
                        chunk_loader, model, opt, loss_fn, scaler, device, amp_enabled,
                        log_every=args.log_every,
                        epoch_desc=f"Ep {ep} | Chunk {i+1}/{len(chunks)}"
                    )
                    
                    epoch_mse_sum += c_mse
                    epoch_mae_sum += c_mae
                    epoch_seen += c_n
                    
                    # 4. Explicit Cleanup & GC
                    del block_data, chunk_ds, chunk_loader
                    gc.collect()

            elif args.preload_ram == 2:
                # [Full RAM]
                train_set = MultiModalDataset(
                    ds_full_ram, 
                    idx_train_all,
                    camera_config=camera_config,
                    depth_clip=clip_args,
                    depth_scale=args.depth_scale,
                    depth_downsample=args.depth_downsample
                )
                loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=(device == "cuda"),
                    drop_last=True,
                    persistent_workers=(args.num_workers > 0)
                )
                c_mse, c_mae, c_n = run_train_phase(
                    loader, model, opt, loss_fn, scaler, device, amp_enabled,
                    log_every=args.log_every,
                    epoch_desc=f"Ep {ep} [FullRAM]"
                )
                epoch_mse_sum, epoch_mae_sum, epoch_seen = c_mse, c_mae, c_n

            else:
                # [Disk Mode]
                train_set = MultiModalDataset(
                    ds_disk, 
                    idx_train_all,
                    camera_config=camera_config,
                    depth_clip=clip_args,
                    depth_scale=args.depth_scale,
                    depth_downsample=args.depth_downsample
                )
                
                loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=(device == "cuda"),
                    drop_last=True,
                    persistent_workers=(args.num_workers > 0)
                )
                c_mse, c_mae, c_n = run_train_phase(
                    loader, model, opt, loss_fn, scaler, device, amp_enabled,
                    log_every=args.log_every,
                    epoch_desc=f"Ep {ep} [Disk]"
                )
                epoch_mse_sum, epoch_mae_sum, epoch_seen = c_mse, c_mae, c_n

            # Stats & Val
            train_mse = epoch_mse_sum / max(1, epoch_seen)
            train_mae = epoch_mae_sum / max(1, epoch_seen)

            val_mse, val_mae = run_val()

            save_ckpt(last_path, ep, best_val)

            improved = (val_mse < (best_val - es_delta))
            if improved:
                best_val = val_mse
                save_ckpt(best_path, ep, best_val)
                es_bad = 0
            else:
                es_bad += 1

            print(
                f"[EPOCH {ep:03d}] "
                f"train_mse={train_mse:.6f}, train_mae={train_mae:.6f} | "
                f"val_mse={val_mse:.6f}, val_mae={val_mae:.6f} | "
                f"best_val={best_val:.6f} | "
                f"es_bad={es_bad}/{es_pat}"
            )

            if es_pat > 0 and es_bad >= es_pat:
                print(f"[EARLY STOP] no improvement for {es_pat} epochs (min_delta={es_delta}).")
                break

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt detected. Saving last checkpoint...")
        try:
            save_ckpt(last_path, ep if "ep" in locals() else 0, best_val)
            print(f"[INFO] saved: {last_path}")
        except Exception as e:
            print(f"[ERR] failed to save last checkpoint: {e}")

    print(f"[DONE] best={best_path}, last={last_path}")


if __name__ == "__main__":
    main()