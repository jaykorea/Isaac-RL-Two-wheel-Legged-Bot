#!/usr/bin/env python3
# eval.py
import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Headless plot backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.utils.dataset_loader import CbfDataset
from network.model import CbfCrossAttnTransformer
from network.utils.zarr_utils import open_zarr_group, get_transitions_dataset


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


@torch.inference_mode()
def evaluate(model, loader, device, *, collect_max_points: int = 50000, seed: int = 0):
    """
    기존 기능 유지:
      - mse_sum, mae_sum을 샘플 수(n=batch size 누적)로 나눠 반환하는 흐름 유지

    추가 기능:
      - RMSE, Max|err| 등 산출
      - 플롯 생성을 위한 (y, pred, err) 샘플링 수집
      - 배치별 RMSE 추이 수집
    """
    model.train()

    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    max_abs_err = 0.0

    # Plot용 샘플링 (최대 collect_max_points 개)
    rng = np.random.default_rng(seed)
    y_samples = []
    pred_samples = []
    err_samples = []

    # 배치별 RMSE 추이
    batch_rmse = []

    collected = 0

    for depth, ee, y in tqdm(loader, desc="eval", dynamic_ncols=True):
        depth = depth.to(device, non_blocking=True)
        ee = ee.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(depth, ee)
        diff = pred - y

        # ---- 기존 출력과 동일한 스케일 유지(샘플 기준) ----
        # (y가 스칼라 타깃이라는 전제에 가까운 구현이지만, 기존 로직 변경하지 않음)
        mse_sum += float((diff * diff).sum().item())
        mae_sum += float(diff.abs().sum().item())
        n += int(y.shape[0])

        # ---- 추가 통계 ----
        batch_max_abs = float(diff.abs().max().item())
        if batch_max_abs > max_abs_err:
            max_abs_err = batch_max_abs

        # 배치 RMSE(샘플 기준 n으로 나누는 기존 관례를 유지하려면,
        # 배치 내 요소 수가 y.shape[0]라고 가정)
        # y가 (B,1) 등일 때도 sum이므로 동일하게 동작.
        b = int(y.shape[0])
        if b > 0:
            b_mse = float((diff * diff).sum().item()) / float(b)
            batch_rmse.append(math.sqrt(max(0.0, b_mse)))

        # ---- 플롯용 샘플 수집(무작위 서브샘플) ----
        if collect_max_points > 0 and collected < collect_max_points:
            # CPU로 이동 (메모리 부담 줄이기 위해 필요한 만큼만)
            # 다차원 타깃이면 1차원으로 flatten하여 포인트를 구성
            y_cpu = y.detach().float().cpu().view(-1).numpy()
            pred_cpu = pred.detach().float().cpu().view(-1).numpy()
            err_cpu = (pred.detach().float().cpu() - y.detach().float().cpu()).view(-1).numpy()

            remaining = collect_max_points - collected
            m = y_cpu.shape[0]

            if m <= remaining:
                take_idx = np.arange(m)
            else:
                take_idx = rng.choice(m, size=remaining, replace=False)

            y_samples.append(y_cpu[take_idx])
            pred_samples.append(pred_cpu[take_idx])
            err_samples.append(err_cpu[take_idx])

            collected += int(take_idx.shape[0])

    mse = mse_sum / max(1, n)
    mae = mae_sum / max(1, n)
    rmse = math.sqrt(max(0.0, mse))

    # 샘플 concat
    if len(y_samples) > 0:
        y_s = np.concatenate(y_samples, axis=0)
        p_s = np.concatenate(pred_samples, axis=0)
        e_s = np.concatenate(err_samples, axis=0)
    else:
        y_s = np.array([], dtype=np.float32)
        p_s = np.array([], dtype=np.float32)
        e_s = np.array([], dtype=np.float32)

    out = {
        "mse": _safe_float(mse),
        "mae": _safe_float(mae),
        "rmse": _safe_float(rmse),
        "max_abs_err": _safe_float(max_abs_err),
        "n_samples": int(n),
        "plot_points_collected": int(y_s.shape[0]),
        "batch_rmse": batch_rmse,  # list[float]
        "plot_arrays": {
            "y": y_s,
            "pred": p_s,
            "err": e_s,
        },
    }
    return out


def _save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _plot_and_save(eval_dir: Path, tag: str, results: dict):
    """
    results: evaluate() 반환 dict
    """
    y = results["plot_arrays"]["y"]
    pred = results["plot_arrays"]["pred"]
    err = results["plot_arrays"]["err"]
    batch_rmse = results["batch_rmse"]

    # 1) Pred vs True scatter
    if y.size > 0 and pred.size > 0:
        plt.figure(figsize=(7, 6))
        plt.scatter(y, pred, s=4, alpha=0.35)
        # y=x reference line (범위 기반)
        lo = float(np.min([y.min(), pred.min()]))
        hi = float(np.max([y.max(), pred.max()]))
        plt.plot([lo, hi], [lo, hi], linewidth=1.0)
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.title("Pred vs True")
        plt.tight_layout()
        plt.savefig(eval_dir / f"plot_pred_vs_true__{tag}.png", dpi=150)
        plt.close()

    # 2) Error histogram
    if err.size > 0:
        plt.figure(figsize=(7, 5))
        plt.hist(err, bins=80)
        plt.xlabel("Error (pred - true)")
        plt.ylabel("Count")
        plt.title("Error Histogram")
        plt.tight_layout()
        plt.savefig(eval_dir / f"plot_error_hist__{tag}.png", dpi=150)
        plt.close()

    # 3) Batch RMSE curve
    if batch_rmse is not None and len(batch_rmse) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(batch_rmse)
        plt.xlabel("Batch index")
        plt.ylabel("RMSE (per-batch)")
        plt.title("Batch RMSE")
        plt.tight_layout()
        plt.savefig(eval_dir / f"plot_batch_rmse__{tag}.png", dpi=150)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zarr_path", type=str)
    p.add_argument("--ckpt", type=str, required=True, help="path to best.pt (or last.pt)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--limit", type=int, default=None, help="optional: limit number of samples for quick test")

    # 추가: 플롯/샘플링 관련 옵션 (기능에 영향 없음)
    p.add_argument("--max_plot_points", type=int, default=50000,
                   help="max number of (y,pred,err) points to collect for plots (default: 50000)")
    args = p.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("checkpoint does not contain 'config'. (train.py에서 저장하도록 되어 있어야 함)")

    # dataset size
    g = open_zarr_group(args.zarr_path)
    ds = get_transitions_dataset(g)
    T = int(ds.shape[0])
    if T <= 0:
        raise RuntimeError("empty transitions")

    idx_all = np.arange(T, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_all)
    n_val = max(1, int(0.1 * T))
    idx_val = idx_all[:n_val]
    idx_train = idx_all[n_val:]

    if args.split == "train":
        idx = idx_train
    elif args.split == "val":
        idx = idx_val
    else:
        idx = idx_all

    if args.limit is not None:
        idx = idx[: int(args.limit)]

    depth_clip = (float(cfg["depth_clip_lo"]), float(cfg["depth_clip_hi"]))

    dset = CbfDataset(
        args.zarr_path,
        idx,
        depth_clip=depth_clip,
        depth_scale=float(cfg["depth_scale"]),
        depth_downsample=int(cfg["depth_downsample"]),
    )
    depth_hw = (dset.H_ds, dset.W_ds)

    model = CbfCrossAttnTransformer(
        depth_hw=depth_hw,
        d_model=int(cfg["d_model"]),
        patch=int(cfg["patch"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        mlp_ratio=float(cfg["mlp_ratio"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # ---- 평가 ----
    results = evaluate(
        model,
        loader,
        device,
        collect_max_points=int(args.max_plot_points),
        seed=int(args.seed),
    )

    mse = results["mse"]
    mae = results["mae"]
    rmse = results["rmse"]

    print(f"[RESULT] split={args.split} | mse={mse:.8f} | mae={mae:.8f} | rmse={rmse:.8f} | n={len(dset)}")

    # ---- 저장 경로: ckpt가 있는 폴더 아래 eval/ ----
    eval_dir = ckpt_path.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 파일 태그(충돌 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_tag = "all" if args.limit is None else f"limit{int(args.limit)}"
    tag = f"{ckpt_path.stem}__{args.split}__{limit_tag}__seed{int(args.seed)}__{timestamp}"

    # ---- metrics 저장(txt/json) ----
    metrics = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt_path.resolve()),
        "zarr_path": str(Path(args.zarr_path).resolve()),
        "device": device,
        "split": args.split,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "n_dataset": int(len(dset)),
        "mse": results["mse"],
        "mae": results["mae"],
        "rmse": results["rmse"],
        "max_abs_err": results["max_abs_err"],
        "plot_points_collected": results["plot_points_collected"],
        "max_plot_points": int(args.max_plot_points),
        # config 일부를 함께 남겨두면 재현에 유리
        "config": cfg,
    }

    txt_lines = []
    txt_lines.append(f"time_local: {metrics['time_local']}")
    txt_lines.append(f"ckpt: {metrics['ckpt']}")
    txt_lines.append(f"zarr_path: {metrics['zarr_path']}")
    txt_lines.append(f"device: {metrics['device']}")
    txt_lines.append(f"split: {metrics['split']}")
    txt_lines.append(f"limit: {metrics['limit']}")
    txt_lines.append(f"batch_size: {metrics['batch_size']}")
    txt_lines.append(f"num_workers: {metrics['num_workers']}")
    txt_lines.append(f"seed: {metrics['seed']}")
    txt_lines.append(f"n_dataset: {metrics['n_dataset']}")
    txt_lines.append("")
    txt_lines.append(f"MSE: {metrics['mse']:.10f}")
    txt_lines.append(f"MAE: {metrics['mae']:.10f}")
    txt_lines.append(f"RMSE: {metrics['rmse']:.10f}")
    txt_lines.append(f"Max|error|: {metrics['max_abs_err']:.10f}")
    txt_lines.append("")
    txt_lines.append(f"plot_points_collected: {metrics['plot_points_collected']}")
    txt_lines.append(f"max_plot_points: {metrics['max_plot_points']}")
    txt_lines.append("")
    txt_lines.append("config:")
    for k, v in cfg.items():
        txt_lines.append(f"  {k}: {v}")

    _save_text(eval_dir / f"metrics__{tag}.txt", "\n".join(txt_lines))
    _save_json(eval_dir / f"metrics__{tag}.json", metrics)

    # ---- plot 저장 ----
    _plot_and_save(eval_dir, tag, results)

    print(f"[INFO] saved eval artifacts to: {str(eval_dir.resolve())}")


if __name__ == "__main__":
    main()
