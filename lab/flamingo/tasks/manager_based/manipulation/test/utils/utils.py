import torch
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms

import numpy as np
import matplotlib.pyplot as plt


def sample_polyline(start, w1, w2, goal, n_per_seg=20):
    def lerp(a, b, t):
        return a * (1 - t) + b * t

    ts = torch.linspace(0, 1, n_per_seg, device=start.device)
    p01 = torch.stack([lerp(start, w1, t) for t in ts], dim=0)
    p12 = torch.stack([lerp(w1, w2, t) for t in ts], dim=0)
    p2g = torch.stack([lerp(w2, goal, t) for t in ts], dim=0)
    pts = torch.cat([p01, p12, p2g], dim=0)  # (3*n_per_seg, 3)
    return pts


def plan_detour_two_waypoints(
    start_pos_w: torch.Tensor,  # (3,)
    goal_pos_w: torch.Tensor,   # (3,)
    obs_pos_w: torch.Tensor,    # (3,)
    obs_quat_w: torch.Tensor,   # (4,)  (w,x,y,z)
    size_x: float,
    size_y: float,
    margin: float = 0.05,
):
    """
    Screen 로컬에서 2D 우회 경로 (start->w1->w2->goal)를 만든다.
    반환: w1_pos_w, w2_pos_w (둘 다 (3,))
    """
    device = start_pos_w.device

    # start/goal을 obstacle local frame으로 변환
    # subtract_frame_transforms(A_frame, B_frame) = A^-1 * B
    # 여기서는 obstacle frame 기준으로 point를 표현하고 싶으니:
    # obs^-1 * point_w 를 만들면 됨.
    start_pos_o, _ = subtract_frame_transforms(
        obs_pos_w.unsqueeze(0), obs_quat_w.unsqueeze(0),
        start_pos_w.unsqueeze(0), torch.tensor([1,0,0,0], device=device).unsqueeze(0)
    )
    goal_pos_o, _ = subtract_frame_transforms(
        obs_pos_w.unsqueeze(0), obs_quat_w.unsqueeze(0),
        goal_pos_w.unsqueeze(0), torch.tensor([1,0,0,0], device=device).unsqueeze(0)
    )
    start_o = start_pos_o[0]  # (3,)
    goal_o  = goal_pos_o[0]   # (3,)

    hx = 0.5 * size_x + margin
    hy = 0.5 * size_y + margin

    sx, sy = start_o[0].item(), start_o[1].item()
    gx, gy = goal_o[0].item(), goal_o[1].item()

    # 4 후보 (로컬)
    cand = []
    # +x, -x
    cand.append((torch.tensor([+hx, sy, start_o[2].item()], device=device),
                 torch.tensor([+hx, gy, goal_o[2].item()],  device=device)))
    cand.append((torch.tensor([-hx, sy, start_o[2].item()], device=device),
                 torch.tensor([-hx, gy, goal_o[2].item()],  device=device)))
    # +y, -y
    cand.append((torch.tensor([sx, +hy, start_o[2].item()], device=device),
                 torch.tensor([gx, +hy, goal_o[2].item()],  device=device)))
    cand.append((torch.tensor([sx, -hy, start_o[2].item()], device=device),
                 torch.tensor([gx, -hy, goal_o[2].item()],  device=device)))

    # 길이 최소 후보 선택 (로컬에서)
    best_L = None
    best = None
    for w1_o, w2_o in cand:
        L = (torch.norm(start_o - w1_o) +
             torch.norm(w1_o - w2_o) +
             torch.norm(w2_o - goal_o))
        if (best_L is None) or (L < best_L):
            best_L = L
            best = (w1_o, w2_o)

    w1_o, w2_o = best

    # obstacle local -> world 로 변환: obs * w_o
    w1_w, _ = combine_frame_transforms(
        obs_pos_w.unsqueeze(0), obs_quat_w.unsqueeze(0),
        w1_o.unsqueeze(0), torch.tensor([1,0,0,0], device=device).unsqueeze(0)
    )
    w2_w, _ = combine_frame_transforms(
        obs_pos_w.unsqueeze(0), obs_quat_w.unsqueeze(0),
        w2_o.unsqueeze(0), torch.tensor([1,0,0,0], device=device).unsqueeze(0)
    )

    return w1_w[0], w2_w[0]

def sample_quadratic_bezier(p0: torch.Tensor, pc: torch.Tensor, p1: torch.Tensor, n: int = 60) -> torch.Tensor:
    """
    Quadratic Bezier curve sampling.
    Supports:
      - single: p0,pc,p1 shape (3,) -> returns (n,3)
      - batch : p0,pc,p1 shape (N,3) -> returns (N,n,3)
    """
    assert p0.shape == pc.shape == p1.shape, "p0, pc, p1 must have same shape"
    assert p0.shape[-1] == 3, "last dim must be 3"
    assert n >= 2, "n must be >= 2"

    # t: (n,)
    t = torch.linspace(0.0, 1.0, n, device=p0.device, dtype=p0.dtype)

    if p0.dim() == 1:
        # (3,) -> (n,1)
        t = t.view(n, 1)
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p1

    elif p0.dim() == 2:
        # (N,3) -> broadcast with (1,n,1)
        t = t.view(1, n, 1)
        p0 = p0.view(-1, 1, 3)
        pc = pc.view(-1, 1, 3)
        p1 = p1.view(-1, 1, 3)
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p1

    else:
        raise ValueError(f"Unsupported dim: {p0.dim()}")
    
class CameraViz:
    """
    One window per instance.
    - mode="rgb": expects camera_rgb (N,H,W,C) or (N,C,H,W)
    - mode="depth": expects camera_depth (N,H,W) or (N,1,H,W) or (N,H,W,1)
    """

    def __init__(self, H=480, W=640, title="camera", env_id=0, mode="rgb"):
        self.env_id = int(env_id)
        self.mode = str(mode).lower()
        assert self.mode in ("rgb", "depth")

        self.title_base = str(title)

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1)

        if self.mode == "rgb":
            self.im = self.ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
        else:
            self.im = self.ax.imshow(
                np.zeros((H, W), dtype=np.float32),
                cmap="viridis",
                vmin=0.0, vmax=1.0
            )

        self.ax.set_title(f"{self.title_base} (env{self.env_id})")
        self.ax.axis("off")
        self.fig.show()

        # depth display options (only used if mode == "depth")
        self.depth_auto_scale = False
        self.depth_vmin = None
        self.depth_vmax = None
        self.depth_percentile = (2.0, 98.0)

        # ✅ “보기 좋은 depth”를 위한 옵션 (meters 기준)
        self.depth_near = 0.1
        self.depth_far = 4.0
        self.depth_invert = False  # 가까울수록 밝게

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _to_hwc_rgb_uint8(img_t):
        x = img_t
        if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
            x = x.permute(1, 2, 0)
        if x.ndim == 3 and x.shape[-1] == 4:
            x = x[..., :3]
        img = x.detach().cpu().numpy()
        if img.dtype != np.uint8:
            mx = float(img.max()) if img.size > 0 else 0.0
            if mx <= 1.0 + 1e-6:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _to_hw_depth_float32(depth_t):
        d = depth_t
        if d.ndim == 3:
            if d.shape[0] == 1:
                d = d[0]
            elif d.shape[-1] == 1:
                d = d[..., 0]
        return d.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _depth_to_vis01(d_np, near, far, invert=True):
        """
        d_np: np.float32 meters, shape (H,W)
        return: np.float32 (H,W) in [0,1] for imshow
        """
        d = d_np.copy()
        # inf/nan 제거
        d[~np.isfinite(d)] = np.nan
        # clip to [near, far]
        d = np.clip(d, near, far)
        # normalize -> [0,1]
        vis = (d - near) / (far - near + 1e-12)
        if invert:
            vis = 1.0 - vis
        # nan -> 0
        vis = np.nan_to_num(vis, nan=0.0)
        return vis.astype(np.float32)

    # ----------------------------
    # public api
    # ----------------------------
    def set_env(self, env_id: int):
        self.env_id = int(env_id)
        self.ax.set_title(f"{self.title_base} (env{self.env_id})")

    def set_depth_display(
        self, *, auto_scale=False, vmin=None, vmax=None, percentile=(2.0, 98.0),
        near=0.1, far=4.0, invert=True
    ):
        """
        Only meaningful for mode="depth".
        - auto_scale=True: per-frame percentile scaling (raw meters 기준)
        - auto_scale=False: fixed vmin/vmax (raw meters 기준)
        - near/far/invert: 표시용 정규화 범위(문서 스타일)
        """
        self.depth_auto_scale = bool(auto_scale)
        self.depth_vmin = vmin
        self.depth_vmax = vmax
        self.depth_percentile = (float(percentile[0]), float(percentile[1]))

        self.depth_near = float(near)
        self.depth_far = float(far)
        self.depth_invert = bool(invert)

    def update(self, camera_tensor, pause_s=0.001):
        x = camera_tensor[self.env_id]

        if self.mode == "rgb":
            rgb_np = self._to_hwc_rgb_uint8(x)
            self.im.set_data(rgb_np)

        else:
            d_np = self._to_hw_depth_float32(x)

            # (옵션) raw meter에서 auto-scale로 near/far 갱신
            near = self.depth_near
            far = self.depth_far
            if self.depth_auto_scale:
                finite = np.isfinite(d_np)
                if finite.any():
                    p_lo, p_hi = self.depth_percentile
                    near = float(np.percentile(d_np[finite], p_lo))
                    far = float(np.percentile(d_np[finite], p_hi))
                    if far <= near:
                        far = near + 1e-3

            # ✅ 표시용 [0,1] 변환
            vis = self._depth_to_vis01(d_np, near=near, far=far, invert=self.depth_invert)
            self.im.set_data(vis)
            self.im.set_clim(0.0, 1.0)  # 항상 고정

        self.fig.canvas.draw_idle()
        plt.pause(pause_s)


class CbfLineViz:
    """
    Real-time rolling plot for Scalar values (True vs Pred).
    """
    def __init__(self, title="CBF Tracking", env_id=0, window_len=100, y_range=None):
        self.env_id = int(env_id)
        self.window_len = window_len
        self.y_range = y_range

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 3)) # 크기는 조절 가능
        self.ax.set_title(f"{title} (env {self.env_id})")
        
        # x축은 고정, y축 데이터 버퍼 초기화
        self.x_data = np.arange(window_len)
        self.y_true = np.zeros(window_len)
        self.y_pred = np.zeros(window_len)

        # 라인 생성
        self.line_true, = self.ax.plot(self.x_data, self.y_true, label='True(Env)', color='blue', alpha=0.7)
        self.line_pred, = self.ax.plot(self.x_data, self.y_pred, label='Pred(Model)', color='red', linestyle='--', alpha=0.9)
        
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_xlim(0, window_len - 1)
        
        # Y축 범위 고정 (CBF 값 범위에 맞춰 설정하면 보기가 편함)
        if y_range:
            self.ax.set_ylim(y_range)
        else:
            self.ax.set_ylim(-0.5, 5.0) # 기본값 (상황에 맞게 조절)

        self.fig.tight_layout()
        self.fig.show()

    def update(self, true_val, pred_val):
        # 데이터 시프트 (오래된 데이터 삭제)
        self.y_true[:-1] = self.y_true[1:]
        self.y_true[-1] = float(true_val)
        
        self.y_pred[:-1] = self.y_pred[1:]
        self.y_pred[-1] = float(pred_val)

        # 그래프 업데이트
        self.line_true.set_ydata(self.y_true)
        self.line_pred.set_ydata(self.y_pred)

        # (옵션) Y축 오토 스케일링이 필요하면 주석 해제
        # if self.y_range is None:
        #     vals = np.concatenate([self.y_true, self.y_pred])
        #     mn, mx = vals.min(), vals.max()
        #     margin = (mx - mn) * 0.1 if mx > mn else 0.5
        #     self.ax.set_ylim(mn - margin, mx + margin)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()