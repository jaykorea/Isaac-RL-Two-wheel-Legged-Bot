from isaaclab.sensors.ray_caster import RayCaster
from collections.abc import Sequence
import torch
from dataclasses import dataclass
from isaaclab.markers import VisualizationMarkers

@dataclass
class LiftMaskData:
    """Data container for the ray-cast sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame.
    Shape is (N, B, 3), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """

    mask: torch.Tensor = None
    """The mask for the lift sensor.
    Shape is (N, T, 1), where N is the number of sensors, 1 is the mask value.
    in the scan pattern per sensor.
    """

    mask_history: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, T, 1), where N is the number of sensors and T is the configured history length

    In the history dimension, the first index is the most recent and the last index is the oldest.
    """
    
class LiftMask(RayCaster):  
    def __init__(self, *args, **kwargs):
        from .lift_mask_cfg import LiftMaskCfg  # Lazy import to avoid circular dependency

        self.cfg = LiftMaskCfg
        super().__init__(*args, **kwargs)
        self._data = LiftMaskData()
        self._height_map_w = int(round(self.cfg.pattern_cfg.size[0] / self.cfg.pattern_cfg.resolution) + 1)  
        self._height_map_h = int(round(self.cfg.pattern_cfg.size[1] / self.cfg.pattern_cfg.resolution) + 1) 
        self._last_zero_index = round((self._height_map_h-self.cfg.last_zero_num)/2)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Lift-mask @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\tstep height threshold (m)     : {self.cfg.gradient_threshold}\n"
            f"\tlast zero num     : {self.cfg.last_zero_num}"
        )
    
    @property
    def data(self) -> LiftMaskData:
        # update sensors if needed
        self._update_outdated_buffers()  # Use the stored env object
        # return the data
        return self._data
    
    def reset(self, env_ids: Sequence[int] | None = None):
        # 상위 클래스(RayCaster)의 reset 호출 (타이머나 기타 내부 변수 초기화)
        super().reset(env_ids)
        # env_ids가 None인 경우 전체 환경을 대상으로 함
        if env_ids is None:
            env_ids = slice(None)

        # 현재 mask 값을 0으로 초기화
        if self._data.mask is not None:
            self._data.mask[env_ids] = 0.0

        # history 기능이 활성화되어 있다면 mask_history 버퍼를 0으로 초기화
        if self.cfg.history_length > 0 and self._data.mask_history is not None:
            self._data.mask_history[env_ids] = 0.0

    def _initialize_impl(self):
        # 상위 클래스의 초기화를 통해 RayCaster 관련 모든 기본 초기화 수행
        super()._initialize_impl()

        self._graidient_mask = torch.ones((self._height_map_h, self._height_map_w-1), dtype=torch.float32, device=self._device)
        for i in range(self._last_zero_index):
            self._graidient_mask[:i+1, (self._height_map_w-1) - self._last_zero_index+i]  = 0.0
            self._graidient_mask[self._height_map_h-(i+1):, (self._height_map_w-1) - self._last_zero_index+i]  = 0.0

        # lift_mask에서 추가로 초기화할 부분:
        # 현재 센서 수는 self._view.count (혹은 RayCaster에서 설정한 값)을 이용합니다.
        self._data.mask = torch.zeros(self._view.count, 1, device=self._device)
        if self.cfg.history_length > 0:
            self._data.mask_history = torch.zeros(self._view.count, self.cfg.history_length, 1, device=self._device)
            

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Call parent class
        super()._update_buffers_impl(env_ids)

        # additional data update
        self._update_lift_mask()
        
        # === History 업데이트 코드 추가 ===
        if self.cfg.history_length > 0:
            # shift history buffer: 이전 프레임의 값들을 한 칸씩 뒤로 밀어냅니다.
            self._data.mask_history[env_ids, 1:] = self._data.mask_history[env_ids, :-1].clone()
            # 최신 mask 값을 history의 0번 인덱스에 저장합니다.
            # _update_lift_mask()에서 계산된 self._data.mask는 보통 (N,) 형태일 가능성이 있으므로
            # unsqueeze를 통해 (N, 1)로 맞춥니다.
            if self._data.mask.dim() == 1:
                current_mask = self._data.mask.unsqueeze(-1)
            else:
                current_mask = self._data.mask
            self._data.mask_history[env_ids, 0] = current_mask[env_ids]
        # =================================

    def _update_lift_mask(self):
        heights = self._data.ray_hits_w[..., 2]

        grid = heights.reshape(-1, self._height_map_h, self._height_map_w)
        gradients = (grid[:, :, 1:] - grid[:, :, :-1]) * self._graidient_mask
        
        row_max = torch.max(gradients, dim=2).values
        column_max = (torch.max(row_max, dim=1).values > self.cfg.gradient_threshold).float()
        self._data.mask = column_max 

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer_green") or not hasattr(self, "ray_visualizer_red"):
                self.ray_visualizer_green = VisualizationMarkers(self.cfg.green_visualizer_cfg)
                self.ray_visualizer_red = VisualizationMarkers(self.cfg.red_visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer_green.set_visibility(True)
            self.ray_visualizer_red.set_visibility(True)
        else:
            # 올바른 속성 이름을 사용해서 마커의 visibility를 끕니다.
            if hasattr(self, "ray_visualizer_green"):
                self.ray_visualizer_green.set_visibility(False)
            if hasattr(self, "ray_visualizer_red"):
                self.ray_visualizer_red.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Ensure mask and ray hits are valid
        if self._data.mask is None:
            self._data.mask = torch.zeros((self._data.ray_hits_w.shape[0], 1), dtype=torch.float32)

        # Get indices where mask is 0 and 1
        indices_0 = (self._data.mask == 0).nonzero(as_tuple=True)[0]
        indices_1 = (self._data.mask == 1).nonzero(as_tuple=True)[0]
        # Ensure ray hits have the correct dimensions

        # Visualize ray hits with green for mask == 0 and red for mask == 1
        if indices_0.numel() > 0:  # Ensure there are points to visualize
            self.ray_visualizer_green.visualize(self._data.ray_hits_w[indices_0, :].view(-1, 3))

        if indices_1.numel() > 0:  # Ensure there are points to visualize
            self.ray_visualizer_red.visualize(self._data.ray_hits_w[indices_1, :].view(-1, 3))