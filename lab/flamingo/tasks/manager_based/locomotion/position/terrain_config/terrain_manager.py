# import torch
# import numpy as np

# import torch.nn.functional as F
# from abc import ABC
# from genesis.utils import *


# # ----------------------------- Terrain base class ----------------------------

# class Terrain(ABC):
#     def __init__(
#                 self,
#                 map_x_size: float,
#                 map_y_size: float,
#                 elevation_map_size: tuple, # (emap_x [m], emap_y [m])
#                 elevation_map_resolution: tuple, # (x_res, y_res)
#                 device: torch.device | str = "gpu",):

#         self._terrain_resolution = 1024
#         self._map_x_size = map_x_size
#         self._map_y_size = map_y_size
#         self._ems = elevation_map_size
#         self._emr = elevation_map_resolution
#         self.device = torch.device(device)
# # ------------------------------ Pyramid Stairs -------------------------------
# class PyramidStairs(Terrain):
#     def __init__(
#                 self,
#                 map_x_size: float,
#                 map_y_size: float,
#                 elevation_map_resolution: tuple,
#                 elevation_map_size: tuple,
#                 step_width: float,
#                 step_height: float,
#                 num_steps: int,
#                 ground_r: float,
#                 scale: float,
#                 device: torch.device | str = "gpu",):
#         super().__init__(map_x_size,
#                          map_y_size,
#                          elevation_map_size,
#                          elevation_map_resolution,
#                          device,)
        
#         self.map_x_size = map_x_size
#         self.map_y_size = map_y_size
#         self.step_width = step_width
#         self.step_height = step_height
#         self.num_steps = num_steps
#         self.ground_r = ground_r
#         self.scale = scale

#         self._terrain_map = self._generate_terrain_map()
#         self._height_field = self._generate_height_field()
        
#     def _generate_height_field(self) -> np.ndarray:
#         H = int(self.map_y_size * self.scale)
#         W = int(self.map_x_size * self.scale)
#         x = (np.arange(W) - W / 2.0)
#         y = (np.arange(H) - H / 2.0)
#         X, Y = np.meshgrid(x, y)
#         dist = np.maximum(np.abs(X), np.abs(Y))
#         # 계산된 거리에서 ground_r 를 뺀 값을 기준으로 계단 높이 결정
#         dist_from_ground = dist - self.ground_r * self.scale
#         level = np.ceil(dist_from_ground / (self.step_width * self.scale))
#         level = np.clip(level, 0, self.num_steps)
#         terrain_map = level * self.step_height * self.scale
#         height_field = terrain_map
#         return height_field.astype(np.float32)
#     def _generate_terrain_map(self) -> torch.Tensor:
#         """Create (H,W) = (1024,1024) elevation map on GPU (float32)."""
#         res = self._terrain_resolution
#         dx = self._map_x_size / res
#         dy = self._map_y_size / res
#         center = res / 2.0
#         # 두 축 모두에 대해 for‑loop 대신 vector-wise 계산
#         xi = torch.arange(res, dtype=torch.float32).view(-1, 1)
#         yj = torch.arange(res, dtype=torch.float32).view(1, -1)
#         x = (xi - center) * dx
#         y = (yj - center) * dy
#         dist = torch.maximum(x.abs(), y.abs())
#         dist_from_ground = dist - self.ground_r
#         level = torch.clamp(
#         torch.ceil(dist_from_ground / self.step_width), min=0, max=self.num_steps
#         )
#         terrain_map = (level * self.step_height).float()
#         terrain_map = terrain_map.to(self.device).unsqueeze(0).unsqueeze(0) # batch: (1,1,H,W)
#         return terrain_map

# class TerrainManager:
#     def __init__(self, scene, terrain, robot_base_pos, robot_base_quat):
#         self._scene = scene

#         self._terrain = terrain
#         self.robot_base_pos = robot_base_pos # (N,3)
#         self.robot_base_quat = robot_base_quat # (N,4)
    
#     def generate_terrain(self, origin, friction=1.0):
#         pos_x = -(self._terrain._map_x_size / 2) + origin[0]
#         pos_y = -(self._terrain._map_y_size / 2) + origin[1]
#         pos_z = origin[2]
#         terrain = self._scene.add_entity(morph=gs.morphs.Terrain(
#             pos=(pos_x, pos_y, pos_z),
#             horizontal_scale=1 / self._terrain.scale,
#             vertical_scale=1 / self._terrain.scale,
#             height_field=self._terrain._height_field,
#             ))
#         terrain.set_friction(friction)

#     @torch.no_grad()
#     def get_elevation_map(self) -> torch.Tensor: # (N, x_res, y_res)
#     # ① 위치 고도
#         pos_xy = self.robot_base_pos[:, :2]   # (N,2) → (x,y)
#         pos_z = self.robot_base_pos[:, 2]     # (N,)
    
#     # ② 쿼터니안 → yaw(heading) 각도만 추출
#         w, x, y, z = self.robot_base_quat[:, 0], self.robot_base_quat[:, 1], \
#                      self.robot_base_quat[:, 2], self.robot_base_quat[:, 3]
#         siny_cosp = 2 * (w * z + x * y)
#         cosy_cosp = 1 - 2 * (y * y + z * z)
#         yaw = torch.atan2(siny_cosp, cosy_cosp) # (N,)
    
#     # ③ 로컬 그리드 생성 (ROS 좌표계)

#         N = pos_xy.size(0)
#         x_res, y_res = self._terrain._emr
#         emap_x, emap_y = self._terrain._ems
#         res = float(self._terrain._terrain_resolution)
#         dx, dy = self._terrain._map_x_size / res, self._terrain._map_y_size / res
#         center = (res - 1.0) / 2.0
#         xs = torch.linspace(-emap_x / 2, emap_x / 2, x_res, device=self._terrain.device) # 전진 방향
#         ys = torch.linspace(-emap_y / 2, emap_y / 2, y_res, device=self._terrain.device) # 왼쪽 방향
    
#         xx, yy = torch.meshgrid(xs, ys, indexing="ij") # (x_res, y_res)
#         local_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(N, -1, -1, -1) # (N,x_res,y_res,2)
        
#         # ④ 로컬 월드 좌표 변환
#         cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
#         rot = torch.stack([
#               torch.stack([cos_yaw, -sin_yaw], dim=1),
#               torch.stack([sin_yaw, cos_yaw], dim=1)], dim=-1) # (N,2,2)
#         rotated = torch.einsum("nhwc,ncd->nhwd", local_grid, rot) # (N,x_res,y_res,2)
#         world = rotated + pos_xy[:, None, None, :]

#         idx_x = world[..., 0] / dx + center
#         idx_y = world[..., 1] / dy + center
#         grid = torch.stack([(idx_x / (res - 1)) * 2 - 1, # x
#                             (idx_y / (res - 1)) * 2 - 1 # y
#                             ], dim=-1)

#         terrain_batch = self._terrain._terrain_map.expand(N, -1, -1, -1) # (N,1,H,W)
#         sampled = F.grid_sample(terrain_batch, grid,
#                                 mode="bilinear", padding_mode="border",
#                                 align_corners=False).squeeze(1) # (N,x_res,y_res)
#         sampled = sampled - pos_z.view(-1, 1, 1)

#         return sampled.contiguous() # (N, x_res, y_res)