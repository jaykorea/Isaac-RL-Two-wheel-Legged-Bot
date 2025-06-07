from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCasterCfg
from .lift_mask import LiftMask


GREEN_MARKER_CFG = VisualizationMarkersCfg(
    prim_path = "/Visuals/RayCaster",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

RED_MARKER_CFG = VisualizationMarkersCfg(
    prim_path = "/Visuals/RayCaster",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

@configclass
class LiftMaskCfg(RayCasterCfg):
    class_type: type = LiftMask
    green_visualizer_cfg: VisualizationMarkersCfg = GREEN_MARKER_CFG
    red_visualizer_cfg: VisualizationMarkersCfg = RED_MARKER_CFG
    gradient_threshold: float = 0.05
    last_zero_num: int = 3