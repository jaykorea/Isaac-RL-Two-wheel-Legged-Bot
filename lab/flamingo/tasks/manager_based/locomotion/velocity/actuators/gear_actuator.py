from __future__ import annotations
from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils.types import ArticulationActions

class GearDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with an additional gear ratio and gamma scaling on applied torques."""

    cfg: "GearDelayedPDActuatorCfg"

    def compute(
        self, control_action: ArticulationActions, joint_pos, joint_vel
    ) -> ArticulationActions:
        joint_pos = joint_pos * self.cfg.gear_ratio # OBS_{joint space} -> OBS_{motor space}
        joint_vel = joint_vel * self.cfg.gear_ratio # OBS_{joint space} -> OBS_{motor space}
        # 먼저 DelayedPDActuator의 compute를 호출해서 delayed torque 계산
        control_action = super().compute(control_action, joint_pos, joint_vel) # motor space에서 torque 계산

        # 여기서 gear ratio를 곱해줌
        if self.cfg.gear_ratio != 1.0:
            self.computed_effort = self.computed_effort * self.cfg.gear_ratio * self.cfg.gamma # TORQUE_{motor space} -> TORQUE_{joint space}
            self.applied_effort = self._clip_effort(self.computed_effort)
            control_action.joint_efforts = self.applied_effort

        return control_action