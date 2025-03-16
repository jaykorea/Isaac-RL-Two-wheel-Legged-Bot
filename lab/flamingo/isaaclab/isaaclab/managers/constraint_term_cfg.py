
from __future__ import annotations


from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import Callable
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    """Configuration for a constraint term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the constraint signals as torch boolean tensors of
    shape (num_envs,).
    """

    p_max: float = 1.0
    """The maximum scaling factor for the termination probability for this constraint.

    For hard constraints, set p_max to 1.0 to strictly enforce the constraint.
    For soft constraints, use a value lower than 1.0 (e.g., 0.25) to allow some exploration,
    and optionally schedule p_max to increase over training.
    """

    use_curriculum: bool = False
    """Whether to use a soft probability curriculum for this constraint.
    """

    time_out: str = "terminate"
    """Whether the constraint term contributes towards episodic timeouts. Defaults to False.

    Note:
        These usually correspond to tasks that have a fixed time limit.
    """