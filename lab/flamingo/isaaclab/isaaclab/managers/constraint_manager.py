# constraint_manager.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint manager for computing constraint violation signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from collections import defaultdict
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from .constraint_term_cfg import ConstraintTermCfg

if TYPE_CHECKING:
    from lab.flamingo.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv


class ConstraintManager(ManagerBase):
    """Manager for computing continuous constraint violation signals for a given world.

    Each constraint term is a function that takes the environment as an argument and returns
    a float tensor of shape (num_envs,) in [0, 1] representing the degree of violation.
    The overall constraint signal is computed as the element-wise maximum over the individual
    term signals, with terms flagged as timeouts (time_out=True) stored in one buffer and
    the remaining terms in another.
    """

    _env: ManagerBasedConstraintRLEnv

    def __init__(self, cfg: object, env: ManagerBasedConstraintRLEnv):
        """Initializes the constraint manager.

        Args:
            cfg: The configuration object or dictionary for constraint terms,
                 where each term should be an instance of ConstraintTermCfg.
            env: An environment object.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[ConstraintTermCfg] = []
        self._class_term_cfgs: list[ConstraintTermCfg] = []

        super().__init__(cfg, env)  # _prepare_terms() called here

        self.env = env

        # hyperparameters
        self.tau = 0.95  # 지수 이동 평균 계수
        self.min_p = 0.0  # 최소 종료 확률
        # TODO should be set by the user argument (agent_cfg)
        self.num_transitions_per_env = 24
        self.max_iterations = 5000
        self.step_cur = 1.0 / (self.num_transitions_per_env * self.max_iterations)
        self.static_curriculum_steps = 30000

        # prepare a dictionary to store each term's value (float tensor)
        self._term_values = {}
        self.curriculum = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            self._term_values[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

            if term_cfg.time_out not in ["truncate", "terminate", "constraint"]:
                raise ValueError(f"Invalid time_out value '{term_cfg.time_out}' for term '{term_cfg.func}'.")

            # Constraint curriculum 초기화
            if term_cfg.time_out == "constraint" and term_cfg.use_curriculum:
                self.curriculum[name] = 0.0

        # 각 제약 조건에 대한 확률적 종료 관련 변수
        self._running_maxes = {}  # 지수 이동 평균 최대값
        self._probs = {}  # 확률적 종료 확률

        # create buffers for managing constraint signals as float tensors
        self._truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._delta_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)


    def __str__(self) -> str:
        """Returns a string representation for the constraint manager."""
        msg = f"<ConstraintManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Constraint Terms"
        table.field_names = ["Index", "Name", "p_max"]
        table.align["Name"] = "l"
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, getattr(term_cfg, "p_max", 1.0)])
        msg += table.get_string() + "\n"
        return msg

    @property
    def active_terms(self) -> list[str]:
        """Name of active constraint terms."""
        return self._term_names
    
    @property
    def dones(self) -> dict[str, torch.Tensor]:  #* not using  yet
        """Returns the episodic sums for each constraint term."""
        return torch.max(self._truncated_buf, self._delta_buf)

    @property
    def get_termination_probs(self) -> torch.Tensor:  #* not using  yet
        """Returns the final termination probability across all constraints."""
        if len(self._probs) == 0:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        probs = torch.cat(list(self._probs.values()), dim=1)
        return probs.max(dim=1).values
    
    @property
    def time_outs(self) -> torch.Tensor:
        """Returns the timeout signal computed from timeout constraint terms."""
        return self._truncated_buf

    @property
    def constrained(self) -> torch.Tensor:
        """Returns the soft constraint signal (delta) computed from non-timeout terms."""
        return self._delta_buf
    
    @property
    def hard_constrained(self) -> torch.Tensor:
        """Returns the hard constraint signal (delta) computed from timeout terms."""
        return self._delta_buf == 1.0
    

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the constraint term values for a new episode and returns summary information.

        Args:
            env_ids: The environment ids to reset (if None, reset all environments).

        Returns:
            A dictionary containing the episodic sums for each constraint term.
        """
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for key, term in zip(self._term_values.keys(), self._term_cfgs):
            if term.time_out == "truncate":
                extras["Episode_Constraint/" + key] = torch.count_nonzero(self._term_values[key][env_ids].float()).item()
            elif term.time_out == "terminate":
                extras["Episode_Constraint/" + key] = torch.count_nonzero(self._term_values[key][env_ids].float()).item()
            else:
                extras["Episode_Constraint/" + key] = torch.mean(self._term_values[key][env_ids]).item()

        # Reset probs and delta_buf only for the specified env_ids
        for key in self._probs.keys():
            self._probs[key][env_ids] = 0.0  # 환경별 초기화

        # reset all the constraints terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self) -> torch.Tensor:
        """Computes the stochastic termination probabilities based on constraint violations.

        Returns:
            A tensor of shape (num_envs,) with termination probabilities in [0, 1].
        """
        self._truncated_buf.zero_()
        self._delta_buf.zero_()


        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params).float()

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)

            if term_cfg.time_out == "truncate":
                value = torch.clamp(value, 0.0, 1.0)
                if not torch.all((value == 0.0) | (value == 1.0)):
                    raise ValueError("value must be either 0 or 1.")
                self._truncated_buf = torch.max(self._truncated_buf, value)
                self._term_values[name][:] = value
            elif term_cfg.time_out == "terminate":
                value = torch.clamp(value, 0.0, 1.0)
                if not torch.all((value == 0.0) | (value == 1.0)):
                    raise ValueError("value must be either 0 or 1.")
                self._delta_buf = torch.max(self._delta_buf, value)
                self._term_values[name][:] = value
            elif term_cfg.time_out == "constraint":
                p_max = getattr(term_cfg, "p_max")
                if term_cfg.use_curriculum:
                    if self.env.common_step_counter < self.static_curriculum_steps:
                        p_max = 0.0
                    else:
                        self.curriculum[name] = min(self.curriculum[name] + self.step_cur, 1.0)
                        T_start = 20
                        T_end = max(1.0 / p_max, 1e-6)
                        p_max = 1.0 / (T_start + self.curriculum[name] * (T_end - T_start))

                constraint_max = value.max(dim=0, keepdim=True)[0].clamp(min=1e-6) # max(s,a E batch) c_i^max

                if name not in self._running_maxes:
                    self._running_maxes[name] = constraint_max.clone()
                else:
                    self._running_maxes[name] = (
                        self.tau * self._running_maxes[name] + (1.0 - self.tau) * constraint_max
                    )

                mask = value > 0.0
                probs = torch.zeros_like(value, dtype=torch.float32)  # float32 명시적 지정
                probs[mask] = (self.min_p + torch.clamp(
                    value[mask] / self._running_maxes[name].expand(value.size())[mask],
                    min=0.0, max=1.0
                ) * (p_max - self.min_p)).to(probs.dtype)  # float32 변환 추가

                self._probs[name] = probs
                self._delta_buf = torch.max(self._delta_buf, probs)

                self._term_values[name][:] = value

        reset_buf = torch.max(self._truncated_buf, self._delta_buf)

        return reset_buf == 1.0


    def get_term(self, name: str) -> torch.Tensor:
        """Returns the constraint term value for the specified name.

        Args:
            name: The name of the constraint term.

        Returns:
            A float tensor of shape (num_envs,) representing the term's value.
        """
        return self._term_values[name]

    def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
        """Sets the configuration for the specified constraint term.

        Args:
            term_name: The name of the constraint term.
            cfg: The new configuration.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        """Gets the configuration for the specified constraint term.

        Args:
            term_name: The name of the constraint term.

        Returns:
            The configuration for the constraint term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active constraint terms as an iterable sequence of tuples.

        Each tuple contains the term name and a list with the term's raw value for the specified environment index.

        Args:
            env_idx: The index of the environment to extract term values from.

        Returns:
            A sequence of tuples in the form (term_name, [value]).
        """
        terms = []
        for key in self._term_names:
            terms.append((key, [self._term_values[key][env_idx].float().cpu().item()]))
        return terms

    def _prepare_terms(self):
        """Parses the configuration and prepares the constraint terms."""
        # 먼저, cfg가 dict인지 아닌지에 따라 항목을 가져옵니다.
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg. "
                    f"Received: '{type(term_cfg)}'."
                )
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_constraint_cfgs.append(term_cfg)