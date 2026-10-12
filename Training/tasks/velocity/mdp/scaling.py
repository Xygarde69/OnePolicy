"""Per-robot scale factors relative to Go2.

The task's thresholds are Go2's. Each generated robot gets them rescaled by
dynamic similarity (Froude scaling) to its own anatomy, with
  s_L = leg length / Go2 leg length,   s_m = mass / Go2 mass,
  speed ~ sqrt(s_L),  time ~ sqrt(s_L),  angular rate ~ 1/sqrt(s_L),
  force ~ s_m,  torque ~ s_m * s_L.
A robot that is Go2's size gets exactly Go2's values. Angles and ratios
(orientation limits, joint pose stds, gait duty factor, ...) need no scaling.

The `morphology` startup event fills `env.morph_scales`; the reward,
termination, command and event terms read it.
"""

from __future__ import annotations

__all__ = ["MorphScales", "get_morph_scales", "normalized_command"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from assets.morph_quadruped.constants import NOMINAL_FOOT_RADIUS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class MorphScales:
  """Per-env scale factors, each of shape [num_envs]."""

  length: torch.Tensor
  """Leg length / Go2's."""
  mass: torch.Tensor
  """Total mass / Go2's."""
  foot_radius: torch.Tensor
  """Foot radius [m] (absolute)."""

  @property
  def speed(self) -> torch.Tensor:
    """Froude speed scale, sqrt(s_L)."""
    return torch.sqrt(self.length)

  @property
  def time(self) -> torch.Tensor:
    """Froude time scale, sqrt(s_L)."""
    return torch.sqrt(self.length)


def get_morph_scales(env: ManagerBasedRlEnv) -> MorphScales:
  """The env's scales, created at Go2's (all 1) on first use.

  Managers evaluate their terms once while being built, before the startup
  events run; the `morphology` event then fills in each robot's values.
  """
  scales = getattr(env, "morph_scales", None)
  if scales is None:
    ones = torch.ones(env.num_envs, device=env.device)
    scales = MorphScales(length=ones.clone(), mass=ones.clone(),
                         foot_radius=ones * NOMINAL_FOOT_RADIUS)
    env.morph_scales = scales
  return scales


def normalized_command(env: ManagerBasedRlEnv, command: torch.Tensor) -> torch.Tensor:
  """Velocity command in Go2 units: linear part divided by the speed scale,
  yaw rate multiplied by the time scale.

  Compare this, not the raw command, against Go2-tuned speed thresholds.
  """
  scales = get_morph_scales(env)
  return torch.cat([command[:, :2] / scales.speed[:, None],
                    command[:, 2:] * scales.time[:, None]], dim=1)
