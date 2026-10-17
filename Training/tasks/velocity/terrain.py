"""Rough terrain in robot-size bins.

The terrain is one static model shared by every env, so it can't be resized
per env. Instead the grid holds the usual sub-terrains once per size bin, with
heights (stairs, noise, waves) and step widths scaled to that bin's leg length,
and `assign_terrain_bins` sends each robot only to the columns of its own bin.
Column index = bin * len(TERRAIN_TYPES) + type.
"""

from __future__ import annotations

__all__ = ["NUM_BINS", "NUM_TYPES", "scaled_rough_terrain_cfg", "assign_terrain_bins"]

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from tasks.velocity.mdp.scaling import get_morph_scales

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Leg-length scale (vs Go2) covered by the bins: ~p1-p99 of generated robots.
MIN_SCALE, MAX_SCALE = 0.2, 4.5
NUM_BINS = 4
NUM_ROWS = 8
NUM_TYPES = len(ROUGH_TERRAINS_CFG.sub_terrains)
PATCH_SIZE = 10.0  # Large robots need room for a start platform plus steps.
_MAX_PLATFORM = 4.5
_MIN_STEP_WIDTH = 0.2  # Keeps the stair box count (and per-env model memory) down.
_MAX_STEP_WIDTH = 0.75

_EDGES = MIN_SCALE * (MAX_SCALE / MIN_SCALE) ** (np.arange(NUM_BINS + 1) / NUM_BINS)
BIN_SCALES = np.sqrt(_EDGES[:-1] * _EDGES[1:])  # Geometric bin centres.


def _scaled(cfg, s: float):
  """One sub-terrain with its lengths in metres scaled by the bin's s_L."""
  changes = {}
  for field in ("step_height_range", "noise_range", "amplitude_range"):
    if hasattr(cfg, field):
      lo, hi = getattr(cfg, field)
      changes[field] = (lo * s, hi * s)
  if hasattr(cfg, "noise_step"):
    changes["noise_step"] = max(cfg.noise_step * s, cfg.vertical_scale)
  if hasattr(cfg, "step_width"):
    changes["step_width"] = float(np.clip(cfg.step_width * s, _MIN_STEP_WIDTH, _MAX_STEP_WIDTH))
  if hasattr(cfg, "platform_width"):
    changes["platform_width"] = min(cfg.platform_width * max(s, 0.5), _MAX_PLATFORM)
  # Hf terrains need borders of at least one pixel (0.1 m).
  if hasattr(cfg, "border_width") and cfg.border_width > 0:
    changes["border_width"] = max(cfg.border_width * max(s, 1.0), 0.15)
  return replace(cfg, **changes)


def scaled_rough_terrain_cfg():
  """ROUGH_TERRAINS_CFG with a copy of each sub-terrain per size bin."""
  sub_terrains = {
    f"{name}_s{b}": replace(_scaled(cfg, float(BIN_SCALES[b])), proportion=1.0)
    for b in range(NUM_BINS)
    for name, cfg in ROUGH_TERRAINS_CFG.sub_terrains.items()
  }
  return replace(ROUGH_TERRAINS_CFG, size=(PATCH_SIZE, PATCH_SIZE),
                 num_rows=NUM_ROWS, num_cols=NUM_BINS * NUM_TYPES, sub_terrains=sub_terrains)


def assign_terrain_bins(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  randomize: bool = False,
) -> None:
  """Put each env on the terrain columns of its robot's size bin.

  Startup event, after `morphology`. The terrain types within a bin are shared
  out evenly among its envs. With `randomize` (reset mode, for play) each reset
  picks a random row and type within the bin instead.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  length = get_morph_scales(env).length[env_ids]
  edges = torch.as_tensor(_EDGES[1:-1], dtype=length.dtype, device=env.device)
  size_bin = torch.bucketize(length, edges)

  if randomize:
    kind = torch.randint(0, NUM_TYPES, (len(env_ids),), device=env.device)
    terrain.terrain_levels[env_ids] = torch.randint(
      0, terrain.max_terrain_level, (len(env_ids),), device=env.device)
  else:
    kind = torch.zeros_like(size_bin)
    for b in range(NUM_BINS):
      members = (size_bin == b).nonzero().flatten()
      kind[members] = torch.arange(len(members), device=env.device) % NUM_TYPES
  terrain.terrain_types[env_ids] = size_bin * NUM_TYPES + kind
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
