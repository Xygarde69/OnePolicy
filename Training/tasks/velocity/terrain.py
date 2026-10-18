"""Rough terrain scaled per env to the robot's size.

The terrain is mjlab's ROUGH_TERRAINS_CFG grid, built once at Go2 scale (one
column per terrain type, rows easy to hard) and shared by every env. Each env
then sees the whole grid scaled by its robot's leg length, snapped to one of
NUM_SCALES log-spaced values: every tile in that world is the robot's size, so
walking onto a neighbouring tile is harmless.

- Box geoms (stairs, flat, border) are scaled per world: geom_size, rbound,
  aabb and position (about the grid centre) times s.
- Heightfields can't be: mujoco_warp's hfield size is one per hfield, not per
  world. Each hfield tile is compiled NUM_SCALES times with the same heights
  and its size baked at each scale (`_ScaledCopies`); per world only the copy
  of its scale stays in place, the others are parked far below.

Terrain geoms are static, so mujoco_warp computes their world positions once
in make_data; `scale_terrain` writes data.geom_xpos as well as geom_pos.
"""

from __future__ import annotations

__all__ = [
  "NUM_TYPES", "TERRAIN_SCALES", "scaled_rough_terrain_cfg", "scale_terrain",
  "scale_env_origins",
]

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.terrains import heightfield_terrains
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput

from tasks.velocity.mdp.scaling import get_morph_scales

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Leg-length scale (vs Go2) covered: generated robots are ~0.17-4.9x (p1-p99);
# robots outside get the end scale. Neighbouring scales are 16% apart, so a
# robot's terrain is within 8% of its own size.
MIN_SCALE, MAX_SCALE = 0.15, 5.0
NUM_SCALES = 24
TERRAIN_SCALES = MIN_SCALE * (MAX_SCALE / MIN_SCALE) ** (np.arange(NUM_SCALES) / (NUM_SCALES - 1))
NUM_ROWS = 10
NUM_TYPES = len(ROUGH_TERRAINS_CFG.sub_terrains)
_PARK_DEPTH = 1000.0  # Inactive hfield copies sit this far below the grid.
# The compiled (unscaled) model keeps the copy nearest Go2's size in place.
_DEFAULT_COPY = int(np.argmin(np.abs(np.log(TERRAIN_SCALES))))


def _copy_name(name: str, k: int) -> str:
  return f"{name}_k{k}"


def _copy_index(name: str) -> int:
  return int(name.rsplit("_k", 1)[1])


@dataclass(kw_only=True)
class _ScaledCopies(SubTerrainCfg):
  """A heightfield sub-terrain compiled once per TERRAIN_SCALES entry.

  The copies share the heights; copy k's hfield size (x, y, top, bottom) and
  its geom's position within the tile are the Go2-scale ones times
  TERRAIN_SCALES[k] once `scale_terrain` scales positions per world, so only
  the hfield size is baked here. All but the copy nearest Go2's size start
  parked, so the unscaled model is the plain Go2-scale grid.
  """

  inner: SubTerrainCfg

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    output = replace(self.inner, size=self.size).function(difficulty, spec, rng)
    (base,) = output.geometries
    assert base.geom is not None and base.hfield is not None
    size, pos = np.array(base.hfield.size), np.array(base.geom.pos)
    name = base.hfield.name
    geometries = []
    for k, s in enumerate(TERRAIN_SCALES):
      if k == 0:
        field, geom = base.hfield, base.geom
        field.name = _copy_name(name, 0)
        field.size = size * s
        geom.hfieldname = field.name
      else:
        field = spec.add_hfield(name=_copy_name(name, k), size=size * s, nrow=base.hfield.nrow,
                                ncol=base.hfield.ncol, userdata=base.hfield.userdata)
        geom = spec.body("terrain").add_geom(
          type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
          pos=pos, material=base.geom.material)
      if k != _DEFAULT_COPY:
        geom.pos[2] -= _PARK_DEPTH
      geometries.append(replace(base, geom=geom, hfield=field))
    return replace(output, geometries=geometries)


def scaled_rough_terrain_cfg():
  """ROUGH_TERRAINS_CFG at Go2 scale, one column per terrain type."""
  sub_terrains = {}
  for name, cfg in ROUGH_TERRAINS_CFG.sub_terrains.items():
    cfg = replace(cfg, proportion=1.0)
    if type(cfg).__module__ == heightfield_terrains.__name__:
      cfg = _ScaledCopies(inner=cfg, proportion=1.0)
    sub_terrains[name] = cfg
  return replace(ROUGH_TERRAINS_CFG, num_rows=NUM_ROWS, num_cols=NUM_TYPES,
                 sub_terrains=sub_terrains)


def _terrain_scale_index(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
  """Index into TERRAIN_SCALES nearest (in log) to each robot's leg-length scale."""
  log_s = torch.log(get_morph_scales(env).length[env_ids])
  step = np.log(MAX_SCALE / MIN_SCALE) / (NUM_SCALES - 1)
  k = torch.round((log_s - np.log(MIN_SCALE)) / step)
  return k.clamp(0, NUM_SCALES - 1).long()


def get_terrain_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Each env's terrain scale (1 until `scale_terrain` has run)."""
  scale = getattr(env, "terrain_scale", None)
  if scale is None:
    scale = torch.ones(env.num_envs, device=env.device)
    env.terrain_scale = scale
  return scale


def scale_env_origins(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Set env origins from their terrain level and type, in their own scale."""
  terrain = env.scene.terrain
  scale = get_terrain_scale(env)[env_ids, None]
  terrain.env_origins[env_ids] = scale * terrain.terrain_origins[
    terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]


@requires_model_fields("geom_pos", "geom_size", "geom_rbound", "geom_aabb")
def scale_terrain(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  randomize: bool = False,
) -> None:
  """Scale each env's terrain to its robot and place it on the grid.

  Startup event, after `morphology`. Terrain types are shared out evenly, and
  evenly across robot sizes (envs sorted by leg length take the types in
  turn). With `randomize` (reset mode, for play) each reset only picks a random
  row and type instead.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)

  if randomize:
    terrain.terrain_types[env_ids] = torch.randint(0, NUM_TYPES, (len(env_ids),), device=env.device)
    terrain.terrain_levels[env_ids] = torch.randint(
      0, terrain.max_terrain_level, (len(env_ids),), device=env.device)
    scale_env_origins(env, env_ids)
    return

  k = _terrain_scale_index(env, env_ids)
  scale = torch.as_tensor(TERRAIN_SCALES, dtype=torch.float, device=env.device)[k]
  get_terrain_scale(env)[env_ids] = scale

  mj_model = env.sim.mj_model
  terrain_body = mj_model.body("terrain").id
  geom_ids = np.flatnonzero(mj_model.geom_bodyid == terrain_body)
  is_hfield = mj_model.geom_type[geom_ids] == mujoco.mjtGeom.mjGEOM_HFIELD
  boxes = torch.as_tensor(geom_ids[~is_hfield], device=env.device)
  hfields = torch.as_tensor(geom_ids[is_hfield], device=env.device)
  copy_index = torch.as_tensor(
    [_copy_index(mj_model.hfield(mj_model.geom_dataid[g]).name) for g in geom_ids[is_hfield]],
    device=env.device)

  model, data = env.sim.model, env.sim.data
  rows = env_ids[:, None]
  for field in ("geom_size", "geom_aabb", "geom_rbound"):
    model_field = getattr(model, field)
    default = env.sim.get_default_field(field)[boxes].reshape(len(boxes), *model_field.shape[2:])
    model_field[rows, boxes] = default * scale.view(-1, *([1] * default.dim()))

  # The terrain body sits at the origin, unrotated: world position = geom_pos.
  all_ids = torch.as_tensor(geom_ids, device=env.device)
  hfield_mask = torch.as_tensor(is_hfield, device=env.device)
  default_pos = env.sim.get_default_field("geom_pos")[all_ids].clone()
  default_pos[hfield_mask, 2] += _PARK_DEPTH * (copy_index != _DEFAULT_COPY)
  pos = default_pos * scale[:, None, None]
  parked = torch.zeros(len(env_ids), len(geom_ids), dtype=torch.bool, device=env.device)
  parked[:, hfield_mask] = copy_index[None] != k[:, None]
  pos[..., 2] -= _PARK_DEPTH * parked
  model.geom_pos[rows, all_ids] = pos
  data.geom_xpos[rows, all_ids] = pos

  # Terrain types evenly over envs and over robot sizes.
  order = env_ids[torch.argsort(get_morph_scales(env).length[env_ids])]
  terrain.terrain_types[order] = torch.arange(len(order), device=env.device) % NUM_TYPES
  scale_env_origins(env, env_ids)
