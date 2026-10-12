"""Left-right mirror symmetry for PPO data augmentation (rsl_rl symmetry_cfg).

Mirroring about the base's x-z plane maps a quadruped state onto the state of
the same robot reflected left-right. Generated robots are left-right symmetric
(up to the randomised base COM), so the mirrored transition is a valid sample
and the policy can learn from both. rsl_rl calls `mirror(env, obs, actions)`
with a batch and expects [original; mirrored] back for each non-None input.

Per term (observation names from velocity_env_cfg.py):
  base_ang_vel     (wx, wy, wz) -> (-wx, wy, -wz)   (pseudovector)
  projected_gravity, base_lin_vel   y -> -y
  command          (vx, vy, wz) -> (vx, -vy, -wz)
  phase            -> -phase: swapping FR<->FL shifts the trot clock by half
                   a period (foot_gait offsets [0, 0.5, 0.5, 0])
  joint_pos/vel, actions   FR<->FL, RR<->RL; hip (abduction) joints negated
  height_scan      grid rows (y) reversed
  foot_*           FR<->FL, RR<->RL; contact force y negated (world frame,
                   approximate when the base is yawed; critic only)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper

_FEET = ("FR", "FL", "RR", "RL")
_MIRROR_LEG = {"FR": "FL", "FL": "FR", "RR": "RL", "RL": "RR"}

# Cached per env: {group: (index [D], sign [D])}, applied as obs[:, index] * sign.
_CACHE: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def _joint_map(names: list[str]) -> tuple[list[int], list[float]]:
  index = [names.index(_MIRROR_LEG[n[:2]] + n[2:]) for n in names]
  sign = [-1.0 if "_hip_" in n else 1.0 for n in names]
  return index, sign


def _foot_map(per_foot: int, negate: tuple[int, ...] = ()) -> tuple[list[int], list[float]]:
  index, sign = [], []
  for foot in _FEET:
    src = _FEET.index(_MIRROR_LEG[foot])
    for k in range(per_foot):
      index.append(src * per_foot + k)
      sign.append(-1.0 if k in negate else 1.0)
  return index, sign


def _term_map(env, term: str, dim: int) -> tuple[list[int], list[float]]:
  """(index, sign) of one observation term, local to the term."""
  identity = list(range(dim))
  if term == "base_ang_vel":
    return identity, [-1.0, 1.0, -1.0]
  if term in ("projected_gravity", "base_lin_vel"):
    return identity, [1.0, -1.0, 1.0]
  if term == "command":
    return identity, [1.0, -1.0, -1.0]
  if term == "phase":
    return identity, [-1.0] * dim
  if term in ("joint_pos", "joint_vel", "actions"):
    return _joint_map(list(env.scene["robot"].joint_names))
  if term == "height_scan":
    sensor = env.scene["terrain_scan"].cfg.pattern
    nx = int(round(sensor.size[0] / sensor.resolution)) + 1
    ny = dim // nx
    assert nx * ny == dim, f"height_scan of {dim} rays is not a {nx}-wide grid"
    # meshgrid(x, y, indexing="xy") flattens y-major: ray = iy * nx + ix.
    return [(ny - 1 - i // nx) * nx + i % nx for i in range(dim)], [1.0] * dim
  if term in ("foot_height", "foot_air_time", "foot_contact"):
    return _foot_map(1)
  if term == "foot_contact_forces":
    return _foot_map(3, negate=(1,))
  raise KeyError(
    f"No mirror rule for observation term '{term}'; add one to symmetry.py.")


def _group_maps(vec_env: RslRlVecEnvWrapper) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  env = vec_env.unwrapped
  if id(env) in _CACHE:
    return _CACHE[id(env)]
  om = env.observation_manager
  maps = {}
  for group in om.active_terms:
    index: list[int] = []
    sign: list[float] = []
    offset = 0
    for term, shape in zip(om.active_terms[group], om.group_obs_term_dim[group]):
      assert len(shape) == 1, f"{group}/{term}: expected a flat term, got {shape}"
      dim = shape[0]
      term_index, term_sign = _term_map(env, term, dim)
      assert len(term_index) == dim and len(term_sign) == dim, (group, term)
      index += [offset + i for i in term_index]
      sign += term_sign
      offset += dim
    maps[group] = (torch.tensor(index, device=env.device),
                   torch.tensor(sign, device=env.device))
  idx, sgn = _joint_map(list(env.scene["robot"].joint_names))
  maps["__actions__"] = (torch.tensor(idx, device=env.device),
                         torch.tensor(sgn, device=env.device))
  _CACHE[id(env)] = maps
  return maps


def mirror(env: RslRlVecEnvWrapper, obs: TensorDict | None = None,
           actions: torch.Tensor | None = None):
  """Return ([obs; mirrored obs], [actions; mirrored actions])."""
  maps = _group_maps(env)
  obs_out = None
  if obs is not None:
    groups = {}
    for key in obs.keys():
      index, sign = maps[key]
      groups[key] = torch.cat([obs[key], obs[key][:, index] * sign], dim=0)
    obs_out = TensorDict(groups, batch_size=[obs.batch_size[0] * 2])
  actions_out = None
  if actions is not None:
    index, sign = maps["__actions__"]
    actions_out = torch.cat([actions, actions[:, index] * sign], dim=0)
  return obs_out, actions_out
