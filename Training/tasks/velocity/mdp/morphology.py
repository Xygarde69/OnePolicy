from __future__ import annotations

__all__ = ["apply_morphologies", "scaled_body_com_offset", "scaled_push_by_setting_velocity"]

from typing import TYPE_CHECKING

import numpy as np
import torch

from generation.bank import BODY_FIELDS, GEOM_FIELDS, MORPHOLOGY_FIELDS, SITE_FIELDS
from mjlab.actuator import DcMotorActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .scaling import get_morph_scales

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ACTUATOR_FIELDS = ("actuator_ctrlrange", "actuator_forcerange", "dof_armature")


@requires_model_fields(
  *MORPHOLOGY_FIELDS, *_ACTUATOR_FIELDS, recompute=RecomputeLevel.set_const
)
def apply_morphologies(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bank_file: str,
  nominal_mass: float,
  nominal_leg_length: float,
  scale_actuators: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Give env i the geometry and inertia of generated robot i (startup event).

  Writes every per-robot field from `bank_file` (see generation/bank.py) into
  the batched model, sets each env's spawn height from its leg lengths, its
  motors' velocity limit from the robot's max joint velocity and
  `env.morph_scales` (see scaling.py) from its size relative to Go2
  (`nominal_mass`, `nominal_leg_length`).

  With `scale_actuators`, Go2's actuators are rescaled to the robot: effort
  limits and stiffness by s_m * s_L (the torque to hold the body up),
  damping by s_m * s_L^1.5 and armature by s_m * s_L^2 (link inertia).
  Must run before other startup events that read these fields.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)

  bank = np.load(bank_file)
  if len(bank["seeds"]) != env.num_envs:
    raise ValueError(
      f"{bank_file} holds {len(bank['seeds'])} robots for {env.num_envs} envs.")
  bank_ids = env_ids.cpu().numpy()

  def per_env(key: str) -> torch.Tensor:
    return torch.as_tensor(bank[key][bank_ids], dtype=torch.float, device=env.device)

  scales = get_morph_scales(env)
  scales.length[env_ids] = per_env("leg_length") / nominal_leg_length
  scales.mass[env_ids] = per_env("total_mass") / nominal_mass
  scales.foot_radius[env_ids] = per_env("foot_radius")

  mj_model = env.sim.mj_model
  prefix = f"{asset_cfg.name}/"

  def scene_ids(kind: str, names) -> torch.Tensor:
    return torch.tensor([getattr(mj_model, kind)(prefix + str(n)).id for n in names],
                        device=env.device, dtype=torch.long)

  groups = (
    ("body", bank["body_names"], BODY_FIELDS),
    ("geom", bank["geom_names"], GEOM_FIELDS),
    ("site", bank["site_names"], SITE_FIELDS),
  )
  for kind, names, fields in groups:
    ids = scene_ids(kind, names)
    for field in fields:
      element_ids = ids
      values = bank[field][bank_ids]
      if field in ("body_pos", "body_quat"):
        # The free-floating root keeps the scene's placement.
        keep = np.array([n != "base_link" for n in names])
        element_ids, values = ids[torch.from_numpy(keep).to(env.device)], values[:, keep]
      model_field = getattr(env.sim.model, field)
      model_field[env_ids[:, None], element_ids] = torch.as_tensor(
        values, dtype=model_field.dtype, device=env.device
      ).reshape(len(env_ids), len(element_ids), *model_field.shape[2:])

  asset.data.default_root_state[env_ids, 2] = torch.as_tensor(
    bank["root_height"][bank_ids], dtype=torch.float, device=env.device)

  max_vel = per_env("max_joint_velocity")[:, None]
  s_m = scales.mass[env_ids, None]
  s_L = scales.length[env_ids, None]
  ones = torch.ones_like(s_m)
  torque = s_m * s_L if scale_actuators else ones
  damping = s_m * s_L**1.5 if scale_actuators else ones
  inertia = s_m * s_L**2 if scale_actuators else ones

  rows = env_ids[:, None]
  for act in asset.actuators:
    if not isinstance(act, DcMotorActuator):
      raise TypeError(f"Expected DcMotorActuator, got {type(act).__name__}.")
    act.stiffness[env_ids] = act.default_stiffness[env_ids] * torque
    act.damping[env_ids] = act.default_damping[env_ids] * damping
    act.force_limit[env_ids] = act.default_force_limit[env_ids] * torque
    act.saturation_effort[env_ids] = act.cfg.saturation_effort * torque
    act.velocity_limit_motor[env_ids] = max_vel
    # Corner velocity of the torque-speed curve (see DcMotorActuator.initialize).
    act._vel_at_effort_lim[env_ids] = act.velocity_limit_motor[env_ids] * (
      1 + act.force_limit[env_ids] / act.saturation_effort[env_ids])

    # The <motor> actuators also clamp ctrl and force at the unscaled limit.
    ctrl_ids = act.global_ctrl_ids
    for field in ("actuator_ctrlrange", "actuator_forcerange"):
      default = env.sim.get_default_field(field)[ctrl_ids]
      getattr(env.sim.model, field)[rows, ctrl_ids] = default * torque[..., None]
    dof_ids = asset.indexing.joint_v_adr[act.target_ids].long()
    default = env.sim.get_default_field("dof_armature")[dof_ids]
    env.sim.model.dof_armature[rows, dof_ids] = default * inertia


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def scaled_body_com_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: dict[int, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Shift body COMs by uniform offsets in Go2 metres, scaled by s_L (startup).

  Adds to each robot's own COM (mjlab's dr.body_com_offset "add" starts from
  the template robot's, which would overwrite the morphology).
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].long()
  length = get_morph_scales(env).length[env_ids]
  offset = torch.zeros(len(env_ids), len(body_ids), 3, device=env.device)
  for axis, (lo, hi) in ranges.items():
    offset[..., axis] = torch.empty_like(offset[..., axis]).uniform_(lo, hi)
  env.sim.model.body_ipos[env_ids[:, None], body_ids] += offset * length[:, None, None]


def scaled_push_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """mjlab's push_by_setting_velocity in Go2 units: the linear part (x, y, z)
  scaled by each robot's speed scale, the angular part by its inverse time
  scale."""
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in keys],
                        device=env.device)
  push = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(
    len(env_ids), 6, device=env.device)
  scales = get_morph_scales(env)
  push[:, :3] *= scales.speed[env_ids, None]
  push[:, 3:] /= scales.time[env_ids, None]
  asset.write_root_link_velocity_to_sim(vel_w + push, env_ids=env_ids)
