from __future__ import annotations

__all__ = ["apply_morphologies"]

from typing import TYPE_CHECKING

import numpy as np
import torch

from generation.bank import BODY_FIELDS, GEOM_FIELDS, MORPHOLOGY_FIELDS, SITE_FIELDS
from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ACTUATOR_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")


@requires_model_fields(
  *MORPHOLOGY_FIELDS, *_ACTUATOR_FIELDS, recompute=RecomputeLevel.set_const
)
def apply_morphologies(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bank_file: str,
  nominal_mass: float,
  scale_actuators_by_mass: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Give env i the geometry and inertia of generated robot i (startup event).

  Writes every per-robot field from `bank_file` (see generation/bank.py) into
  the batched model, sets each env's spawn height from its leg lengths and,
  optionally, scales PD gains and effort limits by robot mass / `nominal_mass`.
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

  if scale_actuators_by_mass:
    scale = torch.as_tensor(bank["total_mass"][bank_ids] / nominal_mass,
                            dtype=torch.float, device=env.device)[:, None]
    ctrl_ids = asset.indexing.ctrl_ids.long()
    gainprm = env.sim.get_default_field("actuator_gainprm")[ctrl_ids]
    biasprm = env.sim.get_default_field("actuator_biasprm")[ctrl_ids]
    forcerange = env.sim.get_default_field("actuator_forcerange")[ctrl_ids]
    rows = env_ids[:, None]
    env.sim.model.actuator_gainprm[rows, ctrl_ids, 0] = gainprm[:, 0] * scale
    env.sim.model.actuator_biasprm[rows, ctrl_ids, 1] = biasprm[:, 1] * scale
    env.sim.model.actuator_biasprm[rows, ctrl_ids, 2] = biasprm[:, 2] * scale
    env.sim.model.actuator_forcerange[rows, ctrl_ids, 0] = forcerange[:, 0] * scale
    env.sim.model.actuator_forcerange[rows, ctrl_ids, 1] = forcerange[:, 1] * scale
