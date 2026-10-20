from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .scaling import get_morph_scales, normalized_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_heights_above_ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, ray_prefix: str | None
) -> torch.Tensor:
  """Height of each foot site [B, N] above the ground right below it.

  With `ray_prefix`, the ground is where the downward ray of sensor
  `<ray_prefix><site name>` hits (rough terrain); without, it is z = 0 (flat
  plane). A ray that misses counts as ground at the foot.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  if ray_prefix is None:
    return foot_z
  ids = asset_cfg.site_ids
  if isinstance(ids, slice):
    ids = range(len(asset.site_names))[ids]
  ground_z = torch.stack(
    [env.scene[f"{ray_prefix}{asset.site_names[i]}"].data.hit_pos_w[:, 0, 2] for i in ids],
    dim=1)
  return foot_z - ground_z


def foot_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ray_prefix: str | None = None,
) -> torch.Tensor:
  return foot_heights_above_ground(env, asset_cfg, ray_prefix)  # (num_envs, num_sites)


def knee_direction(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """+1 per leg whose knee bends forward, -1 backward, from the sign of the
  robot's default knee angle (">>" robots: all -1, "><": rear +1). joint_pos
  is relative to the default pose, so without this the two look the same."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sign(asset.data.default_joint_pos[:, asset_cfg.joint_ids])


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    """Gait clock. `period` is Go2's [s], scaled by each robot's time scale."""
    period = period * get_morph_scales(env).time
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    command = normalized_command(env, env.command_manager.get_command(command_name))
    stand_mask = torch.linalg.norm(command, dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

