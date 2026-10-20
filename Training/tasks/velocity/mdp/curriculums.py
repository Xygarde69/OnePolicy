from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Go2's terrain curriculum on each batch's own grid.

  Go2 moves up after walking half a patch in an episode. A robot moves up
  after half of its own (scaled) patch, so it has crossed as much of its
  terrain as Go2 has of Go2's, times its command level: commands are scaled
  by the level, so at level 0.5 a perfect tracker ends a 20 s random-heading
  episode a median 2.4 m out and could rarely move up."""
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains: half of their
  # batch's patch (scaled with the terrain), times their command level.
  command_level = env.command_manager.get_term(command_name).command_level[env_ids]
  move_up = distance > terrain_generator.size[0] / 2 * command_level

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def command_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  up_error: float = 0.25,
  down_error: float = 0.5,
  step: float = 0.1,
  min_level: float = 0.2,
  min_episode_s: float = 1.0,
) -> torch.Tensor:
  """Per-robot command curriculum, applied as each episode ends.

  A robot whose mean xy tracking error over the episode (Go2 units [m/s]) was
  below `up_error` and that did not fall gets its command level raised by
  `step`; one above `down_error`, or that fell, gets it lowered, within
  [min_level, 1]. Episodes shorter than `min_episode_s` are skipped. Unlike
  the global velocity stages, every robot advances at its own pace.
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  steps = command_term.episode_steps[env_ids]
  valid = steps * env.step_dt >= min_episode_s
  error = command_term.episode_error_sum[env_ids] / torch.clamp(steps, min=1)
  fell = env.termination_manager.terminated[env_ids]
  up = valid & (error < up_error) & ~fell
  down = valid & ((error > down_error) | fell)
  level = command_term.command_level
  level[env_ids] = torch.clamp(
    level[env_ids] + step * (up.float() - down.float()), min_level, 1.0)
  return torch.mean(level)


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
