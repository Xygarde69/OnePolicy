from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .scaling import get_morph_scales
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
  up_distance: float = 2.6,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Terrain curriculum on the distance a robot can cover.

  Commands are Go2's ranges times the robot's command level and speed scale
  sqrt(s_L), with a new random heading every 3-8 s, so a perfect tracker's
  net distance after an episode is proportional to
  sqrt(s_L) * command_level. A robot moves up after `up_distance` (Go2 m at
  level 1) times that, and down after less than half of it or a fall. At 2.6
  a perfect tracker of any size (0.24-3.37) and level (0.2-1) moves up ~80%
  and down ~5% of episodes (Monte Carlo of the command sampling). The old
  thresholds (half a patch, ~ s_L; half the last command x episode time) left
  even perfect 3x robots moving down more often than up."""
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Expected distance for this robot's size and command level.
  command_level = env.command_manager.get_term(command_name).command_level[env_ids]
  required = up_distance * get_morph_scales(env).speed[env_ids] * command_level

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > required

  # Robots that walked less than half of it, or fell, go to simpler terrains.
  fell = env.termination_manager.terminated[env_ids]
  move_down = (distance < 0.5 * required) | fell
  move_up &= ~fell
  move_down &= ~move_up

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
