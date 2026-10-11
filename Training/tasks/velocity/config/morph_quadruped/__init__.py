"""Procedurally generated quadruped velocity task (Go2-compatible)."""

from .env_cfgs import (
  morph_quadruped_flat_env_cfg as morph_quadruped_flat_env_cfg,
)
from .env_cfgs import (
  morph_quadruped_rough_env_cfg as morph_quadruped_rough_env_cfg,
)
from .env_cfgs import set_morphologies as set_morphologies
from .rl_cfg import morph_quadruped_ppo_runner_cfg as morph_quadruped_ppo_runner_cfg
