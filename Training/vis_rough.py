"""Look at the rough training terrain in a viewer.

The grid is mjlab's ROUGH_TERRAINS_CFG at one (Go2) scale for every robot:
num_rows difficulty rows x num_cols columns, each column one terrain type
(given out by proportion). One generated robot (--seed) is dropped on the patch
you pick and shown at rest in a MuJoCo viewer with a free camera, so you can fly
over the whole grid (right-drag to rotate, scroll to zoom, ctrl+right-drag to
pan).

  python vis_rough.py                         # row 0, column 0
  python vis_rough.py --row 7 --col 12
  python vis_rough.py --seed 5                # another robot
  python vis_rough.py --terrain-scale 1.5     # as train.py --terrain-scale
  python vis_rough.py --viewer False          # print the layout only

Rows go easy to hard.
"""

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import mjlab  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

from assets.morph_quadruped.constants import INIT_JOINT_POS  # noqa: E402
from generation.bank import MorphologyGenCfg, generate_bank  # noqa: E402
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_rough_env_cfg,
  set_morphologies,
)
from tasks.velocity.velocity_env_cfg import scale_terrain  # noqa: E402


@dataclass(frozen=True)
class VisConfig:
  seed: int = 0
  """Seed of the generated robot."""
  col: int = 0
  """Grid column (each column is one terrain type, listed at startup)."""
  row: int = 0
  """Difficulty row, 0 = easiest."""
  terrain_scale: float = 1.0
  """Scale of the terrain grid (as train.py --terrain-scale)."""
  viewer: bool = True
  device: str | None = None


def column_types(gen_cfg) -> list[str]:
  """Terrain type of each column, as mjlab's curriculum generator assigns them."""
  names = list(gen_cfg.sub_terrains)
  proportions = np.array([c.proportion for c in gen_cfg.sub_terrains.values()])
  cumsum = np.cumsum(proportions / proportions.sum())
  return [names[int(np.where(c / gen_cfg.num_cols + 0.001 < cumsum)[0][0])]
          for c in range(gen_cfg.num_cols)]


def main():
  cfg = tyro.cli(VisConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = morph_quadruped_rough_env_cfg()
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  scale_terrain(env_cfg, cfg.terrain_scale)
  gen_cfg = env_cfg.scene.terrain.terrain_generator
  num_rows, num_cols = gen_cfg.num_rows, gen_cfg.num_cols
  types = column_types(gen_cfg)
  print(f"[INFO] grid: {num_rows} rows x {num_cols} columns, "
        f"patch {gen_cfg.size[0]:g} x {gen_cfg.size[1]:g} m")
  print(f"[INFO] columns: {', '.join(f'{i}={t}' for i, t in enumerate(types))}")
  if not 0 <= cfg.row < num_rows or not 0 <= cfg.col < num_cols:
    raise SystemExit(f"row must be 0..{num_rows - 1}, col 0..{num_cols - 1}")

  with tempfile.TemporaryDirectory(prefix="vis_rough_") as tmp:
    generate_bank(1, Path(tmp), MorphologyGenCfg(seed=cfg.seed, workers=1), INIT_JOINT_POS)
    set_morphologies(env_cfg, Path(tmp))
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    terrain = env.scene.terrain
    terrain.terrain_levels[0] = cfg.row
    terrain.terrain_types[0] = cfg.col
    terrain.env_origins[0] = terrain.terrain_origins[cfg.row, cfg.col]
    env.reset()  # Spawns the robot at the new origin.
    origin = terrain.env_origins[0].cpu().numpy()
    print(f"[INFO] robot seed {cfg.seed}: column {cfg.col} ({types[cfg.col]}), "
          f"row {cfg.row}, origin {np.round(origin, 2).tolist()}")

    if cfg.viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      print("[WARN] No display found; not opening the viewer.")
      cfg = VisConfig(**{**cfg.__dict__, "viewer": False})
    if cfg.viewer:
      model = env.sim.mj_model
      data = mujoco.MjData(model)
      data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
      mujoco.mj_forward(model, data)
      with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                        show_right_ui=False) as handle:
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        handle.cam.lookat[:] = origin
        handle.cam.distance = 12.0
        handle.cam.elevation = -35.0
        handle.cam.azimuth = 135.0
        while handle.is_running():
          handle.sync()
          time.sleep(0.02)
    env.close()


if __name__ == "__main__":
  main()
