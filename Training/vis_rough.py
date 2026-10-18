"""Look at the rough training terrain (tasks/velocity/terrain.py) in a viewer.

The grid is NUM_ROWS difficulty rows x NUM_TYPES terrain types, scaled to the
robot's leg length. One generated robot (--seed) is dropped on the patch you
pick and shown at rest in a MuJoCo viewer with a free camera, so you can fly
over the whole grid (right-drag to rotate, scroll to zoom, ctrl+right-drag to
pan). The viewer shows env 0's scaled terrain.

  python vis_rough.py                         # row 0, type 0
  python vis_rough.py --row 7 --kind 2
  python vis_rough.py --seed 5                # another robot (and size)
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
from mjlab.terrains.config import ROUGH_TERRAINS_CFG  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

from assets.morph_quadruped.constants import INIT_JOINT_POS  # noqa: E402
from generation.bank import MorphologyGenCfg, generate_bank  # noqa: E402
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_rough_env_cfg,
  set_morphologies,
)
from tasks.velocity.terrain import (  # noqa: E402
  NUM_ROWS,
  NUM_TYPES,
  get_terrain_scale,
  scale_env_origins,
)


@dataclass(frozen=True)
class VisConfig:
  seed: int = 0
  """Seed of the generated robot."""
  kind: int = 0
  """Terrain type, in the order of mjlab's ROUGH_TERRAINS_CFG."""
  row: int = 0
  """Difficulty row, 0 = easiest."""
  viewer: bool = True
  device: str | None = None


def main():
  cfg = tyro.cli(VisConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  kinds = list(ROUGH_TERRAINS_CFG.sub_terrains)
  print(f"[INFO] grid: {NUM_ROWS} rows x {NUM_TYPES} types")
  print(f"[INFO] types: {', '.join(f'{i}={k}' for i, k in enumerate(kinds))}")
  if not 0 <= cfg.row < NUM_ROWS or not 0 <= cfg.kind < NUM_TYPES:
    raise SystemExit(f"row must be 0..{NUM_ROWS - 1}, kind 0..{NUM_TYPES - 1}")

  env_cfg = morph_quadruped_rough_env_cfg()
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  with tempfile.TemporaryDirectory(prefix="vis_rough_") as tmp:
    generate_bank(1, Path(tmp), MorphologyGenCfg(seed=cfg.seed, workers=1), INIT_JOINT_POS)
    set_morphologies(env_cfg, Path(tmp))
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    terrain = env.scene.terrain
    terrain.terrain_levels[0] = cfg.row
    terrain.terrain_types[0] = cfg.kind
    scale_env_origins(env, torch.zeros(1, dtype=torch.long, device=device))
    env.reset()  # Spawns the robot at the new origin.
    origin = terrain.env_origins[0].cpu().numpy()
    print(f"[INFO] robot seed {cfg.seed}: terrain x{get_terrain_scale(env)[0]:.2f} Go2, "
          f"type {cfg.kind} ({kinds[cfg.kind]}), row {cfg.row}, "
          f"origin {np.round(origin, 2).tolist()}")

    if cfg.viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      print("[WARN] No display found; not opening the viewer.")
      cfg = VisConfig(**{**cfg.__dict__, "viewer": False})
    if cfg.viewer:
      model = env.sim.mj_model
      # Env 0's per-world terrain scaling, into the CPU model the viewer draws.
      for field in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb"):
        getattr(model, field)[:] = getattr(env.sim.model, field)[0].cpu().numpy().reshape(
          getattr(model, field).shape)
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
