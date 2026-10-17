"""Look at the rough training terrain (tasks/velocity/terrain.py) in a viewer.

The grid is NUM_ROWS difficulty rows x NUM_BINS size bins x NUM_TYPES terrain
types. One generated robot (--seed) is dropped on the patch you pick and shown
at rest in a MuJoCo viewer with a free camera, so you can fly over the whole
grid (right-drag to rotate, scroll to zoom, ctrl+right-drag to pan).

  python vis_rough.py                         # robot's own bin, row 0, type 0
  python vis_rough.py --bin 3 --row 7 --kind 2
  python vis_rough.py --seed 5 --bin 0        # force a bin other than the robot's
  python vis_rough.py --viewer False          # print the layout only

Bins are leg length x Go2 (geometric, small to large); rows go easy to hard.
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
  BIN_SCALES,
  NUM_BINS,
  NUM_ROWS,
  NUM_TYPES,
)


@dataclass(frozen=True)
class VisConfig:
  seed: int = 0
  """Seed of the generated robot."""
  bin: int | None = None
  """Size bin 0..NUM_BINS-1 (default: the robot's own)."""
  kind: int = 0
  """Terrain type within the bin, in the order of mjlab's ROUGH_TERRAINS_CFG."""
  row: int = 0
  """Difficulty row, 0 = easiest."""
  viewer: bool = True
  device: str | None = None


def main():
  cfg = tyro.cli(VisConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  kinds = list(ROUGH_TERRAINS_CFG.sub_terrains)
  print(f"[INFO] grid: {NUM_ROWS} rows x {NUM_BINS} bins x {NUM_TYPES} types")
  print(f"[INFO] bin leg-length scales (x Go2): "
        f"{', '.join(f'{s:.2f}' for s in BIN_SCALES)}")
  print(f"[INFO] types: {', '.join(f'{i}={k}' for i, k in enumerate(kinds))}")
  if not 0 <= cfg.row < NUM_ROWS or not 0 <= cfg.kind < NUM_TYPES:
    raise SystemExit(f"row must be 0..{NUM_ROWS - 1}, kind 0..{NUM_TYPES - 1}")
  if cfg.bin is not None and not 0 <= cfg.bin < NUM_BINS:
    raise SystemExit(f"bin must be 0..{NUM_BINS - 1}")

  env_cfg = morph_quadruped_rough_env_cfg()
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  with tempfile.TemporaryDirectory(prefix="vis_rough_") as tmp:
    generate_bank(1, Path(tmp), MorphologyGenCfg(seed=cfg.seed, workers=1), INIT_JOINT_POS)
    set_morphologies(env_cfg, Path(tmp))
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    terrain = env.scene.terrain
    own_bin = int(terrain.terrain_types[0].item()) // NUM_TYPES
    size_bin = own_bin if cfg.bin is None else cfg.bin
    col = size_bin * NUM_TYPES + cfg.kind
    terrain.terrain_levels[0] = cfg.row
    terrain.terrain_types[0] = col
    terrain.env_origins[0] = terrain.terrain_origins[cfg.row, col]
    env.reset()  # Spawns the robot at the new origin.
    origin = terrain.env_origins[0].cpu().numpy()
    print(f"[INFO] robot seed {cfg.seed} is in bin {own_bin}; showing bin {size_bin} "
          f"(x{BIN_SCALES[size_bin]:.2f}), type {cfg.kind} ({kinds[cfg.kind]}), "
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
