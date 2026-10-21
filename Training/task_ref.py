"""Reference run: train the velocity policy on Go2's own model only (RSL-RL).

Same env, rewards, curriculum, scaling and PPO settings as train.py, but every
env is a copy of the real Go2 (../quadrupeds/unitree_go2), adapted to the
training convention as play.py / train.py --generalise do, instead of a
procedurally generated robot:

  python task_ref.py                          # rough terrain, 2048 envs
  python task_ref.py --task flat
  python task_ref.py --num-envs 4096 --plot
  python task_ref.py --lvl 2 --agent.resume True   # full command ranges

Go2 is the unit of every size scale (s_L = s_m = 1), so it trains on mjlab's
unscaled rough grid. All --env.* and --agent.* overrides of train.py apply.
Logs go to logs/rsl_rl/<experiment>/<date>_go2_ref.
"""

import argparse
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import mjlab
import tyro

from train import (
  ENV_CFGS,
  Sim,
  TrainConfig,
  _drop_hfield_overflow_warnings,
  _launch,
)
from assets.morph_quadruped.constants import INIT_JOINT_POS
from generation.catalog import real_robot_bank
from tasks.velocity.config.morph_quadruped import set_morphologies
from tasks.velocity.velocity_env_cfg import TRAINING_LEVELS, set_training_level

ROBOT = "go2"


def launch_go2(args: TrainConfig) -> None:
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_dir = log_root_path / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                             + f"_{args.agent.run_name or 'go2_ref'}")

  if args.num_envs is not None:
    args.env.scene.num_envs = args.num_envs
  robot_dir = Path(tempfile.mkdtemp(prefix="go2_ref_"))
  try:
    bank_dir = robot_dir / ROBOT
    pose, notes = real_robot_bank(ROBOT, args.env.scene.num_envs, bank_dir, INIT_JOINT_POS)
    print(f"[INFO] {ROBOT} (own model) x {args.env.scene.num_envs}"
          + "".join(f"\n[INFO]   {n}" for n in notes), flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "morphology_seed.txt").write_text(
      f"real model {ROBOT} x {args.env.scene.num_envs}\n")
    set_morphologies(args.env, bank_dir, pose)
    _launch(args, log_dir, (Sim(ROBOT, bank_dir, pose, terrain_scale=1.0),))
  finally:
    shutil.rmtree(robot_dir, ignore_errors=True)


def main():
  parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
  parser.add_argument("--task", choices=list(ENV_CFGS), default="rough")
  parser.add_argument("--plot", nargs="?", const="True", default="False",
                      choices=["True", "False"])
  parser.add_argument("--lvl", type=int, choices=list(TRAINING_LEVELS), default=1)
  task_args, remaining_args = parser.parse_known_args()
  default = TrainConfig.from_task(task_args.task)
  set_training_level(default.env, task_args.lvl)
  default = replace(default, num_envs=2048, plot=task_args.plot == "True")

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=default,
    prog=sys.argv[0] + f" --task {task_args.task}",
    config=mjlab.TYRO_FLAGS,
  )

  with _drop_hfield_overflow_warnings():
    launch_go2(args)


if __name__ == "__main__":
  main()
