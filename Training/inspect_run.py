"""Print a training run's TensorBoard scalars as a table, averaged over windows.

  python inspect_run.py                          # latest run, key scalars
  python inspect_run.py --run 2026-10-21_03-39-26 --window 100
  python inspect_run.py --tags Episode_Reward/   # every tag with this prefix
  python inspect_run.py --list                   # list all scalar tags
"""

import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = Path(__file__).parent / "logs/rsl_rl/morph_quadruped_velocity"
DEFAULT_TAGS = [
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Curriculum/terrain_levels",
  "Curriculum/command_levels",
  "Episode_Termination/time_out",
  "Episode_Termination/fell_over",
  "Episode_Termination/illegal_contact",
  "Metrics/twist/error_vel_xy",
  "Loss/learning_rate",
  "Policy/mean_std",
]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", help="run directory name (default: latest)")
  parser.add_argument("--window", type=int, default=50)
  parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                      help="tags, or prefixes ending in '/'")
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args()

  run = LOG_ROOT / args.run if args.run else max(p for p in LOG_ROOT.iterdir() if p.is_dir())
  ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
  ea.Reload()
  available = ea.Tags()["scalars"]
  print(run.name)
  if args.list:
    print("\n".join(available))
    return

  tags = [t for want in args.tags for t in available
          if t == want or (want.endswith("/") and t.startswith(want))]
  data = {t: np.array([s.value for s in ea.Scalars(t)]) for t in tags}
  n = max(len(v) for v in data.values())
  print(f"{n} iterations, mean over windows of {args.window}")
  for t in tags:
    print(f"{t:<45s}" + " ".join(
      f"{data[t][a:a + args.window].mean():9.4g}" for a in range(0, len(data[t]), args.window)))


if __name__ == "__main__":
  main()
