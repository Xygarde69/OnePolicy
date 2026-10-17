import os
import statistics
from collections import deque
from pathlib import Path

import numpy as np
import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


# With --plot, the mean episode reward (rsl_rl's "Mean reward") is appended to
# <log_dir>/mean_reward.csv every iteration and drawn into mean_reward.png every
# PLOT_EVERY iterations and when training ends.
PLOT_EVERY = 10

# Finished-episode returns are grouped, to catch the policy favouring some
# bodies: by robot type (generated, spot, a1, ...) when the bank mixes types
# (train.py --generalise), else into thirds by leg length. Each group's mean is
# logged as Morphology/mean_reward_<group> (and plotted with --plot).
SIZE_BUCKETS = ("small", "medium", "large")

# Episodes shorter than this [s] are left out of the group means: a robot that
# dies right after every reset finishes ~170 episodes per 20 s and would
# otherwise fill its group's window on its own.
MIN_EPISODE_S = 0.5

# Reward terms whose per-step weighted value is also split by group, logged as
# Morphology/<term>_<group> (iteration mean), with each group's mean command
# level as Morphology/command_level_<group>. Shows which terms a lagging
# group pays for, and whether it is held back by the curriculum.
SPLIT_TERMS = ("track_linear_velocity", "track_angular_velocity",
               "action_rate_l2", "action_acc_l2")


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper
  # Export policy.onnx next to every saved checkpoint (train.py --export-onnx).
  export_onnx: bool = True

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # (group, return) of the last 100 episodes of at least MIN_EPISODE_S: the
    # same window size as rsl_rl's rewbuffer (which keeps short ones too).
    self._sized_returns: deque[tuple[str, float]] = deque(maxlen=100)
    self._groups: tuple[str, ...] = ()
    self._track_size_buckets()

  def _track_size_buckets(self) -> None:
    """Split finished-episode returns by robot type or size (rank 0 only)."""
    if self.logger.log_dir is None or self.logger.disable_logs:
      return
    env = self.env.unwrapped
    # A MultiVecEnv (train.py --generalise) lists the robots of all its sims.
    sources = getattr(self.env, "morph_sources", getattr(env, "morph_sources", None))
    if sources is not None and len(set(sources)) > 1:
      self._groups = tuple(dict.fromkeys(sources))  # First-seen order.
      self._size_bucket = torch.tensor([self._groups.index(s) for s in sources],
                                       device=self.env.device)
      self._group_label = {g: g for g in self._groups}
    else:
      self._groups = SIZE_BUCKETS
      length = env.morph_scales.length
      edges = torch.quantile(length, torch.tensor([1 / 3, 2 / 3], device=length.device))
      self._size_bucket = torch.bucketize(length, edges)  # [N] in {0, 1, 2}
      self._group_label = {g: f"{g} legs" for g in self._groups}
    # A MultiVecEnv concatenates its sims' envs in batch order.
    self._sub_envs = [e.unwrapped for e in getattr(self.env, "envs", [self.env])]
    names = self._sub_envs[0].reward_manager.active_terms
    self._split_terms = [t for t in SPLIT_TERMS if t in names]
    self._split_idx = [names.index(t) for t in self._split_terms]
    self._group_size = torch.bincount(self._size_bucket, minlength=len(self._groups))
    self._term_sums = torch.zeros(len(self._groups), len(self._split_terms),
                                  device=self.env.device)
    self._term_steps = 0
    process = self.logger.process_env_step
    log = self.logger.log
    min_steps = MIN_EPISODE_S / self._sub_envs[0].step_dt

    def process_and_split(rewards, dones, extras, *a, **kw):
      # rsl_rl counts this step into cur_episode_length in process() below.
      long_enough = self.logger.cur_episode_length.flatten() + 1 >= min_steps
      done = (dones > 0).flatten() & long_enough
      if torch.any(done):
        returns = (self.logger.cur_reward_sum + rewards.flatten())[done].tolist()
        for bucket, ret in zip(self._size_bucket[done].tolist(), returns):
          self._sized_returns.append((self._groups[bucket], ret))
      step = torch.cat([e.reward_manager._step_reward[:, self._split_idx]
                        for e in self._sub_envs])
      self._term_sums.index_add_(0, self._size_bucket, step)
      self._term_steps += 1
      process(rewards, dones, extras, *a, **kw)

    def log_with_sizes(it, *a, **kw):
      log(it, *a, **kw)
      if self.logger.writer is None:
        return
      for bucket, mean in self._size_means().items():
        if mean is not None:
          self.logger.writer.add_scalar(f"Morphology/mean_reward_{bucket}", mean, it)
      self._log_split_terms(it)

    self.logger.process_env_step = process_and_split
    self.logger.log = log_with_sizes

  def _log_split_terms(self, it: int) -> None:
    """Log each group's iteration-mean SPLIT_TERMS values and command level."""
    if self._term_steps:
      size = torch.clamp(self._group_size, min=1)[:, None] * self._term_steps
      means = (self._term_sums / size).tolist()
      for g, bucket in enumerate(self._groups):
        if self._group_size[g] == 0:
          continue
        for t, term in enumerate(self._split_terms):
          self.logger.writer.add_scalar(f"Morphology/{term}_{bucket}", means[g][t], it)
    self._term_sums.zero_()
    self._term_steps = 0
    levels = []
    for e in self._sub_envs:
      term = e.command_manager.get_term("twist")
      if term is None or not hasattr(term, "command_level"):
        return
      levels.append(term.command_level)
    level = torch.cat(levels)
    for g, bucket in enumerate(self._groups):
      if self._group_size[g]:
        mean = level[self._size_bucket == g].mean().item()
        self.logger.writer.add_scalar(f"Morphology/command_level_{bucket}", mean, it)

  def _size_means(self) -> dict[str, float | None]:
    """Mean recent episode return per group (None if it has none)."""
    means = {}
    for bucket in self._groups:
      returns = [r for b, r in self._sized_returns if b == bucket]
      means[bucket] = statistics.mean(returns) if returns else None
    return means

  def enable_reward_plot(self) -> None:
    """Record and plot the mean episode reward per iteration (rank 0 only)."""
    if self.logger.log_dir is None or self.logger.disable_logs:
      return
    self._reward_csv = Path(self.logger.log_dir) / "mean_reward.csv"
    self._reward_png = Path(self.logger.log_dir) / "mean_reward.png"
    # (iteration, overall mean, {bucket: mean or None})
    self._reward_history: list[tuple[int, float, dict]] = []
    columns = ",".join(f"mean_reward_{b}" for b in self._groups)
    self._reward_csv.write_text(f"iteration,mean_reward,{columns}\n")
    log = self.logger.log

    def log_and_record(it, *a, **kw):
      log(it, *a, **kw)
      if not self.logger.rewbuffer:
        return
      mean_reward = statistics.mean(self.logger.rewbuffer)
      sizes = self._size_means()
      self._reward_history.append((it, mean_reward, sizes))
      cells = ",".join("" if v is None else f"{v:.6f}" for v in sizes.values())
      with self._reward_csv.open("a") as f:
        f.write(f"{it},{mean_reward:.6f},{cells}\n")
      if len(self._reward_history) % PLOT_EVERY == 0:
        self.plot_rewards()

    self.logger.log = log_and_record
    print(f"[INFO] Plotting mean reward to {self._reward_png}", flush=True)

  def plot_rewards(self) -> None:
    history = getattr(self, "_reward_history", None)
    if not history:
      return
    import matplotlib

    matplotlib.use("Agg")  # Training renders with EGL; no display needed.
    import matplotlib.pyplot as plt

    its = [h[0] for h in history]
    lines = [("all", its, [h[1] for h in history])]
    for bucket in self._groups:
      points = [(h[0], h[2][bucket]) for h in history if h[2][bucket] is not None]
      if points:
        lines.append((self._group_label[bucket], *zip(*points)))
    # A rare simulation blow-up gives one iteration a mean of ~-1e12, which
    # would flatten everything else: drop points beyond a 3 IQR fence (the
    # line breaks there) and report how many. Blow-ups are orders of
    # magnitude off, so the fence also gets a floor of the rewards' own size,
    # which keeps early iterations (tiny IQR) intact.
    values = np.array([v for _, _, y in lines for v in y])
    q1, q3 = np.percentile(values[np.isfinite(values)], [25, 75])
    fence = 3 * (q3 - q1) + max(abs(q1), abs(q3), 1.0)
    lo, hi = q1 - fence, q3 + fence
    off_scale = int(np.sum(~((values >= lo) & (values <= hi))))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, x, y in lines:
      y = np.array(y, dtype=float)
      y[~((y >= lo) & (y <= hi))] = np.nan
      style = (dict(linewidth=1.8, color="black") if label == "all"
               else dict(linewidth=1.0, alpha=0.8))
      ax.plot(x, y, label=label, **style)
    if off_scale:
      ax.text(0.01, 0.01, f"{off_scale} outlier points not drawn (see csv)",
              transform=ax.transAxes, fontsize=8, color="gray")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"{Path(self.logger.log_dir).name}: mean reward")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(self._reward_png, dpi=120)
    plt.close(fig)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    if not self.export_onnx:
      return
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
