import os
import statistics
from pathlib import Path

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


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def enable_reward_plot(self) -> None:
    """Record and plot the mean episode reward per iteration (rank 0 only)."""
    if self.logger.log_dir is None or self.logger.disable_logs:
      return
    self._reward_csv = Path(self.logger.log_dir) / "mean_reward.csv"
    self._reward_png = Path(self.logger.log_dir) / "mean_reward.png"
    self._reward_history: list[tuple[int, float]] = []
    self._reward_csv.write_text("iteration,mean_reward\n")
    log = self.logger.log

    def log_and_record(it, *a, **kw):
      log(it, *a, **kw)
      if not self.logger.rewbuffer:
        return
      mean_reward = statistics.mean(self.logger.rewbuffer)
      self._reward_history.append((it, mean_reward))
      with self._reward_csv.open("a") as f:
        f.write(f"{it},{mean_reward:.6f}\n")
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

    its, rewards = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(its, rewards, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"{Path(self.logger.log_dir).name}: mean reward")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(self._reward_png, dpi=120)
    plt.close(fig)

  def save(self, path: str, infos=None):
    super().save(path, infos)
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
