import os
import statistics

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


# Once the mean episode reward (rsl_rl's "Mean reward") exceeds YAW_BOOST_REWARD,
# yaw-rate commands are sampled with BOOST_SCALE times their magnitude; once it
# then exceeds LIN_BOOST_REWARD, at least MIN_BOOST_GAP iterations after the yaw
# boost, x/y velocity commands are too. One-way.
YAW_BOOST_REWARD = 43.0
LIN_BOOST_REWARD = 45.0
BOOST_SCALE = 5.0
MIN_BOOST_GAP = 100


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._yaw_boost_it: int | None = None
    log = self.logger.log

    def log_and_maybe_boost(*a, **kw):
      log(*a, **kw)
      self._maybe_boost_commands()

    self.logger.log = log_and_maybe_boost

  def _maybe_boost_commands(self) -> None:
    cfg = self.env.unwrapped.command_manager.get_term("twist").cfg
    if cfg.lin_vel_scale != 1.0 or not self.logger.rewbuffer:
      return
    mean_reward = statistics.mean(self.logger.rewbuffer)
    if cfg.ang_vel_z_scale == 1.0:
      if mean_reward > YAW_BOOST_REWARD:
        cfg.ang_vel_z_scale = BOOST_SCALE
        self._yaw_boost_it = self.current_learning_iteration
        print(f"[INFO] Mean reward {mean_reward:.2f} > {YAW_BOOST_REWARD}: "
              f"yaw commands now x{BOOST_SCALE}", flush=True)
    elif (mean_reward > LIN_BOOST_REWARD and self._yaw_boost_it is not None
          and self.current_learning_iteration - self._yaw_boost_it >= MIN_BOOST_GAP):
      cfg.lin_vel_scale = BOOST_SCALE
      print(f"[INFO] Mean reward {mean_reward:.2f} > {LIN_BOOST_REWARD}: "
            f"x/y velocity commands now x{BOOST_SCALE}", flush=True)

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
