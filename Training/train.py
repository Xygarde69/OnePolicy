"""Train one velocity policy across procedurally generated quadrupeds (RSL-RL).

Before training, one random GenLoco quadruped is generated and compiled per
env, so env i simulates robot i:

  python train.py --num-envs 4096                # flat terrain
  python train.py --num-envs 4096 --task rough   # rough terrain
  python train.py --num-envs 4096 --morphology.seed 7
  python train.py --num-envs 4096 --plot         # also plot the mean reward

The robots live in a temporary directory that is deleted when training ends;
the base seed is saved as <log_dir>/morphology_seed.txt (rerun with
--morphology.seed <seed> to get the same robots). All --env.* and --agent.*
overrides from the Unitree train script still apply.

With --plot, the mean episode reward per iteration is written to
<log_dir>/mean_reward.csv and drawn into <log_dir>/mean_reward.png (refreshed
every 10 iterations and at the end).
"""

import argparse
import contextlib
import logging
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import tyro  # noqa: E402

import mjlab  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.gpu import select_gpus  # noqa: E402
from mjlab.utils.os import dump_yaml, get_checkpoint_path  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

from assets.morph_quadruped.constants import INIT_JOINT_POS  # noqa: E402
from generation.bank import BANK_FILE, MorphologyGenCfg, generate_bank  # noqa: E402
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_flat_env_cfg,
  morph_quadruped_ppo_runner_cfg,
  morph_quadruped_rough_env_cfg,
  set_morphologies,
)
from tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

Task = Literal["flat", "rough"]
ENV_CFGS = {"flat": morph_quadruped_flat_env_cfg, "rough": morph_quadruped_rough_env_cfg}


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  num_envs: int | None = None
  """Number of envs, i.e. of generated robots. Defaults to env.scene.num_envs."""
  morphology: MorphologyGenCfg = field(default_factory=MorphologyGenCfg)
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  plot: bool = False
  """Plot the mean episode reward per iteration to <log_dir>/mean_reward.png."""
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task: Task) -> "TrainConfig":
    return TrainConfig(env=ENV_CFGS[task](), agent=morph_quadruped_ppo_runner_cfg())


def run_train(cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    # Load checkpoint from local filesystem.
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  if cfg.plot:
    runner.enable_reward_plot()
  try:
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
  finally:
    # Also on Ctrl+C, so the plot covers every logged iteration.
    if cfg.plot:
      runner.plot_rewards()

  env.close()


def launch_training(args: TrainConfig):
  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Procedurally generate and compile one robot per env, before any worker starts.
  if args.num_envs is not None:
    args.env.scene.num_envs = args.num_envs
  # The robots are only read while the envs are built, but are kept until
  # training ends (multi-GPU workers build their envs later).
  morphology_dir = Path(tempfile.mkdtemp(prefix="morphologies_"))
  try:
    generate_bank(args.env.scene.num_envs, morphology_dir, args.morphology,
                  INIT_JOINT_POS)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_seed = int(np.load(morphology_dir / BANK_FILE)["seeds"][0])
    (log_dir / "morphology_seed.txt").write_text(f"{base_seed}\n")
    set_morphologies(args.env, morphology_dir)
    _launch(args, log_dir)
  finally:
    shutil.rmtree(morphology_dir, ignore_errors=True)
    print(f"[INFO] Removed generated robots in {morphology_dir}", flush=True)


def _launch(args: TrainConfig, log_dir: Path) -> None:
  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, args, log_dir)


def main():
  # The task picks the default env config, so parse it before the rest.
  task_parser = argparse.ArgumentParser(add_help=False)
  task_parser.add_argument("--task", choices=list(ENV_CFGS), default="flat")
  # mjlab's tyro flags need `--plot True`; accept a bare `--plot` too.
  task_parser.add_argument("--plot", nargs="?", const="True", default="False",
                           choices=["True", "False"])
  task_args, remaining_args = task_parser.parse_known_args()

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=replace(TrainConfig.from_task(task_args.task),
                    plot=task_args.plot == "True"),
    prog=sys.argv[0] + f" --task {task_args.task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  with _drop_hfield_overflow_warnings():
    launch_training(args)


# Printed by mujoco_warp's heightfield collider (on the GPU, so it can't be
# silenced from Python) when a large generated base lands on rough terrain and
# touches more than 50 triangles. The deepest contacts are still kept and
# illegal_contact ends the episode, so the message is only noise.
_HFIELD_OVERFLOW_LINES = (
  b"height field collision overflow",
  b" decrease the number of hfield rows/cols",
)


@contextlib.contextmanager
def _drop_hfield_overflow_warnings():
  """Filter the heightfield overflow message out of the process's stdout fd."""
  sys.stdout.flush()
  real_stdout = os.dup(1)
  read_fd, write_fd = os.pipe()
  os.dup2(write_fd, 1)
  os.close(write_fd)

  def forward():
    with os.fdopen(read_fd, "rb") as pipe:
      for line in pipe:
        if not line.startswith(_HFIELD_OVERFLOW_LINES):
          os.write(real_stdout, line)

  thread = threading.Thread(target=forward, daemon=True)
  thread.start()
  try:
    yield
  finally:
    sys.stdout.flush()
    os.dup2(real_stdout, 1)  # Closes the pipe's write end, ending forward().
    thread.join(timeout=5.0)
    os.close(real_stdout)


if __name__ == "__main__":
  main()
