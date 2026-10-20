"""Train one velocity policy across procedurally generated quadrupeds (RSL-RL).

Before training, one random GenLoco quadruped is generated and compiled per
env, so env i simulates robot i:

  python train.py --num-envs 4096                # flat terrain
  python train.py --num-envs 4096 --task rough   # rough terrain, 10 terrain batches
  python train.py --num-envs 4096 --task rough --terrain-scales 1.0   # one grid
  python train.py --num-envs 4096 --morphology.seed 7
  python train.py --num-envs 4096 --plot         # also plot the mean reward
  python train.py --num-envs 4096 --generalise   # 1/2 generated, 1/2 real robots
  python train.py --num-envs 4096 --generalise 0.33 --generalise-robots spot go1

The robots live in a temporary directory that is deleted when training ends;
the base seed is saved as <log_dir>/morphology_seed.txt (rerun with
--morphology.seed <seed> to get the same robots). All --env.* and --agent.*
overrides from the Unitree train script still apply.

With --generalise F (bare flag: 0.5), a fraction F of the envs (0.33 = 33%)
is split equally between the real robots in --generalise-robots (default
Spot, A1, Barkour v0, Go2; any robot in ../quadrupeds, by name or unique
partial name) and the rest get generated robots. Each real robot is trained on
its own model, adapted as play.py does, in its own batched sim next to the
generated robots' (mjwarp batches one model structure per sim); PPO learns from
all sims at once (tasks/velocity/rl/multi_env.py). With --generalise-proxies
True, real robots are instead rebuilt in the generated structure
(generation/proxy.py): one sim, faster, but an approximation of the real
model. Episode reward is logged per robot type.

On rough terrain the robots are split into terrain batches (--terrain-scales,
default 0.3 0.5 0.7 1 1.5 2 3; the generator keeps legs within 0.225-3.5x
Go2's): each robot goes to the scale nearest its leg length vs Go2's (0.25 and
0.35 both go to 0.3), and each batch is its own sim on mjlab's rough grid
scaled by that value, placed and moved through the rows as Go2 is in Go2 units
(curriculum: half of the batch's scaled patch times the robot's command level
to move up). Real robots (--generalise) train on the batch nearest their size. Episode reward is
logged per batch.

With --plot, the mean episode reward per iteration is written to
<log_dir>/mean_reward.csv and drawn into <log_dir>/mean_reward.png (refreshed
every 10 iterations and at the end).
"""

import argparse
import contextlib
import copy
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

from assets.morph_quadruped.constants import INIT_JOINT_POS, NOMINAL_LEG_LENGTH  # noqa: E402
from generation.bank import BANK_FILE, MorphologyGenCfg, generate_bank, subset_bank  # noqa: E402
from generation.catalog import find_robot, real_robot_bank  # noqa: E402
from generation.proxy import make_proxy  # noqa: E402
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_flat_env_cfg,
  morph_quadruped_ppo_runner_cfg,
  morph_quadruped_rough_env_cfg,
  drop_body_contact_if_unmatched,
  set_morphologies,
)
from tasks.velocity.rl import MultiVecEnv, VelocityOnPolicyRunner  # noqa: E402
from tasks.velocity.rl.symmetry import mirror  # noqa: E402
from tasks.velocity.velocity_env_cfg import (  # noqa: E402
  TERRAIN_BATCH_SCALES,
  nearest_terrain_scale,
  scale_terrain,
)

Task = Literal["flat", "rough"]
ENV_CFGS = {"flat": morph_quadruped_flat_env_cfg, "rough": morph_quadruped_rough_env_cfg}


@dataclass(frozen=True)
class Sim:
  """One batched sim: a robot bank on one terrain scale."""

  name: str
  bank_dir: Path
  pose: dict | None = None
  """Default joint pose of a real robot's own model (None: generated)."""
  terrain_scale: float = 1.0


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
  export_onnx: bool = True
  """Export <log_dir>/policy.onnx at every checkpoint (the web UI turns it off:
  its runs are played from the .pt checkpoints)."""
  generalise: float = 0.0
  """Fraction of envs, in [0, 1), split equally between proxies of
  generalise_robots (generation/proxy.py); the rest are generated robots."""
  generalise_robots: tuple[str, ...] = ("spot", "a1", "barkour_v0", "go2")
  """Real robots (in ../quadrupeds) to train on with --generalise."""
  generalise_proxies: bool = False
  """Train on proxies of the real robots (their numbers in the generated
  structure, one sim) instead of their own models (one extra sim each)."""
  terrain_scales: tuple[float, ...] = TERRAIN_BATCH_SCALES
  """Rough task: terrain batches. Each robot goes to the batch whose scale is
  nearest its leg length vs Go2's; each batch is its own sim on mjlab's grid
  scaled by that value (patch size, steps, noise, waves, height scan, spawn
  offset), with Go2's placement and curriculum (move up after half of Go2's
  patch times the robot's speed scale). One value puts every robot on that
  one grid (1.0: mjlab's grid)."""
  symmetry: bool = True
  """Left-right mirror data augmentation in PPO (tasks/velocity/rl/symmetry.py).
  The generated robots are left-right symmetric, so every mirrored sample is a
  valid one; the policy also learns a symmetric gait."""
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task: Task) -> "TrainConfig":
    return TrainConfig(env=ENV_CFGS[task](), agent=morph_quadruped_ppo_runner_cfg())


def run_train(cfg: TrainConfig, log_dir: Path, sims: tuple[Sim, ...]) -> None:
  """sims: the generated robots' terrain batches, then each real robot trained
  on its own model (--generalise); PPO learns from all of them at once."""
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

  envs = []
  for i, sim in enumerate(sims):
    sim_cfg = copy.deepcopy(cfg.env)
    sim_cfg.scene.num_envs = len(np.load(sim.bank_dir / BANK_FILE)["seeds"])
    sim_cfg.seed = seed + 1000 * i
    set_morphologies(sim_cfg, sim.bank_dir, sim.pose)
    if sim.pose is not None:
      drop_body_contact_if_unmatched(sim_cfg, sim.bank_dir / "env_00000.xml")
    scale_terrain(sim_cfg, sim.terrain_scale)
    if len(sims) > 1:
      print(f"[INFO] Building the sim of {sim.name} ({sim_cfg.scene.num_envs} envs)",
            flush=True)
    env = ManagerBasedRlEnv(cfg=sim_cfg, device=device,
                            render_mode="rgb_array" if cfg.video and i == 0 else None)
    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and i == 0 and rank == 0:
      env = VideoRecorder(
        env,
        video_folder=Path(log_dir) / "videos" / "train",
        step_trigger=lambda step: step % cfg.video_interval == 0,
        video_length=cfg.video_length,
        disable_logger=True,
      )
      print("[INFO] Recording videos during training.")
    envs.append(RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions))
  env = envs[0] if len(envs) == 1 else MultiVecEnv(envs)

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    # Load checkpoint from local filesystem.
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  agent_cfg = asdict(cfg.agent)
  # The runner gets its own copy: rsl_rl adds the symmetry function and the env
  # to it, which params/agent.yaml can't hold.
  runner_cfg = asdict(cfg.agent)
  if cfg.symmetry:
    runner_cfg["algorithm"]["symmetry_cfg"] = {
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": mirror,
    }
  env_cfg = asdict(cfg.env)

  runner = VelocityOnPolicyRunner(env, runner_cfg, str(log_dir), device)
  runner.export_onnx = cfg.export_onnx

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


def generalise_split(num_envs: int, fraction: float, robots: tuple[str, ...],
                     proxies: bool) -> dict[str, int]:
  """Envs per real robot for --generalise: round(fraction * num_envs), split
  equally between `robots` (the first robots take any remainder)."""
  if not 0.0 <= fraction < 1.0:
    raise SystemExit(f"--generalise must be in [0, 1), got {fraction}.")
  if not robots:
    raise SystemExit("--generalise-robots is empty.")
  # Resolve partial names ("spot", "anymal_c") to robot file stems. Proxies
  # are built once now, as the bank builds them only after the generated
  # robots (real models are built before those anyway).
  names = list(dict.fromkeys(find_robot(r).stem for r in robots))
  for name in names if proxies else ():
    try:
      make_proxy(name, INIT_JOINT_POS, MorphologyGenCfg().foot_clearance)
    except ValueError as e:
      raise SystemExit(f"--generalise-robots: cannot build a proxy of {name}: {e}")
  num_real = round(num_envs * fraction)
  share, extra = divmod(num_real, len(names))
  split = {name: share + (i < extra) for i, name in enumerate(names)}
  print(f"[INFO] --generalise: {num_envs - num_real} generated robots, "
        + ", ".join(f"{n} x {c}" for n, c in split.items()), flush=True)
  return split


def _length_scale(bank_dir: Path) -> np.ndarray:
  """Leg length vs Go2's of every robot in a bank."""
  return np.load(bank_dir / BANK_FILE)["leg_length"] / NOMINAL_LEG_LENGTH


def terrain_batches(bank_dir: Path, scales: tuple[float, ...], out_root: Path) -> list[Sim]:
  """Split the generated robots into one sim per terrain scale: each robot goes
  to the scale nearest its leg length vs Go2's. Empty batches are left out."""
  if len(scales) == 1:
    return [Sim("generated", bank_dir, terrain_scale=scales[0])]
  length = _length_scale(bank_dir)
  batch = nearest_terrain_scale(length, scales)
  sims = []
  print(f"[INFO] Terrain batches ({len(length)} robots, leg length vs Go2):", flush=True)
  for k, scale in enumerate(scales):
    ids = np.flatnonzero(batch == k)
    if len(ids) == 0:
      print(f"[INFO]   x{scale:<5g}    0 robots (no sim)", flush=True)
      continue
    name = f"terrain_x{scale:g}"
    subset_bank(bank_dir, ids, out_root / name, source=name)
    sims.append(Sim(name, out_root / name, terrain_scale=scale))
    print(f"[INFO]   x{scale:<5g} {len(ids):4d} robots "
          f"({length[ids].min():.2f}-{length[ids].max():.2f})", flush=True)
  return sims


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
  terrain = args.env.scene.terrain
  rough = terrain is not None and terrain.terrain_generator is not None
  scales = tuple(args.terrain_scales) if rough else (1.0,)
  if not scales or min(scales) <= 0:
    raise SystemExit(f"--terrain-scales must be > 0, got {scales}.")
  # The robots are only read while the envs are built, but are kept until
  # training ends (multi-GPU workers build their envs later).
  morphology_dir = Path(tempfile.mkdtemp(prefix="morphologies_"))
  try:
    split = (generalise_split(args.env.scene.num_envs, args.generalise,
                              args.generalise_robots, args.generalise_proxies)
             if args.generalise else {})
    proxies = split if args.generalise_proxies else None
    real_sims = []
    if split and not args.generalise_proxies:
      # Real models first: a robot that can't be adapted fails before the
      # long generation step.
      for name, count in split.items():
        if count == 0:
          continue
        bank_dir = morphology_dir / f"real_{name}"
        pose, notes = real_robot_bank(name, count, bank_dir, INIT_JOINT_POS)
        print(f"[INFO] {name} (own model) x {count}"
              + "".join(f"\n[INFO]   {n}" for n in notes), flush=True)
        real_sims.append(Sim(name, bank_dir, pose))
      args.env.scene.num_envs -= sum(split.values())
    generated = morphology_dir / "generated"
    generate_bank(args.env.scene.num_envs, generated, args.morphology, INIT_JOINT_POS,
                  proxies)
    sims = terrain_batches(generated, scales, morphology_dir)
    # A real robot trains on the terrain batch nearest its own size.
    for sim in real_sims:
      scale = scales[int(nearest_terrain_scale(_length_scale(sim.bank_dir)[0], scales))]
      sims.append(replace(sim, terrain_scale=scale))
      if len(scales) > 1:
        print(f"[INFO] {sim.name} trains on terrain x{scale:g}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_seed = int(np.load(generated / BANK_FILE)["seeds"][0])
    kind = "proxies" if proxies else "real models"
    (log_dir / "morphology_seed.txt").write_text(
      f"{base_seed}\n" + (f"{kind} {split}\n" if split else ""))
    set_morphologies(args.env, generated)
    _launch(args, log_dir, tuple(sims))
  finally:
    shutil.rmtree(morphology_dir, ignore_errors=True)
    print(f"[INFO] Removed generated robots in {morphology_dir}", flush=True)


def _launch(args: TrainConfig, log_dir: Path, sims: tuple[Sim, ...]) -> None:
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
    run_train(args, log_dir, sims)
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
    ).run(run_train, args, log_dir, sims)


def main():
  # The task picks the default env config, so parse it before the rest.
  task_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
  task_parser.add_argument("--task", choices=list(ENV_CFGS), default="flat")
  # mjlab's tyro flags need `--plot True` and `--x "(a,b)"`; accept bare
  # `--plot`, bare `--generalise` (0.5), `--generalise-robots a b` / `a,b` and
  # `--terrain-scales 0.5 1 2`.
  task_parser.add_argument("--plot", nargs="?", const="True", default="False",
                           choices=["True", "False"])
  task_parser.add_argument("--generalise", nargs="?", const=0.5, default=0.0,
                           type=float)
  task_parser.add_argument("--generalise-robots", nargs="+")
  task_parser.add_argument("--terrain-scales", nargs="+", type=float,
                           default=list(TERRAIN_BATCH_SCALES))
  task_args, remaining_args = task_parser.parse_known_args()
  default = replace(TrainConfig.from_task(task_args.task),
                    plot=task_args.plot == "True",
                    generalise=task_args.generalise,
                    terrain_scales=tuple(task_args.terrain_scales))
  if task_args.generalise_robots:
    default = replace(default, generalise_robots=tuple(
      r for arg in task_args.generalise_robots for r in arg.split(",") if r))

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=default,
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
