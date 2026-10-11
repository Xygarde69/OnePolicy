"""Play the trained velocity policy on one quadruped (one env).

Pick the robot with exactly one of:

  python play.py --robot spot --policy <policy_path>     # a robot in ../quadrupeds
  python play.py --random True --policy <policy_path>    # a new random robot
  python play.py --random True --seed 42 --policy <policy_path>
  python play.py --robot-xml path/to/robot.xml           # any MJCF (or set ROBOT_XML)

--robot takes the name of a robot file in ../quadrupeds (spot, go2, go1, a1,
anymal_b, ...); run with --robot ? to list them. Named and custom robots are
rewritten into the training convention by generation/adapt.py, so any
quadruped with a floating base and 4 legs of 3 hinge joints (abduction, hip,
knee) works, whatever its names (see generation/adapt.py for the exact
convention). --random generates a robot exactly as train.py does.

Without --policy, the latest checkpoint of the latest training run is used.
--agent zero / random plays without a policy (to check the robot).
"""

import os
import re
import secrets
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import tyro  # noqa: E402

import mjlab  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.os import get_checkpoint_path  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

from assets.morph_quadruped.constants import INIT_JOINT_POS  # noqa: E402
from generation.adapt import adapt_quadruped, fit_pose  # noqa: E402
from generation.bank import (  # noqa: E402
  MorphologyGenCfg,
  generate_bank,
  single_robot_bank,
)
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_flat_env_cfg,
  morph_quadruped_ppo_runner_cfg,
  morph_quadruped_rough_env_cfg,
  set_morphologies,
)
from tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

# Placeholder: path to the robot MJCF to play (overridden by --robot-xml).
ROBOT_XML = "path/to/your_quadruped.xml"

QUADRUPEDS_DIR = Path(__file__).resolve().parent.parent / "quadrupeds"


@dataclass(frozen=True)
class PlayConfig:
  robot: str | None = None
  """Name of a robot in --quadrupeds-dir (e.g. spot, go2); '?' lists them."""
  random: bool = False
  """Play a newly generated random robot instead."""
  seed: int | None = None
  """Seed of the --random robot. None draws a fresh one."""
  robot_xml: str = ROBOT_XML
  """Robot MJCF (the robot file, not a scene that <include>s it)."""
  quadrupeds_dir: str = str(QUADRUPEDS_DIR)
  """Where --robot names are looked up."""
  policy: str | None = None
  """Policy checkpoint (model_*.pt). Defaults to the latest one in logs/."""
  task: Literal["flat", "rough"] = "flat"
  agent: Literal["zero", "random", "trained"] = "trained"
  """'trained' runs --policy; 'zero' / 'random' send zero / random actions."""
  forward: Literal["auto", "+x", "-x", "+y", "-y"] = "auto"
  """Base axis the robot faces. 'auto' takes the long axis (+x or +y)."""
  init_pose: Literal["auto", "train", "keyframe"] = "auto"
  """Default joint pose: the training pose fitted into the robot's joint ranges
  ("train"), the XML's first keyframe ("keyframe"), or keyframe if the XML has
  one, else train ("auto")."""
  scale_actuators_by_mass: bool = True
  """Scale Go2 PD gains / effort limits by robot mass, as in training."""
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions."""


def resolve_pose(pattern_pose: dict[str, float], names) -> dict[str, float]:
  """Expand a {regex: value} pose to {joint name: value}."""
  return {
    n: next(v for p, v in pattern_pose.items() if re.fullmatch(p, n)) for n in names
  }


def available_robots(quadrupeds_dir: Path) -> dict[str, Path]:
  """Robot name (file stem) -> robot MJCF, skipping scenes and MJX variants."""
  return {
    p.stem: p
    for p in sorted(quadrupeds_dir.glob("*/*.xml"))
    if not p.stem.startswith("scene") and not p.stem.endswith("_mjx")
  }


def find_robot(name: str, quadrupeds_dir: Path) -> Path:
  """Resolve a robot name: exact file stem, else a unique partial match."""
  robots = available_robots(quadrupeds_dir)
  listing = ", ".join(robots) or f"none found in {quadrupeds_dir}"
  if name == "?":
    raise SystemExit(f"Available robots: {listing}")
  key = name.lower()
  if key in robots:
    return robots[key]
  matches = [n for n in robots if key in n or key in robots[n].parent.name.lower()]
  if len(matches) == 1:
    return robots[matches[0]]
  if matches:
    raise ValueError(f"'{name}' matches several robots: {', '.join(matches)}.")
  raise ValueError(f"Unknown robot '{name}'. Available robots: {listing}")


def prepare_robot(cfg: PlayConfig) -> tuple[Path, dict[str, float] | None, str]:
  """Build the one-robot bank to play. Returns (bank dir, default pose, label)."""
  sources = [cfg.random, cfg.robot is not None, cfg.robot_xml != ROBOT_XML]
  if sum(sources) > 1:
    raise ValueError("Pass only one of --random, --robot and --robot-xml.")
  robot_dir = Path(tempfile.mkdtemp(prefix="play_robot_"))

  if cfg.random:
    seed = cfg.seed if cfg.seed is not None else secrets.randbits(31)
    generate_bank(1, robot_dir, MorphologyGenCfg(seed=seed, workers=1), INIT_JOINT_POS)
    return robot_dir, None, f"random_{seed}"

  if cfg.robot is not None:
    xml_path = find_robot(cfg.robot, Path(cfg.quadrupeds_dir))
  else:
    xml_path = Path(cfg.robot_xml)
    if not xml_path.is_file():
      raise FileNotFoundError(
        f"Robot MJCF not found: {xml_path}. Pass --robot <name>, --random or "
        "--robot-xml <path> (or set ROBOT_XML in play.py)."
      )
  print(f"[INFO]: Robot: {xml_path}")

  # Rewrite the robot into the training convention.
  robot = adapt_quadruped(str(xml_path), cfg.forward)
  for line in robot.report:
    print(f"[INFO]: {line}")
  use_keyframe = cfg.init_pose == "keyframe" or (
    cfg.init_pose == "auto" and robot.keyframe_joint_pos is not None
  )
  if use_keyframe:
    if robot.keyframe_joint_pos is None:
      raise ValueError(f"{xml_path} has no keyframe; use --init-pose train.")
    pose = robot.keyframe_joint_pos
    print("[INFO]: Default pose: the robot's keyframe")
  else:
    if not robot.zero_pose_straight:
      message = (
        f"{xml_path}: zero joint angles do not put the legs straight down, "
        "so the training pose does not transfer. Add a standing pose to the "
        "XML (<keyframe><key name=\"home\" qpos=\"...\"/></keyframe>)."
      )
      if cfg.init_pose == "auto":
        raise ValueError(message)
      print(f"[WARN]: {message}")
    pose, notes = fit_pose(resolve_pose(INIT_JOINT_POS, robot.joint_ranges),
                           robot.joint_ranges)
    print("[INFO]: Default pose: training pose")
    for line in notes:
      print(f"[INFO]:   {line}")

  single_robot_bank(robot.xml, robot.assets, robot_dir, pose)
  print(f"[INFO]: Adapted robot written to {robot_dir / 'env_00000.xml'}")
  return robot_dir, pose, xml_path.stem


def run_play(cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  robot_dir, pose, label = prepare_robot(cfg)

  env_cfg = (morph_quadruped_flat_env_cfg if cfg.task == "flat"
             else morph_quadruped_rough_env_cfg)(play=True)
  agent_cfg = morph_quadruped_ppo_runner_cfg()
  env_cfg.scene.num_envs = 1
  set_morphologies(env_cfg, robot_dir, pose)
  env_cfg.events["morphology"].params["scale_actuators_by_mass"] = (
    cfg.scale_actuators_by_mass
  )

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested.
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    if cfg.policy is not None:
      resume_path = Path(cfg.policy)
      if not resume_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {resume_path}")
    else:
      log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
      resume_path = get_checkpoint_path(log_root_path, ".*", r"model_.*\.pt")
    print(f"[INFO]: Loading checkpoint: {resume_path}")
    log_dir = resume_path.parent

  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play" / label,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner = VelocityOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  args = tyro.cli(PlayConfig, default=PlayConfig(), config=mjlab.TYRO_FLAGS)
  run_play(args)


if __name__ == "__main__":
  main()
