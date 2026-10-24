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
--joystick True drives the robot with a DualSense: right stick up/down =
forward/back, left/right = sideways; L1 turns left, R1 turns right (gamepad.py).
--command-port P takes the command from UDP datagrams "forward left yaw" on
127.0.0.1:P instead (the web harness's keyboard driving); no datagram for
0.5 s means stand still.

--task rough plays on a 5 x 5 grid of the training terrains, each patch a
random type and difficulty. --terrain-type and --difficulty fix them to test
one terrain:

  python play.py --robot go2 --task rough --terrain-type pyramid_stairs --difficulty 0.8
"""

import os
import secrets
import socket
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import mjlab  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.terrains.config import ROUGH_TERRAINS_CFG  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.os import get_checkpoint_path  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

from assets.morph_quadruped.constants import INIT_JOINT_POS, NOMINAL_LEG_LENGTH  # noqa: E402
from gamepad import DualSenseGamepad  # noqa: E402
from generation.catalog import (  # noqa: E402
  QUADRUPEDS_DIR,
  find_robot,
  load_robot,
  robot_defaults,
)
from generation.bank import (  # noqa: E402
  BANK_FILE,
  MorphologyGenCfg,
  generate_bank,
  single_robot_bank,
)
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_flat_env_cfg,
  morph_quadruped_ppo_runner_cfg,
  morph_quadruped_rough_env_cfg,
  drop_body_contact_if_unmatched,
  set_morphologies,
)
from tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402
from tasks.velocity.velocity_env_cfg import (  # noqa: E402
  TERRAIN_BATCH_SCALES,
  nearest_terrain_scale,
  scale_terrain,
)

# Placeholder: path to the robot MJCF to play (overridden by --robot-xml).
ROBOT_XML = "path/to/your_quadruped.xml"



@dataclass(frozen=True)
class PlayConfig:
  robot: str | None = None
  """Name of a robot in --quadrupeds-dir (e.g. spot, go2); '?' lists them."""
  random: bool = False
  """Play a newly generated random robot instead."""
  seed: int | None = None
  """Seed of the --random robot. None draws a fresh one."""
  gen_params: str = ""
  """Generator values for the --random robot, as JSON (MorphologyGenCfg.params),
  e.g. '{"scale": 1.5, "x_knee": true}'."""
  robot_xml: str = ROBOT_XML
  """Robot MJCF (the robot file, not a scene that <include>s it)."""
  quadrupeds_dir: str = str(QUADRUPEDS_DIR)
  """Where --robot names are looked up."""
  policy: str | None = None
  """Policy checkpoint (model_*.pt). Defaults to the latest one in logs/."""
  task: Literal["flat", "rough"] = "flat"
  terrain_scale: float | None = None
  """Scale of the rough terrain grid; default: the train.py terrain batch
  nearest the robot's leg length vs Go2's."""
  terrain_type: Literal["flat", "pyramid_stairs", "pyramid_stairs_inv", "hf_pyramid_slope",
                        "hf_pyramid_slope_inv", "random_rough", "wave_terrain"] | None = None
  """Rough task: every patch this terrain type. None: random types, as in training."""
  difficulty: float | None = None
  """Rough task: every patch this difficulty, 0 (easiest) to 1 (training's top row,
  e.g. 10 cm stairs for Go2). None: random in 0..1."""
  agent: Literal["zero", "random", "trained"] = "trained"
  """'trained' runs --policy; 'zero' / 'random' send zero / random actions."""
  forward: Literal["auto", "+x", "-x", "+y", "-y"] = "auto"
  """Base axis the robot faces. 'auto' takes the long axis (+x or +y)."""
  init_pose: Literal["auto", "train", "keyframe"] = "auto"
  """Default joint pose: the training pose fitted into the robot's joint ranges
  ("train"), the XML's first keyframe ("keyframe"), or keyframe if the XML has
  one, else train ("auto")."""
  scale_actuators: bool = True
  """Scale Go2 PD gains / effort limits to the robot's size, as in training."""
  own_motor_limits: bool = False
  """Keep the MJCF's joint actuatorfrcrange (its own motors' torque limits)
  instead of letting the training actuators give what the policy expects."""
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions."""
  joystick: bool = False
  """Drive with a DualSense: right stick = forward/back and sideways, L1/R1 = turn
  left/right."""
  joystick_device: str = "/dev/input/js0"
  joystick_max_forward: float | None = None
  """Full-stick forward speed [Go2 m/s, scaled to the robot]. None: the task's
  command range."""
  joystick_max_backward: float | None = None
  """Full-stick backward speed [Go2 m/s]. None: the task's command range."""
  joystick_max_lateral: float | None = None
  """Full-stick sideways speed [Go2 m/s]. None: the task's command range."""
  joystick_max_yaw: float | None = None
  """Yaw rate while L1/R1 is held [Go2 rad/s]. None: the task's command range."""
  command: tuple[float, ...] = ()
  """Fixed velocity command "(forward, left, yaw)" [m/s, m/s, rad/s] in Go2
  units, scaled to the robot, instead of random commands, e.g.
  --command "(1.0, 0.0, 0.2)". Ignored with --joystick."""
  joystick_scale: float = 2.0
  """Multiplies all four joystick speeds above (forward, backward, sideways,
  yaw), whether defaulted or given."""
  command_port: int | None = None
  """Take the command (Go2 units) from UDP "forward left yaw" datagrams on
  127.0.0.1:<port>. Overrides --command."""


def prepare_robot(cfg: PlayConfig, robot_dir: Path):
  """Build the one-robot bank to play in robot_dir.

  Returns (default pose, label, catalog.LoadedRobot or None for --random)."""
  sources = [cfg.random, cfg.robot is not None, cfg.robot_xml != ROBOT_XML]
  if sum(sources) > 1:
    raise ValueError("Pass only one of --random, --robot and --robot-xml.")

  if cfg.random:
    seed = cfg.seed if cfg.seed is not None else secrets.randbits(31)
    generate_bank(1, robot_dir, MorphologyGenCfg(seed=seed, workers=1, params=cfg.gen_params),
                  INIT_JOINT_POS)
    return None, f"random_{seed}", None

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

  # Per-robot defaults for options left on "auto" (see robot_defaults).
  defaults = robot_defaults(xml_path)
  forward = defaults.get("forward", "auto") if cfg.forward == "auto" else cfg.forward
  if forward != cfg.forward:
    print(f"[INFO]: Forward axis {forward} (from {xml_path.parent / 'play.json'})")

  # Fix the MJCF (writes <stem>.temp.xml next to it, see generation/mjcf_fix.py),
  # then rewrite that into the training convention.
  loaded = load_robot(xml_path, forward, cfg.init_pose, INIT_JOINT_POS,
                      own_motor_limits=cfg.own_motor_limits)
  for line in loaded.fixed.summary().splitlines():
    print(f"[INFO]: {line}")
  for line in loaded.robot.report:
    print(f"[INFO]: {line}")
  for line in loaded.log:
    print(f"[WARN]: {line[5:]}" if line.startswith("WARN ") else f"[INFO]: {line}")
  pose = loaded.pose

  single_robot_bank(loaded.robot.xml, loaded.robot.assets, robot_dir, pose)
  print(f"[INFO]: Adapted robot written to {robot_dir / 'env_00000.xml'}")
  return pose, xml_path.stem, loaded


def attach_gamepad(env: ManagerBasedRlEnv, cfg: PlayConfig) -> None:
  """Let the DualSense set the velocity command (right stick, L1/R1)."""
  term = env.command_manager.get_term("twist")
  ranges = term.cfg.ranges
  def pick(value: float | None, default: float) -> float:
    return default if value is None else value

  k = cfg.joystick_scale
  max_forward = k * pick(cfg.joystick_max_forward, ranges.lin_vel_x[1])
  max_backward = k * pick(cfg.joystick_max_backward, -ranges.lin_vel_x[0])
  max_lateral = k * pick(cfg.joystick_max_lateral, ranges.lin_vel_y[1])
  max_yaw = k * pick(cfg.joystick_max_yaw, ranges.ang_vel_z[1])
  pad = DualSenseGamepad(cfg.joystick_device)

  def command() -> tuple[float, float, float]:
    forward, left, turn = pad.read()
    return (forward * (max_forward if forward > 0 else max_backward),
            left * max_lateral, turn * max_yaw)

  term.gamepad = command
  print(f"[INFO]: Joystick {cfg.joystick_device} (Go2 units, scaled to the robot): "
        f"right stick up/down = forward {max_forward:.2f} / back {max_backward:.2f} "
        f"m/s, left/right = sideways {max_lateral:.2f} m/s; L1/R1 = turn left/right "
        f"{max_yaw:.2f} rad/s")


# Without a UDP command for this long, the robot is told to stand still.
COMMAND_TIMEOUT_S = 0.5


def attach_command_port(env: ManagerBasedRlEnv, port: int) -> None:
  """Let UDP datagrams "forward left yaw" (Go2 units) set the velocity command."""
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind(("127.0.0.1", port))
  latest = {"command": (0.0, 0.0, 0.0), "time": 0.0}

  def listen():
    while True:
      data = sock.recv(256)
      try:
        forward, left, yaw = (float(v) for v in data.decode().split())
      except ValueError:
        continue
      latest.update(command=(forward, left, yaw), time=time.monotonic())

  threading.Thread(target=listen, daemon=True).start()

  def command() -> tuple[float, float, float]:
    if time.monotonic() - latest["time"] > COMMAND_TIMEOUT_S:
      return (0.0, 0.0, 0.0)
    return latest["command"]

  env.command_manager.get_term("twist").gamepad = command
  print(f"[INFO]: Commands from UDP 127.0.0.1:{port}", flush=True)


def pick_terrain(env_cfg, terrain_type: str | None, difficulty: float | None) -> None:
  """Make every patch of the play grid `terrain_type` at `difficulty` (None: random)."""
  gen = env_cfg.scene.terrain.terrain_generator
  if terrain_type is not None:
    gen = replace(gen, sub_terrains={terrain_type: gen.sub_terrains[terrain_type]})
    print(f"[INFO] Terrain type: {terrain_type}")
  if difficulty is not None:
    if not 0.0 <= difficulty <= 1.0:
      raise ValueError(f"--difficulty must be in [0, 1], got {difficulty}.")
    gen = replace(gen, difficulty_range=(difficulty, difficulty))
    print(f"[INFO] Terrain difficulty: {difficulty:g} (training row "
          f"{min(int(difficulty * ROUGH_TERRAINS_CFG.num_rows), ROUGH_TERRAINS_CFG.num_rows - 1)})")
  env_cfg.scene.terrain.terrain_generator = gen


def run_play(cfg: PlayConfig):
  # The robot is deleted when play ends, however it ends.
  robot_dir = Path(tempfile.mkdtemp(prefix="play_robot_"))
  try:
    _play(cfg, robot_dir)
  finally:
    shutil.rmtree(robot_dir, ignore_errors=True)


def _play(cfg: PlayConfig, robot_dir: Path):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  pose, label, _ = prepare_robot(cfg, robot_dir)

  env_cfg = (morph_quadruped_flat_env_cfg if cfg.task == "flat"
             else morph_quadruped_rough_env_cfg)(play=True)
  terrain_scale = cfg.terrain_scale
  if terrain_scale is None:
    length = np.load(robot_dir / BANK_FILE)["leg_length"][0] / NOMINAL_LEG_LENGTH
    terrain_scale = TERRAIN_BATCH_SCALES[int(nearest_terrain_scale(length))]
  scale_terrain(env_cfg, terrain_scale)
  if cfg.task == "rough":
    print(f"[INFO] Rough terrain scaled x{terrain_scale:g}")
    pick_terrain(env_cfg, cfg.terrain_type, cfg.difficulty)
    # nconmax is a pool shared by all envs: training's 64 x 2048 absorbs a robot
    # on a heightfield slope (~70 contacts), one env's 64 overflows.
    env_cfg.sim.nconmax = 512
  elif cfg.terrain_type is not None or cfg.difficulty is not None:
    raise ValueError("--terrain-type / --difficulty need --task rough.")
  agent_cfg = morph_quadruped_ppo_runner_cfg()
  env_cfg.scene.num_envs = 1
  set_morphologies(env_cfg, robot_dir, pose)
  drop_body_contact_if_unmatched(env_cfg, robot_dir / "env_00000.xml")
  env_cfg.events["morphology"].params["scale_actuators"] = cfg.scale_actuators

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

  if cfg.joystick:
    attach_gamepad(env, cfg)
  elif cfg.command_port is not None:
    attach_command_port(env, cfg.command_port)
  elif cfg.command:
    if len(cfg.command) != 3:
      raise ValueError(f"--command takes 3 numbers (forward left yaw), got {cfg.command}")
    command = tuple(cfg.command)
    env.command_manager.get_term("twist").gamepad = lambda: command
    print(f"[INFO]: Fixed command (forward, left, yaw) = {command} (Go2 units)")

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
