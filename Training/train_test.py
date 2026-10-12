"""Check generated robots against the training thresholds, without training.

Spawns random robots one after another, each alone in a training env (same
rewards, terminations, pushes and randomisation as train.py), and runs it until
a termination fires or --max-seconds pass. Each robot is shown in a MuJoCo
viewer (--slow-motion times slower than real time, contact points and forces
drawn; the final frame is held for
--hold-seconds, closing the window skips to the next robot). For every robot it
prints its anatomy, its per-robot thresholds (Go2's, rescaled; see
tasks/velocity/mdp/scaling.py) and why and when its episode ended:

  python train_test.py                         # 10 robots, random actions
  python train_test.py --actions zero          # hold the default pose
  python train_test.py --num-robots 5 --seed 7 --task rough
  python train_test.py --viewer False          # print only

No policy runs here. Random actions stand in for an untrained policy and
exercise the terminations. With zero actions the PD controllers hold the
default pose and sag under the robot's weight (a trained policy learns to
compensate), so many robots end up sitting on their calves within a second;
that alone is not a bug. A termination before landing (~0.1 s) is.
"""

import os
import sys
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import mjlab  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

from assets.morph_quadruped.constants import (  # noqa: E402
  INIT_JOINT_POS,
  NOMINAL_FOOT_RADIUS,
)
from generation.bank import BANK_FILE, MorphologyGenCfg, generate_bank  # noqa: E402
from tasks.velocity.config.morph_quadruped import (  # noqa: E402
  morph_quadruped_flat_env_cfg,
  morph_quadruped_rough_env_cfg,
  set_morphologies,
)
from tasks.velocity.mdp import get_morph_scales  # noqa: E402


@dataclass(frozen=True)
class TestConfig:
  num_robots: int = 10
  seed: int | None = None
  """Base seed; robot i uses seed + i. None draws a fresh one."""
  task: Literal["flat", "rough"] = "flat"
  actions: Literal["zero", "random"] = "random"
  """'random' sends uniform [-1, 1] actions (an untrained policy); 'zero' holds
  the default pose with no compensation for sag."""
  max_seconds: float = 10.0
  viewer: bool = True
  """Show each robot in a MuJoCo viewer (needs a display)."""
  hold_seconds: float = 2.0
  """How long the viewer holds the final frame after a termination."""
  slow_motion: float = 4.0
  """Viewer playback slowdown (1 = real time); episodes often end in < 1 s."""
  """How long the viewer holds the final frame after a termination."""
  device: str | None = None


def describe(env: ManagerBasedRlEnv, bank) -> list[str]:
  """Anatomy and per-robot thresholds of the robot in env 0."""
  s = get_morph_scales(env)
  s_L, s_m = s.length[0].item(), s.mass[0].item()
  s_v, s_t = s.speed[0].item(), s.time[0].item()
  r_foot = s.foot_radius[0].item()
  cfg = env.cfg
  rew, term, ev = cfg.rewards, cfg.terminations, cfg.events
  twist = env.command_manager.get_term("twist").cfg.ranges
  clearance = rew["foot_clearance"].params["target_height"]
  com = ev["base_com"].params["ranges"]
  push = ev["push_robot"].params["velocity_range"]
  robot = env.scene["robot"]
  hip, calf = robot.actuators[0], robot.actuators[2]
  return [
    f"anatomy   leg {bank['leg_length'][0]:.3f} m   mass {bank['total_mass'][0]:.1f} kg   "
    f"foot r {r_foot:.3f} m   spawn h {bank['root_height'][0]:.3f} m   "
    f"vmax {bank['max_joint_velocity'][0]:.1f} rad/s",
    f"scales    s_L {s_L:.2f}   s_m {s_m:.2f}   s_v {s_v:.2f}   s_t {s_t:.2f}",
    f"actuator  hip effort {hip.force_limit[0, 0]:.1f} Nm kp {hip.stiffness[0, 0]:.1f} "
    f"kd {hip.damping[0, 0]:.2f}   calf effort {calf.force_limit[0, 0]:.1f} Nm "
    f"kp {calf.stiffness[0, 0]:.1f} kd {calf.damping[0, 0]:.2f}",
    f"terminate illegal contact > "
    f"{term['illegal_contact'].params['force_threshold'] * s_m:.1f} N   "
    f"tilt > {np.degrees(term['fell_over'].params['limit_angle']):.0f} deg",
    f"commands  vx {twist.lin_vel_x[0] * s_v:+.2f}..{twist.lin_vel_x[1] * s_v:+.2f}  "
    f"vy {twist.lin_vel_y[0] * s_v:+.2f}..{twist.lin_vel_y[1] * s_v:+.2f} m/s   "
    f"track std {rew['track_linear_velocity'].params['std'] * s_v:.2f} m/s   "
    f"moving above {0.1 * s_v:.2f} m/s",
    f"yaw       {twist.ang_vel_z[0] / s_t:+.2f}..{twist.ang_vel_z[1] / s_t:+.2f} rad/s   "
    f"track std {rew['track_angular_velocity'].params['std'] / s_t:.2f} rad/s",
    f"gait      period {rew['foot_gait'].params['period'] * s_t:.2f} s   "
    f"foot clearance {r_foot + (clearance - NOMINAL_FOOT_RADIUS) * s_L:.3f} m",
    f"random    push xy +-{push['x'][1] * s_v:.2f} m/s   "
    f"base COM +-{com[0][1] * s_L:.3f} m",
  ]


def nonfoot_contacts(env: ManagerBasedRlEnv) -> str:
  """Robot geoms (not feet) currently pressing the ground, with force."""
  sensor = env.scene["nonfoot_ground_touch"]
  names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
  force = torch.norm(sensor.data.force_history[0], dim=-1).amax(dim=-1)  # [N]
  hits = [(names[i], force[i].item()) for i in torch.nonzero(force > 0).flatten()]
  return ", ".join(f"{n.split('/')[-1]} {f:.1f} N" for n, f in hits) or "none"


# Filled by `probe` (term params are copied, so it cannot be passed in).
PROBE_STATE: dict = {}


def probe(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Termination term that never fires; records the state the real terms see
  (env.step resets a terminated env before returning)."""
  state = PROBE_STATE
  state["height"] = env.scene["robot"].data.root_link_pos_w[0, 2].item()
  state["contacts"] = nonfoot_contacts(env)
  state["qpos"] = env.sim.data.qpos[0].cpu().numpy()
  state["qvel"] = env.sim.data.qvel[0].cpu().numpy()
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class RobotViewer:
  """Passive MuJoCo viewer showing env 0, tracking the robot's base.

  The template model is this robot's own MJCF (env_00000.xml), so its geometry
  is right; the state comes from the probe (the pre-reset state of each step).
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.model = env.sim.mj_model
    self.data = mujoco.MjData(self.model)
    self.handle = mujoco.viewer.launch_passive(
      self.model, self.data, show_left_ui=False, show_right_ui=False)
    self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    self.handle.cam.trackbodyid = self.model.body("robot/base_link").id
    self.handle.cam.distance = 1.5
    self.handle.cam.elevation = -15.0
    self.handle.cam.azimuth = 135.0
    self.handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    self.handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

  def is_running(self) -> bool:
    return self.handle.is_running()

  def show(self, state: dict) -> None:
    with self.handle.lock():
      self.data.qpos[:] = state["qpos"]
      self.data.qvel[:] = state["qvel"]
      mujoco.mj_forward(self.model, self.data)
    self.handle.sync()

  def hold(self, seconds: float) -> None:
    end = time.monotonic() + seconds
    while self.is_running() and time.monotonic() < end:
      self.handle.sync()
      time.sleep(0.02)

  def close(self) -> None:
    self.handle.close()


def run_robot(cfg: TestConfig, seed: int, out_dir: Path, device: str) -> str:
  env_cfg = (morph_quadruped_flat_env_cfg if cfg.task == "flat"
             else morph_quadruped_rough_env_cfg)()
  env_cfg.scene.num_envs = 1
  env_cfg.seed = seed
  generate_bank(1, out_dir, MorphologyGenCfg(seed=seed, workers=1), INIT_JOINT_POS)
  set_morphologies(env_cfg, out_dir)
  env_cfg.terminations["probe"] = TerminationTermCfg(func=probe)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  state = PROBE_STATE
  env.reset()  # Also applies the command curriculum's first stage.
  bank = np.load(out_dir / BANK_FILE)
  for line in describe(env, bank):
    print(f"  {line}")

  action_dim = env.action_manager.total_action_dim
  max_steps = int(cfg.max_seconds / env.step_dt)
  min_height = float("inf")
  result = f"survived {cfg.max_seconds:.1f} s"
  viewer = RobotViewer(env) if cfg.viewer else None
  start = time.monotonic()
  for step in range(max_steps):
    if viewer is not None and not viewer.is_running():
      result = f"skipped (viewer closed) at {step * env.step_dt:.2f} s"
      break
    if cfg.actions == "zero":
      action = torch.zeros(1, action_dim, device=device)
    else:
      action = 2 * torch.rand(1, action_dim, device=device) - 1
    _, _, terminated, truncated, _ = env.step(action)
    min_height = min(min_height, state["height"])
    if viewer is not None:
      viewer.show(state)
      frame_end = start + (step + 1) * env.step_dt * cfg.slow_motion
      time.sleep(max(0.0, frame_end - time.monotonic()))
    if terminated[0] or truncated[0]:
      fired = [name for name in env.termination_manager.active_terms
               if env.termination_manager.get_term(name)[0]]
      result = f"{'+'.join(fired)} at {(step + 1) * env.step_dt:.2f} s"
      if "illegal_contact" in fired:
        result += f" ({state['contacts']})"
      break
  print(f"  result    {result}   (base height at end {state['height']:.3f} m, "
        f"min {min_height:.3f} m)")
  if viewer is not None:
    viewer.hold(cfg.hold_seconds)
    viewer.close()
  env.close()
  return result


def main():
  cfg = tyro.cli(TestConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    print("[WARN] No display found; running without the viewer.")
    cfg = replace(cfg, viewer=False)
  base_seed = cfg.seed if cfg.seed is not None else secrets.randbits(31)
  root = Path(tempfile.mkdtemp(prefix="train_test_"))
  print(f"[INFO] {cfg.num_robots} robots, base seed {base_seed}, "
        f"{cfg.actions} actions")

  results = []
  try:
    for i in range(cfg.num_robots):
      seed = base_seed + i
      print(f"\n=== robot {i + 1}/{cfg.num_robots} (seed {seed}) ===")
      robot_dir = root / f"robot_{i:02d}"
      results.append((seed, run_robot(cfg, seed, robot_dir, device)))
      shutil.rmtree(robot_dir, ignore_errors=True)
  finally:
    shutil.rmtree(root, ignore_errors=True)

  print("\n=== summary ===")
  for seed, result in results:
    print(f"  seed {seed}: {result}")


if __name__ == "__main__":
  main()
