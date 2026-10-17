"""Make a quadruped walk with the trained OnePolicy (inference only, no training).

  python onboard.py                                      # menu: existing / random / yours
  python onboard.py --robot spot                         # an existing robot (../quadrupeds)
  python onboard.py --random [--seed 42]                 # a randomly generated robot
  python onboard.py path/to/robot.xml                    # your own MJCF, interactive
  python onboard.py path/to/robot.xml --policy <model_N.pt> --name my_dog
  python onboard.py path/to/robot.xml --yes              # no prompts (defaults)

Existing and random robots go straight to steps 6-7. Your own MJCF:
  1. Load the MJCF and list every joint with the leg / role it looks like.
  2. Fix it (generation/mjcf_fix.py): writes <file>.temp.xml next to it with
     every fix marked by a "OnePolicy:" comment (re-zeroed legs, round feet,
     extra joints removed, standing keyframe, torque clamps, damping) and
     lists what to change in your own MJCF / robot. Then check the fixed copy
     against the adapter convention (generation/adapt.py).
  3. Until it passes: the user edits the XML (original backed up as
     <file>.orig), presses Enter, and it is checked again.
  4. Ask what can't be inferred: which way is forward, whether the rear knees
     bend forward (ANYmal-style "><").
  5. Install into ../quadrupeds/<name>/ (XML + every mesh/texture it uses +
     play.json), so `play.py --robot <name>` finds it.
  6. Headless check in simulation: the policy standing (3 s, zero command),
     then walking forward (5 s), with the torques it uses vs. your motors.
  7. Open the viewer with the policy (play.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from assets.morph_quadruped.constants import INIT_JOINT_POS  # noqa: E402
from generation.adapt import (  # noqa: E402
  ASSET_DIR,
  ConventionError,
  MIN_AXIS_ALIGNMENT,
  _collect_assets,
)
from generation.catalog import QUADRUPEDS_DIR, available_robots, find_robot, load_robot  # noqa: E402
from generation.knees import X_KNEE_TAG, is_x_knee  # noqa: E402
from generation.mjcf_fix import FixError  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parent
ROLES = ("abduction", "hip", "knee")
FORWARDS = ("auto", "+x", "-x", "+y", "-y")

# ConventionError text -> what the user should change in the XML.
HINTS = (
  ("<include>", "Pass the robot file itself, not a scene.xml that <include>s it."),
  ("one free joint (floating base)", "Exactly one body (the torso) needs <freejoint/> as its first child."),
  ("12 hinge", "Each leg needs exactly 3 hinge joints and nothing else may move: delete "
               "the <joint> of extra parts (arm, tail, head, toes) so they become rigid, "
               "or remove those bodies."),
  ("4 legs", "Each leg must be one chain hanging off the torso body: "
             "abduction body > thigh body > calf body, one hinge joint in each."),
  ("not below the floating base", "Every leg body must be inside the torso body."),
  ("Two legs attach", "Legs must attach at 4 different corners of the torso; if the robot "
                      "is rotated, try another forward axis."),
  ("no collision geom below its knee", "Give each foot a collision geom, e.g. "
                                       "<geom type=\"sphere\" size=\"0.02\" pos=\"<foot tip in calf frame>\"/> in the "
                                       "calf body (contype/conaffinity must not both be 0)."),
  ("axis", "Joint order per leg must be abduction (axis along the torso's long axis), "
           "hip, knee (both along the sideways axis), each within ~25 degrees."),
  ("same parent", "All 4 legs must be direct children of the same body (the torso)."),
  ("leg(s)", "Each leg must be a chain of 3 hinge joints (abduction > hip > knee bodies) "
             "inside the torso body, the one with the <freejoint/>."),
  ("free joints", "Only the torso may have a <freejoint/>."),
  ("top-level body", "Put the whole robot inside one body (the torso) under <worldbody>."),
  ("one per corner", "Legs must attach at 4 different corners of the torso; try another "
                     "forward axis."),
)


# ---------------------------------------------------------------- prompts

def ask(question: str, default: str, auto: bool) -> str:
  if auto:
    print(f"{question} [{default}] -> {default}")
    return default
  answer = input(f"{question} [{default}]: ").strip()
  return answer or default


def ask_yes(question: str, default: bool, auto: bool) -> bool:
  answer = ask(question + " (y/n)", "y" if default else "n", auto)
  return answer.lower().startswith("y")


# ------------------------------------------------------- 1. joint table

def _base_body(model) -> int | None:
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  return int(model.jnt_bodyid[free[0]]) if len(free) == 1 else None


def _long_axis(model, data, base: int) -> str:
  """'+x' or '+y': the axis the abduction joints turn about (as the adapter does)."""
  rot = data.xmat[base].reshape(3, 3)
  axes = []
  for j in range(model.njnt):
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    b = int(model.body_parentid[model.jnt_bodyid[j]])
    while b not in (base, 0) and model.body_jntnum[b] == 0:
      b = int(model.body_parentid[b])
    if b == base:
      axes.append(np.abs(rot.T @ data.xaxis[j]))
  if not axes:
    return "+x"
  mean = np.mean(axes, axis=0)
  return "+x" if mean[0] >= mean[1] else "+y"


def joint_rows(xml_path: Path, forward: str) -> dict:
  """Every joint with the leg / role it looks like and whether that fits."""
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  base = _base_body(model)
  if base is None:
    rot = np.eye(3)
    origin = np.zeros(3)
  else:
    rot = data.xmat[base].reshape(3, 3)
    origin = data.xpos[base]
    if forward == "auto":
      forward = _long_axis(model, data, base)
  # Front / left directions in the base frame for the chosen forward axis.
  front = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0]}[forward]
  front = np.array(front, float)
  left = np.cross([0, 0, 1], front)
  long_vec, side_vec = np.abs(front), np.abs(left)

  # Leg of each hinge = the base child it hangs from; role = order in that leg.
  def leg_top(b: int) -> int | None:
    while base is not None and b not in (base, 0):
      parent = int(model.body_parentid[b])
      if parent == base:
        return b
      b = parent
    return None

  legs: dict[int, list[int]] = {}
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
      top = leg_top(int(model.jnt_bodyid[j]))
      if top is not None:
        legs.setdefault(top, []).append(j)

  def depth(b: int) -> int:
    n = 0
    while b != 0:
      b, n = int(model.body_parentid[b]), n + 1
    return n

  guess = {}
  for top, joints in legs.items():
    joints.sort(key=lambda j: (depth(int(model.jnt_bodyid[j])), j))
    anchor = rot.T @ (data.xanchor[joints[0]] - origin)
    label = ("F" if anchor @ front > 0 else "R") + ("L" if anchor @ left > 0 else "R")
    for i, j in enumerate(joints):
      guess[j] = (label, ROLES[i] if i < 3 else "EXTRA")

  type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
  rows = []
  for j in range(model.njnt):
    name = model.joint(j).name or f"<unnamed {j}>"
    jtype = int(model.jnt_type[j])
    body = model.body(int(model.jnt_bodyid[j])).name or f"<body {model.jnt_bodyid[j]}>"
    axis_s, range_s, note = "", "", ""
    if jtype in (2, 3):
      axis = rot.T @ data.xaxis[j]
      axis_s = "(" + ", ".join(f"{a:+.2f}" for a in axis) + ")"
      if model.jnt_limited[j]:
        lo, hi = model.jnt_range[j]
        range_s = f"[{lo:+.2f}, {hi:+.2f}]"
      else:
        range_s = "unlimited"
    if jtype == 0:
      note = "floating base  OK" if j == next(
        k for k in range(model.njnt) if model.jnt_type[k] == 0) else "EXTRA free joint  X"
    elif jtype != 3:
      note = f"{type_names[jtype]} joint not allowed  X"
    elif j not in guess:
      note = "not in a leg below the torso  X"
    else:
      label, role = guess[j]
      if role == "EXTRA":
        note = f"{label} 4th+ joint in this leg  X"
      else:
        expected = long_vec if role == "abduction" else side_vec
        ok = abs(axis @ expected) >= MIN_AXIS_ALIGNMENT
        flip = " (axis flipped)" if ok and axis @ expected < 0 else ""
        note = f"{label} {role}{flip}  " + ("OK" if ok else "X axis should be along "
                                            + ("long" if role == "abduction" else "side") + " axis")
    rows.append({"index": j, "name": name, "type": type_names[jtype], "body": body,
                 "axis": axis_s, "range": range_s, "guess": note,
                 "ok": note.endswith("OK")})
  return {"file": xml_path.name, "joints": int(model.njnt), "bodies": int(model.nbody - 1),
          "mass": float(model.body_mass.sum()), "forward": forward, "legs": len(legs),
          "rows": rows}


def print_joint_table(xml_path: Path, forward: str) -> None:
  t = joint_rows(xml_path, forward)
  print(f"\n=== {t['file']}: {t['joints']} joints, {t['bodies']} bodies, "
        f"total mass {t['mass']:.2f} kg ===")
  print(f"{'#':>3}  {'joint':24} {'type':6} {'body':22} {'axis (base frame)':22} "
        f"{'range [rad]':18} guess")
  for r in t["rows"]:
    print(f"{r['index']:>3}  {r['name'][:24]:24} {r['type']:6} {r['body'][:22]:22} "
          f"{r['axis']:22} {r['range']:18} {r['guess']}")
  print(f"(forward axis used for leg labels: {t['forward']}; legs found: {t['legs']})")


# ---------------------------------------------------------- 2. checks

def keyframe_snippet(xml_path: Path) -> str:
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  qpos = model.qpos0.copy()
  names = []
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
      names.append(model.joint(j).name or f"joint{j}")
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  if free:
    adr = model.jnt_qposadr[free[0]]
    qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
  values = " ".join(f"{v:.3g}" for v in qpos)
  return (
    "  Add a standing pose just before </mujoco> and tune the joint angles until\n"
    "  the robot stands (look at it with: python -m mujoco.viewer --mjcf=<file>):\n"
    "    <keyframe>\n"
    f"      <!-- base x y z, quat w x y z, then: {' '.join(names)} -->\n"
    f"      <key name=\"home\" qpos=\"{values}\"/>\n"
    "    </keyframe>"
  )


def run_checks(xml_path: Path, forward: str, x_knee: bool | None):
  """Returns (errors, adapted robot, default pose, notes): fix (mjcf_fix.py,
  writes <file>.temp.xml), adapt, pose. x_knee None: infer it."""
  try:
    mujoco.MjModel.from_xml_path(str(xml_path))
  except Exception as e:  # noqa: BLE001
    msg = str(e)
    if "qpos size" in msg:
      hint = ("A <keyframe> has the wrong number of values (joints were added or "
              "removed): fix its qpos, or delete the <keyframe> block.")
    elif "file" in msg.lower() or "resource" in msg.lower():
      hint = "A mesh/texture path doesn't exist (upload the whole robot folder)."
    else:
      hint = "Fix the XML error above (MuJoCo's message names the element)."
    return [f"MuJoCo cannot load the file: {msg}\n  FIX: {hint}"], None, None, []
  try:
    loaded = load_robot(xml_path, forward, "auto", INIT_JOINT_POS, x_knee)
  except (ConventionError, FixError) as e:
    msg = str(e)
    hint = next((h for key, h in HINTS if key in msg), "See the convention in README.md.")
    return [f"{msg}\n  FIX: {hint}"], None, None, []
  except FileNotFoundError as e:
    return [f"A referenced file is missing: {e}"], None, None, []
  except ValueError as e:
    return [f"No usable standing pose: {e}\n{keyframe_snippet(xml_path)}"], None, None, []
  return [], loaded.robot, loaded.pose, loaded.fixed.summary().splitlines() + loaded.log


def check_until_valid(xml_path: Path, forward: str, x_knee: bool | None, auto: bool):
  backed_up = False
  while True:
    try:
      print_joint_table(xml_path, forward)
    except Exception as e:  # noqa: BLE001
      print(f"(joint table unavailable: {e})")
    errors, robot, pose, notes = run_checks(xml_path, forward, x_knee)
    if not errors:
      print("\nConvention check: PASS")
      return robot, pose, notes
    print("\nConvention check: FAIL")
    for e in errors:
      print(f"- {e}")
    if auto:
      sys.exit(1)
    if not backed_up:
      backup = xml_path.with_name(xml_path.name + ".orig")
      if not backup.exists():
        shutil.copy2(xml_path, backup)
        print(f"(original backed up to {backup})")
      backed_up = True
    answer = input(f"\nEdit {xml_path} and save, then press Enter to re-check "
                   "(q to quit): ").strip().lower()
    if answer == "q":
      sys.exit(1)


# ---------------------------------------------------------- 5. install

def install(xml_path: Path, name: str, x_knee: bool, forward: str, overwrite: bool,
            auto: bool) -> Path:
  stem = name + (f"_{X_KNEE_TAG}" if x_knee and not is_x_knee(name) else "")
  target = QUADRUPEDS_DIR / name
  for other_stem, other in available_robots().items():
    if other_stem == stem and other.parent != target:
      raise SystemExit(f"A robot named {stem} already exists: {other}. Pick another --name.")
  if target.exists():
    if not (overwrite or ask_yes(f"{target} exists. Replace it?", False, auto)):
      raise SystemExit("Not installed.")
    shutil.rmtree(target)

  # Same asset resolution as the adapter: every file goes to assets/<key>.
  tree = ET.parse(xml_path)
  root = tree.getroot()
  assets = _collect_assets(root, str(xml_path.parent))
  if root.find(".//hfield[@file]") is not None:
    print("[WARN] hfield files are copied but not re-pointed; check them.")
  target.mkdir(parents=True)
  for key, data in assets.items():
    out = target / ASSET_DIR / key
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
  installed = target / f"{stem}.xml"
  ET.indent(tree)
  tree.write(installed, encoding="unicode")
  shutil.copy2(xml_path, target / f"{stem}.xml.orig")  # Not *.xml: not listed as a robot.
  if forward != "auto":
    (target / "play.json").write_text(json.dumps({"forward": forward}) + "\n")

  mujoco.MjModel.from_xml_path(str(installed))  # Proves no path points back.
  print(f"\nInstalled {installed} ({len(assets)} asset files"
        + (f", play.json forward {forward}" if forward != "auto" else "") + ")")
  return installed


# ------------------------------------------------- 6. headless sim check

def sim_check(play_cfg, policy: Path | None, walk_command: float) -> bool:
  """The policy standing (3 s, zero command), then walking forward (5 s).
  True if both look fine. Also compares the torques the policy used with the
  robot's own motor limits (from its MJCF), which the sim doesn't enforce."""
  import tempfile

  import torch

  import play
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tasks.velocity.config.morph_quadruped import (
    drop_body_contact_if_unmatched,
    morph_quadruped_flat_env_cfg,
    morph_quadruped_ppo_runner_cfg,
    set_morphologies,
  )
  from tasks.velocity.mdp.rewards import _joint_order_torques
  from tasks.velocity.rl import VelocityOnPolicyRunner
  from dataclasses import asdict

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  robot_dir = Path(tempfile.mkdtemp(prefix="onboard_robot_"))
  try:
    pose, _, loaded = play.prepare_robot(play_cfg, robot_dir)
    env_cfg = morph_quadruped_flat_env_cfg(play=True)
    agent_cfg = morph_quadruped_ppo_runner_cfg()
    env_cfg.scene.num_envs = 1
    set_morphologies(env_cfg, robot_dir, pose)
    drop_body_contact_if_unmatched(env_cfg, robot_dir / "env_00000.xml")
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    command = [(0.0, 0.0, 0.0)]
    raw.command_manager.get_term("twist").gamepad = lambda: command[0]
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    robot = raw.scene["robot"]
    tm = raw.termination_manager
    steps_per_s = round(1.0 / raw.step_dt)
    speed = float(raw.morph_scales.speed[0])
    stand_height = float(raw.morph_scales.base_above_feet[0])

    def run(act, seconds: float, label: str):
      """(survived, forward distance [m], min base height [m], peak |torque| per joint)."""
      obs, _ = env.reset()
      start = robot.data.root_link_pos_w[0].clone()
      heading = robot.data.root_link_quat_w[0].clone()
      low = float(start[2])
      peak = torch.zeros_like(robot.data.joint_pos[0])
      for i in range(int(seconds * steps_per_s)):
        with torch.no_grad():
          obs, _, dones, _ = env.step(act(obs))
        if dones[0]:
          why = [t for t in tm.active_terms if bool(tm.get_term(t)[0])]
          print(f"  {label}: episode ended at {i / steps_per_s:.2f} s ({', '.join(why)})")
          return False, None, None, None
        if i > steps_per_s // 2:  # Skip the landing after the spawn.
          low = min(low, float(robot.data.root_link_pos_w[0, 2]))
          peak = torch.maximum(peak, _joint_order_torques(robot)[0].abs())
      end = robot.data.root_link_pos_w[0]
      # Forward axis of the base at the start, in the ground plane.
      fwd = torch.zeros(3, device=end.device)
      w, x, y, z = heading.tolist()
      fwd[0], fwd[1] = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
      fwd = fwd / torch.clamp(torch.norm(fwd), min=1e-6)
      return True, float((end - start) @ fwd), low, peak.cpu().numpy()

    if policy is None:
      print("  (no policy checkpoint found: skipping the simulation checks)")
      return True
    runner = VelocityOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(str(policy), load_cfg={"actor": True}, strict=True, map_location=device)
    act = runner.get_inference_policy(device=device)

    print("\n=== Headless check 1: policy, zero command, 3 s (should stand still) ===")
    ok, moved, low, _ = run(act, 3.0, "standing")
    if not ok:
      print("  -> Can't even stand. Look at the fixed copy's keyframe in the viewer "
            "(python -m mujoco.viewer --mjcf=<file>.temp.xml): feet on the ground, "
            "knees the right way?")
      return False
    print(f"  stood 3 s: drifted {moved:+.2f} m, lowest base height {low:.3f} m "
          f"(standing pose {stand_height:.3f} m)")
    if low < 0.6 * stand_height:
      print("  -> Sagged far below the standing pose: the robot is too heavy for its "
            "legs, or the knees bend the wrong way (--xknee).")
      return False

    command[0] = (walk_command, 0.0, 0.0)
    expected = walk_command * speed * 5.0
    print(f"\n=== Headless check 2: policy, {walk_command} m/s forward (Go2 units = "
          f"{walk_command * speed:.2f} m/s for this robot), 5 s ===")
    ok, moved, low, peak = run(act, 5.0, "walking")
    if not ok:
      print("  -> Fell while walking. The policy may not be trained far enough, or the "
            "robot is outside the training sizes.")
      return False
    print(f"  moved {moved:+.2f} m forward (ideal {expected:.2f} m), lowest base height "
          f"{low:.3f} m")
    if loaded is not None:
      _report_torques(loaded.fixed.motors, peak, list(robot.joint_names))
    if moved < 0.2 * expected:
      print("  -> Barely moved or went backwards. If it walked backwards, the forward "
            "axis is wrong: re-run with the opposite --forward.")
      return False
    return True
  finally:
    shutil.rmtree(robot_dir, ignore_errors=True)


def _report_torques(motors, peak, joint_names: list[str]) -> None:
  """Peak torque per joint role while walking vs the MJCF's own motor limit."""
  print("  Peak torque the policy used vs your motors (the sim gives it Go2's motors "
        "scaled to this robot):")
  roles = {"abduction": "hip", "hip": "thigh", "knee": "calf"}
  for role, train_role in roles.items():
    used = max(float(peak[i]) for i, n in enumerate(joint_names) if n.endswith(f"_{train_role}_joint"))
    rows = [m for m in motors if m.role == role]
    owns = [m.own_peak for m in rows if m.own_peak is not None]
    line = f"    {role:9} used {used:6.1f} Nm, trained with up to {rows[0].expected_peak:6.1f} Nm"
    if owns:
      have = min(owns)
      line += f", yours {have:6.1f} Nm"
      if used > have:
        line += f"  <- {used / have:.1f}x over: the real robot can't do this gait"
    print(line)


def latest_policy() -> Path | None:
  runs = TRAINING_DIR / "logs" / "rsl_rl" / "morph_quadruped_velocity"
  ckpts = sorted(runs.glob("*/model_*.pt"),
                 key=lambda p: (p.parent.name, int(re.findall(r"\d+", p.stem)[-1])))
  return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------- main

def choose_source(args, auto: bool) -> None:
  """Fill args.robot / args.random / args.xml from a menu if none was given."""
  if args.xml or args.robot or args.random:
    return
  if auto:
    raise SystemExit("With --yes, pass an MJCF path, --robot NAME or --random.")
  print("Which robot?\n  1) an existing robot (Spot, ANYmal, Barkour, Go2, ...)\n"
        "  2) a randomly generated robot\n  3) your own MJCF file")
  choice = ask("Pick 1/2/3", "1", auto)
  if choice == "2":
    args.random = True
  elif choice == "3":
    args.xml = Path(ask("Path to your robot's .xml", "", auto))
  else:
    robots = list(available_robots())
    for i, name in enumerate(robots, 1):
      print(f"  {i:>2}) {name}")
    pick = ask("Number or name", "1", auto)
    args.robot = robots[int(pick) - 1] if pick.isdigit() else pick


def run_and_play(play_args: list[str], play_cfg, policy: Path | None, args, auto: bool) -> None:
  """Steps 6-7: headless sim check, then the viewer (play.py)."""
  if policy is None:
    print("[WARN] No policy checkpoint found (logs/ is empty): the robot will get zero "
          "actions. Pass --policy path/to/model_N.pt.")
  if not args.no_sim:
    ok = sim_check(play_cfg, policy, args.command)
    print("\nSim check:", "PASS" if ok else "FAIL (see above)")
    if not ok and not ask_yes("Open the viewer anyway?", False, auto):
      raise SystemExit(1)
  cmd = [sys.executable, "play.py", *play_args]
  cmd += ["--policy", str(policy)] if policy is not None else ["--agent", "zero"]
  print(f"\nPlay it any time with:\n  cd {TRAINING_DIR} && python {' '.join(cmd[1:])}")
  if not args.no_viewer and ask_yes("Open the viewer with the policy now?", True, auto):
    subprocess.run(cmd, cwd=TRAINING_DIR)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("xml", type=Path, nargs="?",
                      help="your robot MJCF (the robot file, not a scene)")
  parser.add_argument("--robot", help="existing robot in ../quadrupeds (spot, anymal_c, ...)")
  parser.add_argument("--random", action="store_true", help="a randomly generated robot")
  parser.add_argument("--seed", type=int, help="seed of the --random robot")
  parser.add_argument("--name", help="robot name in ../quadrupeds (default: file name)")
  parser.add_argument("--policy", type=Path, help="checkpoint (default: newest model_*.pt)")
  parser.add_argument("--forward", choices=FORWARDS, help="base axis the robot faces")
  parser.add_argument("--xknee", choices=("auto", "yes", "no"), default="auto",
                      help="rear knees bend forward (ANYmal-style ><)")
  parser.add_argument("--command", type=float, default=0.5,
                      help="forward speed for the walking check [Go2 m/s]")
  parser.add_argument("--overwrite", action="store_true", help="replace an installed robot")
  parser.add_argument("--yes", action="store_true", help="no prompts, take every default")
  parser.add_argument("--no-sim", action="store_true", help="skip the headless sim check")
  parser.add_argument("--no-viewer", action="store_true", help="don't open the viewer")
  args = parser.parse_args()
  auto = args.yes
  import play

  policy = args.policy.resolve() if args.policy else latest_policy()
  if policy is not None and not policy.is_file():
    raise SystemExit(f"Policy not found: {policy}")
  print(f"Policy: {policy}")
  choose_source(args, auto)

  if args.random:
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    print(f"Random robot, seed {seed}")
    run_and_play(["--random", "True", "--seed", str(seed)],
                 play.PlayConfig(random=True, seed=seed), policy, args, auto)
    return

  if args.robot:
    xml_path = find_robot(args.robot)
    forward = json.loads((xml_path.parent / "play.json").read_text()).get("forward", "auto") \
      if (xml_path.parent / "play.json").is_file() else "auto"
    print_joint_table(xml_path, forward)
    errors, _, _, notes = run_checks(xml_path, forward, None)
    if errors:
      print("\nConvention check: FAIL")
      for e in errors:
        print(f"- {e}")
      raise SystemExit("Fix the file (or use it as your own MJCF: python onboard.py "
                       f"{xml_path}).")
    print("\nConvention check: PASS")
    print("\n".join(f"  {line}" for line in notes))
    run_and_play(["--robot", xml_path.stem], play.PlayConfig(robot=xml_path.stem),
                 policy, args, auto)
    return

  xml_path = args.xml.expanduser().resolve()
  if not xml_path.is_file():
    raise SystemExit(f"Not a file: {xml_path}")

  # 1-3: structure.
  forward = args.forward or "auto"
  x_knee = None if args.xknee == "auto" else args.xknee == "yes"  # None: infer it.
  _, pose, _ = check_until_valid(xml_path, forward, x_knee, auto)

  # 4: what can't be inferred, then re-check with the answers.
  while True:
    if args.forward is None:
      forward = ask("Which way does the robot face? auto/+x/-x/+y/-y "
                    "(auto picks the long axis, front/back is a guess)", forward, auto)
      if forward not in FORWARDS:
        print(f"Pick one of {FORWARDS}.")
        continue
    if args.xknee == "auto":
      inferred = pose["RR_calf_joint"] > 0  # The fixer's guess (knee ranges, file name).
      x_knee = ask_yes("Do the REAR knees bend forward, like ANYmal (><)?", inferred, auto)
    robot, pose, notes = check_until_valid(xml_path, forward, x_knee, auto)
    print("Adapter decisions:")
    for line in robot.report + notes:
      print(f"  {line}")
    print("Standing pose (training joint names, rad):")
    for joint, angle in pose.items():
      print(f"  {joint:18} {angle:+.3f}")
    if not auto and ask_yes("Open the raw model in the MuJoCo viewer to look at it?", False, auto):
      subprocess.run([sys.executable, "-m", "mujoco.viewer", f"--mjcf={xml_path}"])
    if ask_yes("Is this right (forward axis, knees, pose)?", True, auto):
      break

  # 5: install.
  default_name = re.sub(r"[^a-z0-9_]+", "_", xml_path.stem.lower()).strip("_")
  default_name = default_name.replace(f"_{X_KNEE_TAG}", "")
  name = args.name or ask("Name for ../quadrupeds/<name>", default_name, auto)
  installed = install(xml_path, name, x_knee, forward, args.overwrite, auto)
  stem = installed.stem
  try:
    resolved = find_robot(name)
    robot_arg = name if resolved == installed else stem
  except ValueError:
    robot_arg = stem
  errors, *_ = run_checks(installed, forward, x_knee)
  if errors:
    raise SystemExit(f"Installed copy fails the check: {errors}")

  # 6-7: headless sim, then the viewer.
  run_and_play(["--robot", robot_arg], play.PlayConfig(robot=stem), policy, args, auto)


if __name__ == "__main__":
  main()
