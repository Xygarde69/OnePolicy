"""The real robots in ../quadrupeds: lookup by name, per-robot defaults, pose.

Shared by play.py (play one robot) and train.py --generalise (train on
proxies of some of them, see proxy.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .adapt import AdaptedRobot, adapt_quadruped, fit_pose

QUADRUPEDS_DIR = Path(__file__).resolve().parents[2] / "quadrupeds"


def available_robots(quadrupeds_dir: Path = QUADRUPEDS_DIR) -> dict[str, Path]:
  """Robot name (file stem) -> robot MJCF, skipping scenes and MJX variants."""
  return {
    p.stem: p
    for p in sorted(quadrupeds_dir.glob("*/*.xml"))
    if not p.stem.startswith("scene") and not p.stem.endswith("_mjx")
  }


def find_robot(name: str, quadrupeds_dir: Path = QUADRUPEDS_DIR) -> Path:
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


def robot_defaults(xml_path: Path) -> dict:
  """Optional play.json next to a robot's MJCF, for what 'auto' can't infer.

  Keys: "forward" (+x/-x/+y/-y), used when --forward is auto. E.g. a robot
  whose front legs sit at -x: {"forward": "-x"}.
  """
  path = xml_path.parent / "play.json"
  if not path.is_file():
    return {}
  defaults = json.loads(path.read_text())
  unknown = set(defaults) - {"forward"}
  if unknown:
    raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
  return defaults


def resolve_pose(pattern_pose: dict[str, float], names) -> dict[str, float]:
  """Expand a {regex: value} pose to {joint name: value}."""
  return {
    n: next(v for p, v in pattern_pose.items() if re.fullmatch(p, n)) for n in names
  }


def default_pose(robot: AdaptedRobot, xml_path: Path, init_pose: str,
                 train_pose: dict[str, float]) -> tuple[dict[str, float], list[str]]:
  """The robot's default joint pose and what was chosen, as log lines.

  init_pose: "keyframe" (the XML's first keyframe), "train" (`train_pose`
  fitted into the joint ranges) or "auto" (keyframe if there is one).
  """
  use_keyframe = init_pose == "keyframe" or (
    init_pose == "auto" and robot.keyframe_joint_pos is not None
  )
  if use_keyframe:
    if robot.keyframe_joint_pos is None:
      raise ValueError(f"{xml_path} has no keyframe; use --init-pose train.")
    return robot.keyframe_joint_pos, ["Default pose: the robot's keyframe"]
  log = []
  if not robot.zero_pose_straight:
    message = (
      f"{xml_path}: zero joint angles do not put the legs straight down, "
      "so the training pose does not transfer. Add a standing pose to the "
      "XML (<keyframe><key name=\"home\" qpos=\"...\"/></keyframe>)."
    )
    if init_pose == "auto":
      raise ValueError(message)
    log.append(f"WARN {message}")
  pose, notes = fit_pose(resolve_pose(train_pose, robot.joint_ranges), robot.joint_ranges)
  return pose, log + ["Default pose: training pose"] + [f"  {n}" for n in notes]


def real_robot_bank(name: str, copies: int, out_dir: Path,
                    train_pose: dict[str, float]) -> tuple[dict[str, float], list[str]]:
  """Bank of `copies` envs of real robot `name`'s own model, adapted to the
  training convention exactly as play.py does (train.py --generalise).

  Returns the robot's default pose and what was done, as log lines.
  """
  from .bank import single_robot_bank  # bank.py imports nothing from here.

  xml_path = find_robot(name)
  robot = adapt_quadruped(str(xml_path), robot_defaults(xml_path).get("forward", "auto"))
  pose, log = default_pose(robot, xml_path, "auto", train_pose)
  single_robot_bank(robot.xml, robot.assets, out_dir, pose, copies=copies, source=name)
  return pose, [line for line in log if not line.startswith("  ")]
