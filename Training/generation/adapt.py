"""Rewrite any 12-DoF quadruped MJCF into the training (Go2) convention.

The policy only depends on the robot's *structure*, so any MJCF robot works if
it follows the usual quadruped convention (all MuJoCo Menagerie quadrupeds do):

  * one floating base (a free joint),
  * four legs of three hinge joints each, ordered from the base outwards:
    abduction (axis ~ base x), hip (axis ~ base y), knee (axis ~ base y),
  * at least one collision geom at the end of each leg (the foot),
  * either zero joint angles = legs straight down (as on Go2 / ANYmal / Spot),
    or a standing <keyframe> (the training pose is defined from straight legs).

Names don't matter, and neither does which base axis points forward: CAD
exports often use +y, which is detected from the abduction axes and rotated to
+x (front vs. back is ambiguous from axes alone; pass `forward` to choose).
Legs are identified by where they attach to the base
(front/rear from x, left/right from y), and the robot is rewritten so that the
policy sees it exactly like a training robot:

  * bodies, joints, feet, sites and sensors get the Go2 names
    (see mjcf_builder.py),
  * collision geoms that reach the foot's tip (shins ending in the foot) are
    named {leg}_shin_N_collision; the illegal-contact termination skips them,
  * legs are reordered to the training joint order FR, FL, RR, RL,
  * joint axes pointing the "wrong" way (-x / -y) are flipped, along with their
    ranges, so that positive angles mean the same motion on every robot,
  * the XML's own actuators, sensors and keyframes are dropped (mjlab adds the
    training actuators and sensors), and asset files are bundled with it.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import mujoco
import numpy as np

from .bank import foot_point

TRAINING_LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_ROLES = ("hip", "thigh", "calf")  # Go2 names for abduction, hip, knee.
EXPECTED_AXES = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 np.array([0.0, 1.0, 0.0]))
MIN_AXIS_ALIGNMENT = 0.9  # |cos| between a joint axis and its expected axis.
# Collision geoms below the knee that reach within this fraction of the shin
# length of the foot's tip are part of the foot ("shin" geoms): they touch the
# ground whenever the foot does (A1's calf capsule, Barkour's lower-leg mesh),
# so the illegal-contact termination ignores them. Go2's lowest calf geom stops
# 0.23 shin lengths short of its foot tip; Go1, Spot and ANYmal shins ~0.1.
SHIN_TIP_FRACTION = 0.15

# Attributes that refer to other elements by name, per element kind.
_REFERENCE_ATTRS = {
  "body": ("body", "body1", "body2", "target"),
  "joint": ("joint", "joint1", "joint2", "jointinparent"),
  "geom": ("geom", "geom1", "geom2"),
  "site": ("site", "site1", "site2"),
}
_KIND_TAGS = {
  "body": ("body",),
  "joint": ("joint", "freejoint"),
  "geom": ("geom",),
  "site": ("site",),
}


class ConventionError(ValueError):
  """The robot does not follow the quadruped convention described above."""


@dataclass
class AdaptedRobot:
  xml: str
  """MJCF in the training convention; its files live in ASSET_DIR next to it."""
  assets: dict[str, bytes]
  """Asset files, keyed by their path relative to ASSET_DIR."""
  joint_signs: dict[str, float]
  """Training joint name -> +1, or -1 if its axis was flipped."""
  joint_ranges: dict[str, tuple[float, float] | None]
  """Training joint name -> range after flipping (None if unlimited)."""
  keyframe_joint_pos: dict[str, float] | None
  """The XML's first keyframe in training joint coordinates, if it has one."""
  zero_pose_straight: bool
  """Whether zero joint angles put every foot straight below its hip joint."""
  report: list[str] = field(default_factory=list)


def _body_elements(root: ET.Element) -> list[ET.Element]:
  """Body elements in compiled body-id order (document order), id 1 first."""
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise ConventionError("MJCF has no <worldbody>.")
  return list(worldbody.iter("body"))


def _own_children(body: ET.Element, tags: tuple[str, ...]) -> list[ET.Element]:
  """Elements with `tags` belonging to `body` itself (also inside <frame>s)."""
  found = []
  for child in body:
    if child.tag in tags:
      found.append(child)
    elif child.tag == "frame":
      found.extend(_own_children(child, tags))
  return found


ASSET_DIR = "assets"


def _collect_assets(root: ET.Element, xml_dir: str) -> dict[str, bytes]:
  """Read every asset file and point the MJCF at a relative ASSET_DIR.

  mjlab passes asset bytes through `spec.assets` (keyed by meshdir/file), the
  way the Unitree robot configs do, so the adapted robot carries its files.
  """
  if root.find(".//include") is not None:
    raise ConventionError(
      "MJCF uses <include>; pass the robot file itself (e.g. go2.xml, not scene.xml).")
  compiler = root.find("compiler")
  if compiler is None:
    compiler = ET.Element("compiler")
    root.insert(0, compiler)
  assetdir = compiler.get("assetdir", "")
  dirs = {
    "mesh": compiler.get("meshdir", assetdir),
    "skin": compiler.get("meshdir", assetdir),
    "texture": compiler.get("texturedir", assetdir),
    "hfield": compiler.get("assetdir", ""),
  }
  assets = {}
  for tag, subdir in dirs.items():
    for el in root.iter(tag):
      file = el.get("file")
      if file is None:
        continue
      path = file if os.path.isabs(file) else os.path.join(xml_dir, subdir, file)
      key = os.path.basename(file) if os.path.isabs(file) else file
      with open(path, "rb") as f:
        assets[key] = f.read()
      el.set("file", key)
  compiler.attrib.pop("assetdir", None)
  compiler.set("meshdir", ASSET_DIR)
  compiler.set("texturedir", ASSET_DIR)
  return assets


def _angle_scale(root: ET.Element) -> float:
  """Factor from radians to the MJCF's angle unit."""
  angle = "degree"  # MJCF default.
  for compiler in root.findall("compiler"):
    angle = compiler.get("angle", angle)
  return 180.0 / math.pi if angle == "degree" else 1.0


def _fmt(values) -> str:
  return " ".join(f"{float(v):.8g}" for v in np.atleast_1d(values))


def _lowest_point(model, data, g) -> float:
  """World z of the lowest corner of geom g's bounding box."""
  center, half = model.geom_aabb[g].reshape(2, 3)
  corners = np.array([center + half * np.array(s) for s in product((-1, 1), repeat=3)])
  return float((data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())


def _reach(model, data, g, origin, direction) -> float:
  """How far geom g's bounding box extends from `origin` along `direction`."""
  center, half = model.geom_aabb[g].reshape(2, 3)
  corners = np.array([center + half * np.array(s) for s in product((-1, 1), repeat=3)])
  world = data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T
  return float(((world - origin) @ direction).max())


Forward = Literal["auto", "+x", "-x", "+y", "-y"]
_FORWARD_ANGLE = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}  # degrees about z


def _compile_tree(root: ET.Element, assets: dict[str, bytes]) -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(
    ET.tostring(root, encoding="unicode"),
    {f"{ASSET_DIR}/{k}": v for k, v in assets.items()},
  )


def _quat_mul(a, b) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
  return out


def _align_forward(root: ET.Element, assets: dict[str, bytes], forward: Forward,
                   report: list[str]) -> None:
  """Rotate the base's contents so that the robot's forward axis is base +x."""
  model = _compile_tree(root, assets)
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  if len(free) != 1:
    raise ConventionError(f"Expected one free joint (floating base), found {len(free)}.")
  base = int(model.jnt_bodyid[free[0]])

  if forward == "auto":
    # Abduction joints (first hinge below the base) turn about the long axis.
    first_hinges = []
    for j in range(model.njnt):
      if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
      b = int(model.body_parentid[model.jnt_bodyid[j]])
      while b != base and model.body_jntnum[b] == 0 and b != 0:
        b = int(model.body_parentid[b])
      if b == base:
        first_hinges.append(j)
    rot = data.xmat[base].reshape(3, 3)
    axes = np.abs([rot.T @ data.xaxis[j] for j in first_hinges]).mean(axis=0)
    forward = "+x" if axes[0] >= axes[1] else "+y"
    if forward == "+y":
      report.append("base long axis is y: assuming forward is +y "
                    "(pass forward -y if the robot walks backwards)")
  if forward == "+x":
    return

  # Points rotate by -angle about z, so the forward axis lands on +x.
  half = -math.radians(_FORWARD_ANGLE[forward]) / 2
  quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, quat)
  rot = rot.reshape(3, 3)

  base_el = _body_elements(root)[base - 1]
  frame = ET.Element("frame", quat=_fmt(quat))
  for child in list(base_el):
    if child.tag in ("joint", "freejoint", "inertial"):
      if child.tag == "inertial":
        child.attrib = {
          "pos": _fmt(rot @ model.body_ipos[base]),
          "quat": _fmt(_quat_mul(quat, model.body_iquat[base])),
          "mass": _fmt(model.body_mass[base]),
          "diaginertia": _fmt(model.body_inertia[base]),
        }
      continue
    base_el.remove(child)
    frame.append(child)
  base_el.append(frame)
  report.append(f"rotated base frame: forward {forward} -> +x")


def adapt_quadruped(xml_path: str, forward: Forward = "auto") -> AdaptedRobot:
  """Rewrite the quadruped at `xml_path` into the training convention."""
  xml_path = os.path.abspath(xml_path)
  report: list[str] = []

  tree = ET.parse(xml_path)
  root = tree.getroot()
  assets = _collect_assets(root, os.path.dirname(xml_path))
  _align_forward(root, assets, forward, report)
  model = _compile_tree(root, assets)
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  bodies = _body_elements(root)
  if len(bodies) != model.nbody - 1:
    raise ConventionError("Could not match <body> elements to the compiled model.")

  def body_el(b: int) -> ET.Element:
    return bodies[b - 1]

  def joint_el(j: int) -> ET.Element:
    b = model.jnt_bodyid[j]
    own = _own_children(body_el(b), ("joint", "freejoint"))
    return own[j - model.body_jntadr[b]]

  def geom_el(g: int) -> ET.Element:
    b = model.geom_bodyid[g]
    return _own_children(body_el(b), ("geom",))[g - model.body_geomadr[b]]

  # Floating base.
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  if len(free) != 1:
    raise ConventionError(f"Expected one free joint (floating base), found {len(free)}.")
  base = int(model.jnt_bodyid[free[0]])

  # Legs: chains of three hinge joints hanging off the base.
  hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  others = [j for j in range(model.njnt)
            if j not in hinges and j != free[0]]
  if others or len(hinges) != 12:
    raise ConventionError(
      f"Expected 12 hinge joints (+ free joint), found {len(hinges)} hinge "
      f"and {len(others)} other joints.")

  def ancestors(b: int) -> list[int]:
    chain = []
    while b != base:
      if b == 0:
        raise ConventionError("A leg joint is not below the floating base.")
      chain.append(b)
      b = int(model.body_parentid[b])
    return chain[::-1]  # base child first

  legs: dict[int, list[int]] = {}  # base child body -> its hinge joints
  for j in hinges:
    legs.setdefault(ancestors(int(model.jnt_bodyid[j]))[0], []).append(j)
  if len(legs) != 4 or any(len(js) != 3 for js in legs.values()):
    raise ConventionError(
      "Expected 4 legs of 3 hinge joints each, found legs with "
      f"{sorted(len(js) for js in legs.values())} joints.")

  base_rot = data.xmat[base].reshape(3, 3)
  leg_info = {}
  for top, joints in legs.items():
    joints.sort(key=lambda j: (len(ancestors(int(model.jnt_bodyid[j]))), j))
    anchor = base_rot.T @ (data.xanchor[joints[0]] - data.xpos[base])
    label = ("F" if anchor[0] > 0 else "R") + ("L" if anchor[1] > 0 else "R")
    if label in leg_info:
      raise ConventionError(f"Two legs attach at the {label} corner of the base.")
    leg_info[label] = (top, joints)

  # Feet: the lowest collision geom below each knee.
  is_collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)

  def subtree(b: int) -> set[int]:
    found = {b}
    for c in range(b + 1, model.nbody):  # Children always have larger ids.
      if model.body_parentid[c] in found:
        found.add(c)
    return found

  feet = {}
  for label, (_, joints) in leg_info.items():
    knee_bodies = subtree(int(model.jnt_bodyid[joints[2]]))
    candidates = [g for g in range(model.ngeom)
                  if is_collision[g] and model.geom_bodyid[g] in knee_bodies]
    if not candidates:
      raise ConventionError(f"Leg {label} has no collision geom below its knee (foot).")
    feet[label] = min(candidates, key=lambda g: _lowest_point(model, data, g))

  # Shin geoms: other collision geoms below the knee that reach (nearly) as far
  # along the shin as the foot. Measured along knee -> foot, so the pose the
  # XML happens to compile in doesn't matter.
  shins: dict[str, list[int]] = {}
  for label, (_, joints) in leg_info.items():
    knee = data.xanchor[joints[2]]
    shin = data.geom_xpos[feet[label]] - knee
    direction = shin / np.linalg.norm(shin)
    tip = _reach(model, data, feet[label], knee, direction)
    knee_bodies = subtree(int(model.jnt_bodyid[joints[2]]))
    shins[label] = [
      g for g in range(model.ngeom)
      if is_collision[g] and g != feet[label] and model.geom_bodyid[g] in knee_bodies
      and tip - _reach(model, data, g, knee, direction) < SHIN_TIP_FRACTION * tip]
    for g in shins[label]:
      report.append(f"{model.geom(g).name or f'geom {g}'} reaches the {label} foot tip: "
                    "shin geom (ignored by the illegal-contact termination)")
  shin_label = {g: label for label, gs in shins.items() for g in gs}

  # Zero pose convention: each foot straight below its hip (thigh) joint.
  zero_pose_straight = True
  for label, (_, joints) in leg_info.items():
    leg = base_rot.T @ (data.geom_xpos[feet[label]] - data.xanchor[joints[1]])
    if -leg[2] < 0.9 * np.linalg.norm(leg):
      zero_pose_straight = False
      report.append(f"zero pose of leg {label} is not straight down "
                    f"(foot at {np.round(leg, 3)} from the hip joint)")

  # Element lookups happen before any edit (edits change document order).
  base_el = body_el(base)
  free_el = joint_el(free[0])
  leg_els = {label: (body_el(top), [joint_el(j) for j in joints],
                     [body_el(int(model.jnt_bodyid[j])) for j in joints])
             for label, (top, joints) in leg_info.items()}
  foot_els = {label: (geom_el(g), body_el(int(model.geom_bodyid[g]))) for label, g in feet.items()}
  collision_els = [(geom_el(g), model.body(int(model.geom_bodyid[g])).name or f"body{model.geom_bodyid[g]}",
                    shin_label.get(g))
                   for g in range(model.ngeom)
                   if is_collision[g] and g not in feet.values()
                   and model.geom_bodyid[g] != 0]  # World geoms (floors) are removed.

  # Keyframe pose (before flipping), in training joint names.
  keyframe = None
  if model.nkey:
    keyframe = {}
    for label, (_, joints) in leg_info.items():
      for role, j in zip(JOINT_ROLES, joints):
        keyframe[f"{label}_{role}_joint"] = float(model.key_qpos[0][model.jnt_qposadr[j]])

  # --- Edits. ---------------------------------------------------------------

  def rename(kind: str, el: ET.Element, new: str) -> None:
    """Rename `el`, moving any other element with that name out of the way."""
    for tag in _KIND_TAGS[kind]:
      for other in root.iter(tag):
        if other is not el and other.get("name") == new:
          _set_name(kind, other, f"{new}_orig")
    _set_name(kind, el, new)

  def _set_name(kind: str, el: ET.Element, new: str) -> None:
    old = el.get("name")
    el.set("name", new)
    if old is None:
      return
    for other in root.iter():
      for attr in _REFERENCE_ATTRS[kind]:
        if other.get(attr) == old:
          other.set(attr, new)

  for tag in ("actuator", "sensor", "keyframe"):
    for section in root.findall(tag):
      root.remove(section)
  worldbody = root.find("worldbody")
  for child in list(worldbody):
    if child.tag == "geom":  # Floors etc.; the training scene has its own.
      worldbody.remove(child)

  rename("body", base_el, "base_link")
  rename("joint", free_el, "floating_base_joint")

  scale = _angle_scale(root)
  joint_signs, joint_ranges = {}, {}
  for label, (_, joints) in leg_info.items():
    _, joint_elements, joint_bodies = leg_els[label]
    for role, j, el, body, expected in zip(JOINT_ROLES, joints, joint_elements,
                                           joint_bodies, EXPECTED_AXES):
      name = f"{label}_{role}_joint"
      axis = base_rot.T @ data.xaxis[j]
      alignment = float(axis @ expected)
      if abs(alignment) < MIN_AXIS_ALIGNMENT:
        raise ConventionError(
          f"{model.joint(j).name}: axis {np.round(axis, 2)} (base frame) is not "
          f"along {expected}; joints must be ordered abduction, hip, knee.")
      sign = 1.0 if alignment > 0 else -1.0
      el.set("axis", _fmt(sign * model.jnt_axis[j]))
      if model.jnt_limited[j]:
        lo, hi = sorted(sign * model.jnt_range[j])
        el.set("range", _fmt((lo * scale, hi * scale)))
        el.set("limited", "true")
        joint_ranges[name] = (float(lo), float(hi))
      else:
        joint_ranges[name] = None
      joint_signs[name] = sign
      if sign < 0:
        report.append(f"flipped {model.joint(j).name} -> {name}")
      rename("joint", el, name)
      if len(_own_children(body, ("joint",))) == 1:
        rename("body", body, f"{label}_{role}")

  for label, (geom, foot_body) in foot_els.items():
    rename("geom", geom, f"{label}_foot_collision")
    for site in root.iter("site"):
      if site.get("name") == label:
        rename("site", site, f"{label}_orig")
    # At the foot centre (foot_point): a sphere's centre, or e.g. the tip of
    # a calf mesh that ends in the foot, in the foot body's frame.
    g = feet[label]
    b = int(model.geom_bodyid[g])
    point, _ = foot_point(model, data, g)
    pos = data.xmat[b].reshape(3, 3).T @ (point - data.xpos[b])
    ET.SubElement(foot_body, "site", name=label, pos=_fmt(pos), size="0.01",
                  group="5", rgba="1 0 0 1")
  for k, (geom, body_name, shin) in enumerate(collision_els):
    rename("geom", geom, f"{shin}_shin_{k}_collision" if shin else f"{body_name}_{k}_collision")

  if not any(s.get("name") == "imu" for s in _own_children(base_el, ("site",))):
    for site in root.iter("site"):
      if site.get("name") == "imu":
        rename("site", site, "imu_orig")
    ET.SubElement(base_el, "site", name="imu", pos="0 0 0", size="0.01", group="5")

  sensor = ET.SubElement(root, "sensor")
  ET.SubElement(sensor, "gyro", name="imu_ang_vel", site="imu")
  ET.SubElement(sensor, "velocimeter", name="imu_lin_vel", site="imu")
  ET.SubElement(sensor, "accelerometer", name="imu_lin_acc", site="imu")
  ET.SubElement(sensor, "subtreeangmom", name="root_angmom", body="base_link")

  # Reorder legs to the training joint order (FR, FL, RR, RL).
  tops = [leg_els[label][0] for label in TRAINING_LEG_ORDER]
  parents = {id(p): p for p in root.iter() for c in p if any(c is t for t in tops)}
  if len(parents) != 1:
    raise ConventionError("Legs must hang off the same parent element to be reordered.")
  parent = next(iter(parents.values()))
  children = list(parent)
  slots = sorted(i for i, c in enumerate(children) if any(c is t for t in tops))
  for slot, top in zip(slots, tops):
    children[slot] = top
  for c in list(parent):
    parent.remove(c)
  parent.extend(children)

  # Name every remaining element; per-env fields are matched by name.
  for kind in ("body", "geom", "site"):
    taken = {el.get("name") for el in worldbody.iter(kind)}
    count = 0
    for el in worldbody.iter(kind):
      if el.get("name") is None:
        while f"{kind}{count}" in taken:
          count += 1
        el.set("name", f"{kind}{count}")
        taken.add(f"{kind}{count}")

  if keyframe is not None:
    keyframe = {n: joint_signs[n] * v for n, v in keyframe.items()}

  ET.indent(root)
  return AdaptedRobot(
    xml=ET.tostring(root, encoding="unicode"),
    assets=assets,
    joint_signs=joint_signs,
    joint_ranges=joint_ranges,
    keyframe_joint_pos=keyframe,
    zero_pose_straight=zero_pose_straight,
    report=report,
  )


def fit_pose(pose: dict[str, float],
             joint_ranges: dict[str, tuple[float, float] | None]) -> tuple[dict[str, float], list[str]]:
  """Fit a joint pose into the ranges: mirror a value whose range is mirrored
  (e.g. knees that bend the other way), else clip it."""
  fitted, notes = {}, []
  for name, value in pose.items():
    rng = joint_ranges[name]
    if rng is None or rng[0] <= value <= rng[1]:
      fitted[name] = value
    elif rng[0] <= -value <= rng[1]:
      fitted[name] = -value
      notes.append(f"{name}: {value:+.2f} mirrored to {-value:+.2f}")
    else:
      fitted[name] = float(np.clip(value, *rng))
      notes.append(f"{name}: {value:+.2f} clipped to {fitted[name]:+.2f}")
  return fitted, notes
