"""Make a quadruped MJCF usable by OnePolicy, and say what to change in it.

`fix_mjcf(path)` writes `<stem>.temp.xml` next to the robot's MJCF (so its
mesh paths still resolve) and returns what it did. The temp file keeps the
robot's own names and conventions; adapt.py then rewrites *it* into the
training convention. Every change is marked in the temp file with an
`<!-- OnePolicy: ... -->` comment saying what to change in your own MJCF, and
a header comment lists all changes and suggestions, so the temp file doubles
as a patch to read (diff it against your file).

Fixed automatically (each one is a suggestion for your MJCF too):

  * a scene that <include>s the robot: the robot file is used instead,
  * a keyframe whose qpos no longer fits the joints: dropped,
  * no floating base: a <freejoint/> is added to the body holding the legs,
  * extra joints (arm, head, tail, toes, ...): removed, the parts become rigid,
    with the actuators / sensors / tendons that use them,
  * a leg without a foot collision geom: a sphere is added at the leg's tip,
  * a zero pose with the legs not straight down (Barkour: legs along the body):
    the leg bodies are re-zeroed so that 0 = leg straight down, as on Go2 and
    every training robot. Joint ranges move with it. Without this the training
    pose doesn't transfer and the `knee_direction` observation (sign of the
    default knee angle) can claim the front knees bend forward, which no
    training robot does,
  * which way the knees bend, from the knee joint ranges (or the keyframe):
    rear knees forward = "><" (ANYmal), front knees forward = the robot faces
    the other way (front/back is ambiguous from the axes alone),
  * a standing keyframe "home" in the training pose (">>" or "><"),
  * joint `actuatorfrcrange`: removed. It clamps the training actuators (Go2's
    motors scaled to the robot) below what the policy was trained with, so the
    robot sags or falls in sim for a reason the policy has never seen
    (Byte01 v2: knees clamped at 36.5 Nm, policy expects ~85 Nm),
  * passive joint damping on the legs: set to the training robots' 0.1.

Only reported (physical; the sim can't fix the real robot): motor torque vs
what the policy expects for this size and mass, torque needed to stand,
joint friction, link mass distribution, size outside the training range.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .bank import foot_point
from .knees import is_x_knee

TEMP_SUFFIX = ".temp"
"""robot.xml -> robot.temp.xml (catalog.available_robots skips these)."""
TAG = "OnePolicy"

# Training robots (generation/mjcf_builder.py, assets/morph_quadruped/constants.py).
TRAINING_JOINT_DAMPING = 0.1
GO2_MASS = 15.206
GO2_LEG_LENGTH = 0.426
GO2_PEAK_TORQUE = {"abduction": 23.5, "hip": 23.5, "knee": 45.0}
GO2_CALF_MASS_FRACTION = (0.241352 + 0.04) / GO2_MASS
TRAINING_SIZE_RANGE = (0.24, 3.37)  # leg length / Go2's, p1-p99 of the generator
TRAINING_POSE = {"hip": 0.1, "thigh": 0.9, "calf": -1.8}  # constants.INIT_JOINT_POS
"""Training pose: abduction +-0.1 (outwards: + on the right legs), thigh,
knee. "><" robots negate the rear thigh and knee (knees.py)."""

ROLES = ("abduction", "hip", "knee")
LEG_LABELS = ("FR", "FL", "RR", "RL")
REZERO_TOLERANCE = 0.1  # [rad]: smaller zero-pose offsets (Spot's 4 deg) are left alone.
MOTOR_WARN_FRACTION = 0.65
"""The policy's knees reach ~0.63x the training peak torque walking at 0.5 m/s
(Byte01 v2, model_1200); motors below this fraction of it are reported."""
FOOT_RADIUS_FRACTION = 0.1
"""Added foot sphere radius / shin length (Go2: 0.022 / 0.213)."""
ONE_SIDED = 0.3  # [rad]: a knee range reaching less than this past 0 bends one way.
_FORWARD_ANGLE = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}
_OPPOSITE = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}


class FixError(ValueError):
  """The MJCF can't be made into a 12-DoF quadruped automatically."""


@dataclass
class MotorCheck:
  joint: str
  role: str
  own_peak: float | None
  """The MJCF's own torque limit [Nm] (None: no limit found)."""
  expected_peak: float
  """Go2's peak torque scaled by s_m * s_L: what the training actuators give."""
  stand: float
  """Torque [Nm] this joint needs to hold the robot in the training pose."""


@dataclass
class FixedMjcf:
  path: Path
  """The rewritten MJCF (<stem>.temp.xml), in the robot's own names."""
  source: Path
  """The robot file it was made from."""
  forward: str
  """Base axis the robot faces (+x/-x/+y/-y)."""
  x_knee: bool
  """Rear knees bend forward ("><")."""
  joint_offsets: dict[str, float]
  """Robot joint name -> zero-pose offset [rad] baked into the temp file:
  q_temp = q_yours - offset (to run the policy on the real robot)."""
  motors: list[MotorCheck]
  changes: list[str] = field(default_factory=list)
  suggestions: list[str] = field(default_factory=list)

  def summary(self) -> str:
    lines = [f"{self.path.name} (from {self.source.name}): forward {self.forward}, "
             f"knees {'><' if self.x_knee else '>>'}"]
    lines += ["Changed (marked '" + TAG + ":' in the temp file):"] + \
      [f"  - {c}" for c in self.changes or ["nothing"]]
    if self.suggestions:
      lines += ["Suggested changes to your MJCF / robot:"] + [f"  - {s}" for s in self.suggestions]
    return "\n".join(lines)


# ------------------------------------------------------------- XML helpers

def _parse(path: Path) -> ET.ElementTree:
  return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def _fmt(values) -> str:
  return " ".join(f"{float(v):.6g}" for v in np.atleast_1d(values))


def _free(model) -> int:
  return next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)


def _parents(root: ET.Element) -> dict[int, ET.Element]:
  return {id(c): p for p in root.iter() for c in p}


def _note(root: ET.Element, el: ET.Element, text: str) -> None:
  """Put an <!-- OnePolicy: text --> comment just before `el`."""
  parent = _parents(root)[id(el)]
  text = text.replace("--", "-")
  parent.insert(list(parent).index(el), ET.Comment(f" {TAG}: {text} "))


def _bodies(root: ET.Element) -> list[ET.Element]:
  """<body> elements in compiled body-id order (id 1 first)."""
  return list(root.find("worldbody").iter("body"))


def _own(body: ET.Element, tags: tuple[str, ...]) -> list[ET.Element]:
  """Children of `body` with `tags`, also inside <frame>s (compiled order)."""
  found = []
  for child in body:
    if child.tag in tags:
      found.append(child)
    elif child.tag == "frame":
      found.extend(_own(child, tags))
  return found


def _angle_scale(root: ET.Element) -> float:
  """Radians -> the MJCF's angle unit."""
  angle = "degree"
  for c in root.findall("compiler"):
    angle = c.get("angle", angle)
  return 180.0 / math.pi if angle == "degree" else 1.0


def _local_quat(el: ET.Element, root: ET.Element) -> np.ndarray:
  """A body's orientation from whichever attribute sets it."""
  scale = _angle_scale(root)
  quat = np.array([1.0, 0, 0, 0])
  if el.get("quat"):
    quat = np.array([float(v) for v in el.get("quat").split()])
  elif el.get("axisangle"):
    *axis, angle = (float(v) for v in el.get("axisangle").split())
    mujoco.mju_axisAngle2Quat(quat, np.array(axis) / np.linalg.norm(axis), angle / scale)
  elif el.get("euler"):
    seq = next((c.get("eulerseq") for c in root.findall("compiler") if c.get("eulerseq")), "xyz")
    euler = np.array([float(v) for v in el.get("euler").split()]) / scale
    mujoco.mju_euler2Quat(quat, euler, seq)
  elif el.get("xyaxes"):
    v = np.array([float(x) for x in el.get("xyaxes").split()])
    x = v[:3] / np.linalg.norm(v[:3])
    y = v[3:] - (v[3:] @ x) * x
    y /= np.linalg.norm(y)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, np.cross(x, y)]).ravel())
  elif el.get("zaxis"):
    mujoco.mju_quatZ2Vec(quat, np.array([float(x) for x in el.get("zaxis").split()]))
  return quat / np.linalg.norm(quat)


def _compile(root: ET.Element, near: Path) -> mujoco.MjModel:
  """Compile `root` from a scratch file next to `near` (relative paths resolve)."""
  scratch = near.with_name(f".{near.stem}.{os.getpid()}.scratch.xml")
  try:
    scratch.write_text(ET.tostring(root, encoding="unicode"))
    return mujoco.MjModel.from_xml_path(str(scratch))
  finally:
    scratch.unlink(missing_ok=True)


def _write(root: ET.Element, path: Path) -> None:
  tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  ET.indent(root)
  tmp.write_text(ET.tostring(root, encoding="unicode") + "\n")
  os.replace(tmp, path)


def temp_path(xml_path: Path) -> Path:
  return xml_path.with_name(xml_path.stem + TEMP_SUFFIX + ".xml")


# ------------------------------------------------------------ model helpers

def _subtree(model, b: int) -> list[int]:
  found = [b]
  for c in range(b + 1, model.nbody):
    if model.body_parentid[c] in found:
      found.append(c)
  return found


def _depth(model, b: int) -> int:
  n = 0
  while b:
    b, n = int(model.body_parentid[b]), n + 1
  return n


def _is_collision(model, g: int) -> bool:
  return bool(model.geom_contype[g] or model.geom_conaffinity[g])


def _geom_points(model, data, g: int) -> np.ndarray:
  """World points spanning geom g: mesh vertices, else its bounding box corners."""
  pos, rot = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
  if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
    m = model.geom_dataid[g]
    local = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
  else:
    center, half = model.geom_aabb[g].reshape(2, 3)
    local = np.array([center + half * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])
  return pos + local @ rot.T


def _base_frame(forward: str) -> np.ndarray:
  """Rotation from the MJCF's base frame to one with `forward` along +x."""
  a = -math.radians(_FORWARD_ANGLE[forward])
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class _Leg:
  top: int  # base child body
  joints: list[int]  # abduction, hip, knee
  foot: int | None  # collision geom id
  tip_body: int  # body holding the leg's tip
  tip_local: np.ndarray  # leg tip in tip_body's frame


def _find_legs(model, data, base: int) -> tuple[list[_Leg], list[int]]:
  """The 4 legs (3 hinges each) and every other non-free joint (extras)."""
  tops = [b for b in range(1, model.nbody) if model.body_parentid[b] == base]
  candidates = []
  for top in tops:
    bodies = _subtree(model, top)
    # Deepest body (lowest on ties) = the end of the leg chain.
    end = max(bodies, key=lambda b: (_depth(model, b), -data.xpos[b][2]))
    chain, b = [], end
    while b != base:
      chain.append(b)
      b = int(model.body_parentid[b])
    chain.reverse()
    hinges = [j for b in chain for j in range(model.body_jntadr[b],
                                              model.body_jntadr[b] + model.body_jntnum[b])
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(hinges) >= 3:
      candidates.append((top, hinges[:3], float(data.xpos[end][2])))
  if len(candidates) < 4:
    raise FixError(f"Found {len(candidates)} leg(s) (chains of >= 3 hinge joints hanging "
                   "off the floating base); a quadruped needs 4.")
  # Legs reach lowest (an arm or a head on top doesn't).
  candidates.sort(key=lambda c: c[2])
  legs = []
  for top, joints, _ in candidates[:4]:
    knee_body = int(model.jnt_bodyid[joints[2]])
    below = _subtree(model, knee_body)
    feet = [g for g in range(model.ngeom) if model.geom_bodyid[g] in below and _is_collision(model, g)]
    knee = data.xanchor[joints[2]]
    # The foot reaches farthest from the knee, whatever pose the legs are in.
    foot = max(feet, key=lambda g: np.linalg.norm(_geom_points(model, data, g) - knee,
                                                  axis=1).max()) if feet else None
    # The leg's tip: farthest point of any geom below the knee from the knee.
    best = (-1.0, knee_body, np.zeros(3))
    for g in range(model.ngeom):
      if model.geom_bodyid[g] not in below:
        continue
      pts = _geom_points(model, data, g)
      k = int(np.argmax(np.linalg.norm(pts - knee, axis=1)))
      dist = float(np.linalg.norm(pts[k] - knee))
      if dist > best[0]:
        b = int(model.geom_bodyid[g])
        best = (dist, b, data.xmat[b].reshape(3, 3).T @ (pts[k] - data.xpos[b]))
    legs.append(_Leg(top, joints, foot, best[1], best[2]))
  keep = {j for leg in legs for j in leg.joints}
  extras = [j for j in range(model.njnt)
            if j not in keep and model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  return legs, extras


def _foot_tip_local(model, data, leg: _Leg) -> tuple[int, np.ndarray]:
  """Foot centre (bank.foot_point convention for spheres / capsules, else the
  point farthest from the knee), in its body's frame."""
  g = leg.foot
  b = int(model.geom_bodyid[g])
  if model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            mujoco.mjtGeom.mjGEOM_CYLINDER):
    point, _ = foot_point(model, data, g)
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
      # The end cap away from the knee, whatever the pose.
      rot = data.geom_xmat[g].reshape(3, 3)
      ends = [data.geom_xpos[g] + s * model.geom_size[g][1] * rot[:, 2] for s in (-1, 1)]
      knee = data.xanchor[leg.joints[2]]
      point = max(ends, key=lambda p: np.linalg.norm(p - knee))
  else:
    pts = _geom_points(model, data, g)
    point = pts[np.argmax(np.linalg.norm(pts - data.xanchor[leg.joints[2]], axis=1))]
  return b, data.xmat[b].reshape(3, 3).T @ (point - data.xpos[b])


# --------------------------------------------------------------- the fixer

def _resolve_include(path: Path, changes: list[str]) -> Path:
  root = ET.parse(path).getroot()
  includes = [inc.get("file") for inc in root.iter("include") if inc.get("file")]
  if not includes:
    return path
  robots = [path.parent / f for f in includes
            if (path.parent / f).is_file()
            and ET.parse(path.parent / f).getroot().find(".//worldbody/body") is not None]
  if len(robots) != 1:
    raise FixError(f"{path.name} <include>s {includes}; pass the robot file itself.")
  changes.append(f"{path.name} is a scene: used the robot file it includes, {robots[0].name}")
  return robots[0]


def fix_mjcf(xml_path: str | Path, forward: str = "auto", x_knee: bool | None = None,
             own_motor_limits: bool = False, out_path: Path | None = None) -> FixedMjcf:
  """Write the fixed copy of the quadruped at `xml_path` (see module doc).

  forward: "auto" or the base axis the robot faces. x_knee: whether the rear
  knees bend forward, used only when the knee ranges / keyframe can't tell
  (None: from the file name, see knees.py). own_motor_limits: keep joint actuatorfrcrange (simulate the
  robot's own torque limits instead of the training actuators').
  """
  changes: list[str] = []
  suggestions: list[str] = []
  source = _resolve_include(Path(xml_path).resolve(), changes)
  out = out_path or temp_path(source)
  tree = _parse(source)
  root = tree.getroot()
  if root.find("worldbody") is None:
    raise FixError("The MJCF has no <worldbody>.")
  scale = _angle_scale(root)

  # 1. It must compile. A stale keyframe is the usual culprit after edits.
  try:
    model = _compile(root, out)
  except ValueError as e:
    if "keyframe" not in str(e) and "qpos" not in str(e) and "key" not in str(e).lower():
      raise FixError(f"MuJoCo cannot load {source.name}: {e}") from e
    for kf in root.findall("keyframe"):
      root.remove(kf)
    changes.append(f"dropped <keyframe>s that no longer fit the joints ({e})")
    model = _compile(root, out)
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  bodies = _bodies(root)

  def joint_el(j: int) -> ET.Element:
    b = int(model.jnt_bodyid[j])
    return _own(bodies[b - 1], ("joint", "freejoint"))[j - model.body_jntadr[b]]

  # 2. Floating base.
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  if len(free) > 1:
    raise FixError(f"{len(free)} free joints: only the torso may float.")
  if not free:
    tops = [b for b in root.find("worldbody") if b.tag == "body"]
    if len(tops) != 1:
      raise FixError("No free joint, and not exactly one top-level body to give one.")
    tops[0].insert(0, ET.Element("freejoint", name="floating_base"))
    _note(root, tops[0], "added <freejoint/> to this body: the torso must float")
    changes.append(f"added a <freejoint/> to {tops[0].get('name', 'the top body')}")
    model = _compile(root, out)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    bodies = _bodies(root)
  base = int(model.jnt_bodyid[_free(model)])

  # 3. Legs; everything else that moves becomes rigid.
  legs, extras = _find_legs(model, data, base)
  if extras:
    removed = []
    for j, el in [(j, joint_el(j)) for j in extras]:  # Look up all before removing any.
      name = model.joint(j).name
      removed.append(name or f"joint {j}")
      _note(root, el, f"removed joint {name!r}: OnePolicy drives 4 legs x 3 hinges "
            "only; delete it (or its body) in your MJCF")
      _parents(root)[id(el)].remove(el)
      if name:
        _drop_references(root, name)
    changes.append(f"removed {len(removed)} extra joint(s), now rigid: {', '.join(removed)}")
    for kf in root.findall("keyframe"):
      root.remove(kf)
    model = _compile(root, out)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    bodies = _bodies(root)
    base = int(model.jnt_bodyid[_free(model)])
    legs, extras = _find_legs(model, data, base)

  # 4. A foot collision geom on every leg.
  added_feet = False
  mesh_feet = []
  for leg in legs:
    rounded = leg.foot is not None and model.geom_type[leg.foot] in (
      mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE)
    if rounded:
      continue
    knee = data.xanchor[leg.joints[2]]
    b = leg.tip_body
    tip = data.xpos[b] + data.xmat[b].reshape(3, 3) @ leg.tip_local
    shin = float(np.linalg.norm(tip - knee))
    radius = max(0.008, FOOT_RADIUS_FRACTION * shin)
    # Half the sphere past the tip, so it is the lowest point of the leg.
    center = tip - 0.5 * radius * (tip - knee) / max(shin, 1e-9)
    local = data.xmat[b].reshape(3, 3).T @ (center - data.xpos[b])
    body_el = bodies[b - 1]
    geom = ET.SubElement(body_el, "geom", name=f"{body_el.get('name', f'body{b}')}_foot",
                         type="sphere", size=_fmt(radius), pos=_fmt(local),
                         contype="1", conaffinity="1", rgba="0.2 0.2 0.2 1")
    if leg.foot is None:
      _note(root, geom, "added a foot collision sphere at the tip of this leg; give your "
            "MJCF one (feet are the policy's only allowed ground contact)")
      changes.append(f"added a {radius * 1000:.0f} mm foot sphere to {body_el.get('name', b)}")
    else:
      _note(root, geom, "added a foot sphere at the tip of this leg: the foot was a "
            f"{mujoco.mjtGeom(model.geom_type[leg.foot]).name.removeprefix('mjGEOM_').lower()}, every training "
            "robot has a round foot that rolls; give your MJCF one")
      mesh_feet.append(body_el.get("name", str(b)))
    added_feet = True
  if mesh_feet:
    changes.append(f"added {radius * 1000:.0f} mm foot spheres at the leg tips "
                   f"({', '.join(mesh_feet)}): the feet were meshes/boxes, training feet are "
                   "round (the old foot geoms now count as shins)")
  if added_feet:
    model = _compile(root, out)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    bodies = _bodies(root)
    legs, _ = _find_legs(model, data, base)
  for leg in legs:
    # A foot geom that stops well short of the leg's tip.
    knee = data.xanchor[leg.joints[2]]
    fb, flocal = _foot_tip_local(model, data, leg)
    foot = data.xpos[fb] + data.xmat[fb].reshape(3, 3) @ flocal
    tb = leg.tip_body
    tip = data.xpos[tb] + data.xmat[tb].reshape(3, 3) @ leg.tip_local
    short = np.linalg.norm(tip - knee) - np.linalg.norm(foot - knee)
    if short > 0.15 * np.linalg.norm(tip - knee):
      suggestions.append(f"{model.geom(leg.foot).name or 'the foot geom'}: the foot collision "
                         f"geom ends {short * 1000:.0f} mm short of the leg's tip; the "
                         "visual leg will sink into the ground")

  # 5. Zero pose = legs straight down.
  offsets = _rezero(root, model, data, legs, bodies, scale)
  if offsets:
    names = ", ".join(f"{model.joint(j).name or j} {math.degrees(d):+.0f} deg"
                      for j, d in offsets.items())
    changes.append("re-zeroed the legs so that 0 = leg straight down (the training "
                   f"convention): {names}")
    suggestions.append("make zero joint angles mean 'legs straight down' in your MJCF "
                       "(rotate the leg bodies, as in the temp file), or keep the "
                       "offsets in mind on the real robot: q_policy = q_yours - offset")
    model = _compile(root, out)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    bodies = _bodies(root)
    legs, _ = _find_legs(model, data, base)
  joint_offsets = {model.joint(j).name or f"joint{j}": d for j, d in offsets.items()}

  # 6. Which way the robot faces, and which way the knees bend.
  base_rot = data.xmat[base].reshape(3, 3)
  long_axis = np.abs([base_rot.T @ data.xaxis[leg.joints[0]] for leg in legs]).mean(axis=0)
  guess = "+x" if long_axis[0] >= long_axis[1] else "+y"
  facing = guess if forward == "auto" else forward
  bends = _knee_bends(model, data, base, legs, root)
  front_fwd, rear_fwd = _layout(model, data, base, legs, bends, facing)
  if forward == "auto" and front_fwd is True:
    facing = _OPPOSITE[facing]
    front_fwd, rear_fwd = _layout(model, data, base, legs, bends, facing)
    changes.append(f"front knees would bend forward facing {_OPPOSITE[facing]} (no training "
                   f"robot does): assumed the robot faces {facing}; pass forward "
                   f"{_OPPOSITE[facing]} if that's wrong")
  elif forward == "auto" and guess == "+y":
    changes.append("base long axis is y: assumed the robot faces " + facing)
  if front_fwd is True:
    suggestions.append(f"facing {facing}, the FRONT knees bend forward; OnePolicy only "
                       "knows '>>' and '><' legs, so expect a poor gait")
  if rear_fwd is None:
    rear_fwd = is_x_knee(source.stem) if x_knee is None else x_knee
  elif x_knee is not None and x_knee != rear_fwd:
    suggestions.append(f"you said the rear knees are {'><' if x_knee else '>>'}, but the "
                       f"knee joint ranges say {'><' if rear_fwd else '>>'}: used the ranges")

  # Joint signs in the training convention (adapt.py flips the same axes).
  rot = _base_frame(facing) @ base_rot.T
  expected = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 1.0, 0]))
  labels = {}
  for leg in legs:
    anchor = rot @ (data.xanchor[leg.joints[0]] - data.xpos[base])
    labels[id(leg)] = ("F" if anchor[0] > 0 else "R") + ("L" if anchor[1] > 0 else "R")
  if sorted(labels.values()) != sorted(LEG_LABELS):
    raise FixError(f"Legs attach at {sorted(labels.values())}, not one per corner; "
                   "check the forward axis.")
  signs = {j: float(np.sign((rot @ data.xaxis[j]) @ e) or 1.0)
           for leg in legs for j, e in zip(leg.joints, expected)}

  # 7. Standing keyframe in the training pose.
  pose = {}
  fit_notes = []
  for leg in legs:
    label = labels[id(leg)]
    rear = label[0] == "R"
    flip = -1.0 if rear and rear_fwd else 1.0
    train = {"abduction": TRAINING_POSE["hip"] * (1 if label[1] == "R" else -1),
             "hip": TRAINING_POSE["thigh"] * flip, "knee": TRAINING_POSE["calf"] * flip}
    for role, j in zip(ROLES, leg.joints):
      q = signs[j] * train[role]
      if model.jnt_limited[j]:
        lo, hi = model.jnt_range[j]
        if not lo <= q <= hi:
          clipped = float(np.clip(q, lo, hi))
          fit_notes.append(f"{model.joint(j).name}: training angle {q:+.2f} outside its "
                           f"range [{lo:+.2f}, {hi:+.2f}], clipped")
          q = clipped
      pose[j] = q
  if fit_notes:
    suggestions.append("widen these joint ranges to reach the training pose: "
                       + "; ".join(fit_notes))
  qpos = model.qpos0.copy()
  for j, q in pose.items():
    qpos[model.jnt_qposadr[j]] = q
  adr = model.jnt_qposadr[_free(model)]
  qpos[adr:adr + 7] = [0, 0, 0, 1, 0, 0, 0]
  data.qpos[:] = qpos
  mujoco.mj_kinematics(model, data)
  bottoms = []
  for leg in legs:
    point, radius = foot_point(model, data, leg.foot)
    bottoms.append(point[2] - radius)
  qpos[adr + 2] = -min(bottoms) + 0.001
  old = root.findall("keyframe")
  for kf in old:
    for key in kf.iter("key"):
      _note(root, kf, f"your keyframe {key.get('name', '')!r} was replaced (zero pose "
            f"changed / joints removed); it was qpos=\"{key.get('qpos', '')}\"")
    root.remove(kf)
  keyframe = ET.SubElement(root, "keyframe")
  ET.SubElement(keyframe, "key", name="home", qpos=_fmt(qpos))
  _note(root, keyframe, "standing pose OnePolicy was trained around ("
        + ("'><'" if rear_fwd else "'>>'") + " knees); add it to your MJCF")
  changes.append(f"wrote keyframe 'home': the {'><' if rear_fwd else '>>'} training pose, "
                 f"base at {qpos[adr + 2]:.3f} m")

  # 8. Size, mass, motors (data is in the standing pose now).
  mass = float(model.body_subtreemass[base])
  s_m = mass / GO2_MASS
  leg_lengths = []
  data0 = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data0)
  for leg in legs:
    hip, knee = data0.xanchor[leg.joints[1]], data0.xanchor[leg.joints[2]]
    point, _ = foot_point(model, data0, leg.foot)
    leg_lengths.append(np.linalg.norm(knee - hip) + np.linalg.norm(point - knee))
  s_L = float(np.mean(leg_lengths)) / GO2_LEG_LENGTH
  if not TRAINING_SIZE_RANGE[0] <= s_L <= TRAINING_SIZE_RANGE[1]:
    suggestions.append(f"legs are {s_L:.2f}x Go2's; training covered "
                       f"{TRAINING_SIZE_RANGE[0]}-{TRAINING_SIZE_RANGE[1]}x")
  calf = np.mean([sum(model.body_mass[b] for b in _subtree(model, int(model.jnt_bodyid[leg.joints[2]])))
                  for leg in legs]) / mass
  if calf > 3 * GO2_CALF_MASS_FRACTION:
    suggestions.append(f"each lower leg is {calf:.1%} of the total mass (Go2 "
                       f"{GO2_CALF_MASS_FRACTION:.1%}, training up to ~3x that): heavy shins "
                       "swing slowly; move motor mass up to the body if you can")

  own = _own_torque_limits(model)
  stand = _standing_torques(model, data, legs, mass)
  motors = []
  for leg in legs:
    for role, j in zip(ROLES, leg.joints):
      motors.append(MotorCheck(model.joint(j).name or f"joint{j}", role, own.get(j),
                               GO2_PEAK_TORQUE[role] * s_m * s_L, stand[j]))
  for role in ROLES:
    rows = [m for m in motors if m.role == role]
    owns = [m.own_peak for m in rows if m.own_peak is not None]
    want = rows[0].expected_peak
    need = max(m.stand for m in rows)
    if owns:
      have = min(owns)
      if have < MOTOR_WARN_FRACTION * want:
        suggestions.append(
          f"{role} motors: {have:.1f} Nm in your MJCF, {have / want:.0%} of the {want:.1f} Nm "
          f"OnePolicy was trained with at this size (Go2's {GO2_PEAK_TORQUE[role]:.1f} Nm x "
          f"mass {s_m:.2f} x leg length {s_L:.2f}); standing needs {need:.1f} Nm. Fast or "
          "recovering steps ask for more than the real motors give: raise the gear ratio "
          f"~{MOTOR_WARN_FRACTION * want / have:.1f}x or cut mass (the onboard.py sim check "
          "measures what the policy actually uses)")
      if have < 1.2 * need:
        suggestions.append(f"{role} motors ({have:.1f} Nm) can barely hold the robot up "
                           f"({need:.1f} Nm to stand): the robot is unusable as built")

  # 9. Actuator torque clamps, damping, friction on the leg joints.
  clamped = [j for leg in legs for j in leg.joints if model.jnt_actfrclimited[j]]
  if clamped and not own_motor_limits:
    for j in clamped:
      el = joint_el(j)
      el.set("actuatorfrclimited", "false")
      el.attrib.pop("actuatorfrcrange", None)
      lo, hi = model.jnt_actfrcrange[j]
      _note(root, el, f"removed actuatorfrcrange [{lo:g}, {hi:g}] Nm: it clamps the "
            "training actuators below what the policy expects (see the motor note in the "
            "header); keep it in your MJCF only if your motors really stop there")
    changes.append(f"removed joint actuatorfrcrange (±{abs(model.jnt_actfrcrange[clamped[0]]).max():g} "
                   f"Nm) from {len(clamped)} leg joints: the sim now gives the policy the "
                   "torques it was trained with (play --own-motor-limits keeps them)")
  damped = [j for leg in legs for j in leg.joints
            if abs(model.dof_damping[model.jnt_dofadr[j]] - TRAINING_JOINT_DAMPING) > 0.5 * TRAINING_JOINT_DAMPING]
  if damped:
    values = sorted({round(float(model.dof_damping[model.jnt_dofadr[j]]), 4) for j in damped})
    for j in damped:
      el = joint_el(j)
      el.set("damping", _fmt(TRAINING_JOINT_DAMPING))
      _note(root, el, f"damping {model.dof_damping[model.jnt_dofadr[j]]:g} -> "
            f"{TRAINING_JOINT_DAMPING:g} (training robots' passive damping; the policy's "
            "PD gains do the rest)")
    changes.append(f"set leg joint damping {values} -> {TRAINING_JOINT_DAMPING} (training value)")
  friction = max(float(model.dof_frictionloss[model.jnt_dofadr[j]]) for leg in legs for j in leg.joints)
  if friction > 0.05 * min(m.expected_peak for m in motors):
    suggestions.append(f"joint frictionloss up to {friction:.2f} Nm: training had none "
                       "(the sim drops it too). On the real robot that much friction makes "
                       "small corrections stick; lower it (backdrivable gearing) if you can")

  # 10. Header, then write.
  header = [f"{TAG}: fixed copy of {source.name}, written by Training/generation/mjcf_fix.py.",
            "Play/train read this file; edit your own MJCF, not this one.",
            f"Robot: {mass:.2f} kg ({s_m:.2f}x Go2), legs {s_L:.2f}x Go2's, faces {facing}, "
            f"knees {'><' if rear_fwd else '>>'}.", "", "Changed here:"]
  header += [f"  - {c}" for c in changes] or ["  - nothing"]
  if suggestions:
    header += ["", "Suggested changes to your MJCF / robot:"] + [f"  - {s}" for s in suggestions]
  root.insert(0, ET.Comment("\n  " + "\n  ".join(h.replace("--", "-") for h in header) + "\n"))
  _write(root, out)
  mujoco.MjModel.from_xml_path(str(out))  # The written file must load.
  return FixedMjcf(out, source, facing, rear_fwd, joint_offsets, motors, changes, suggestions)


def _drop_references(root: ET.Element, joint: str) -> None:
  """Remove actuators, sensors, tendon entries, equalities and contact
  excludes that refer to `joint`."""
  parents = _parents(root)
  for el in list(root.iter()):
    if el.tag in ("joint", "freejoint") and parents.get(id(el)) is not None \
        and parents[id(el)].tag in ("body", "frame", "worldbody"):
      continue  # A joint definition, not a reference.
    if any(el.get(a) == joint for a in ("joint", "joint1", "joint2", "jointinparent")):
      parent = parents.get(id(el))
      if parent is not None:
        parent.remove(el)
  for tendon in list(root.iter("tendon")):
    for t in list(tendon):
      if t.tag == "fixed" and not len(t):
        tendon.remove(t)


def _rezero(root, model, data, legs, bodies, scale) -> dict[int, float]:
  """Rotate leg bodies so that zero angles put the legs straight down.

  Per leg, solve for the angles (abduction, hip, knee) at which the knee axis
  is horizontal and the thigh (hip -> knee) and shin (knee -> foot) point
  down, then bake them into the bodies: a body with hinge (anchor p, axis a)
  turned by d gets pos += R (p - Rot(a, d) p) and quat *= quat(a, d); its
  joint range moves by -d. Returns joint id -> offset [rad] (only those
  above REZERO_TOLERANCE)."""
  base = int(model.jnt_bodyid[_free(model)])
  work = mujoco.MjData(model)
  offsets: dict[int, float] = {}
  for leg in legs:
    a, h, k = leg.joints
    fb, flocal = _foot_tip_local(model, data, leg)
    adrs = [model.jnt_qposadr[j] for j in leg.joints]

    def residual(q):
      work.qpos[:] = model.qpos0
      work.qpos[adrs] = q
      mujoco.mj_kinematics(model, work)
      rot = work.xmat[base].reshape(3, 3).T
      axis = rot @ work.xaxis[k]
      hip, knee = rot @ work.xanchor[h], rot @ work.xanchor[k]
      foot = rot @ (work.xpos[fb] + work.xmat[fb].reshape(3, 3) @ flocal)
      out = [axis[2]]
      for v in (knee - hip, foot - knee):
        v = v - (v @ axis) * axis  # In the leg's plane.
        n = np.linalg.norm(v)
        out.append(v[0] / n if n > 1e-9 else 0.0)
      return np.array(out)

    def down(q):
      work.qpos[:] = model.qpos0
      work.qpos[adrs] = q
      mujoco.mj_kinematics(model, work)
      rot = work.xmat[base].reshape(3, 3).T
      foot = rot @ (work.xpos[fb] + work.xmat[fb].reshape(3, 3) @ flocal)
      return (rot @ work.xanchor[k])[2] < (rot @ work.xanchor[h])[2] and \
        foot[2] < (rot @ work.xanchor[k])[2]

    best = None
    starts = [(0.0, hs, ks) for hs in (0, math.pi / 2, -math.pi / 2, math.pi)
              for ks in (0, math.pi / 2, -math.pi / 2, math.pi)]
    for start in starts:
      sol = least_squares(residual, np.array(start, float))
      q = (sol.x + math.pi) % (2 * math.pi) - math.pi
      if sol.cost > 1e-4 or not down(q):
        continue
      score = (round(sol.cost, 6), float(np.abs(q).sum()))
      if best is None or score < best[0]:
        best = (score, q)
    if best is None:
      continue  # Can't tell; leave the leg as it is (adapt.py will complain).
    for j, d in zip(leg.joints, best[1]):
      if abs(d) > REZERO_TOLERANCE:
        offsets[j] = float(d)

  for j, d in offsets.items():
    b = int(model.jnt_bodyid[j])
    body = bodies[b - 1]
    if model.body_jntnum[b] != 1:
      continue  # Can't rotate a body between two of its own joints.
    el = _own(body, ("joint",))[0]
    turn = d - float(model.qpos0[model.jnt_qposadr[j]])  # A joint `ref` is already applied.
    axis, anchor = model.jnt_axis[j], model.jnt_pos[j]
    q_turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_turn, axis, turn)
    r_turn = np.zeros(9)
    mujoco.mju_quat2Mat(r_turn, q_turn)
    old_quat = _local_quat(body, root)
    old_rot = np.zeros(9)
    mujoco.mju_quat2Mat(old_rot, old_quat)
    pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
    pos = pos + old_rot.reshape(3, 3) @ (anchor - r_turn.reshape(3, 3) @ anchor)
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, old_quat, q_turn)
    for attr in ("quat", "axisangle", "euler", "xyaxes", "zaxis"):
      body.attrib.pop(attr, None)
    body.set("pos", _fmt(pos))
    body.set("quat", _fmt(new_quat))
    el.attrib.pop("ref", None)
    for key in root.iter("key"):  # Keyframes keep meaning the same pose.
      values = [float(v) for v in key.get("qpos", "").split()]
      if len(values) == model.nq:
        values[model.jnt_qposadr[j]] -= d
        key.set("qpos", _fmt(values))
    if model.jnt_limited[j]:
      lo, hi = model.jnt_range[j] - d
      el.set("range", _fmt(np.array([lo, hi]) * scale))
    _note(root, body, f"rotated {turn * 180 / math.pi:+.1f} deg about joint "
          f"{model.joint(j).name!r} so that 0 = leg straight down; joint range shifted "
          f"by {-d:+.3f} rad (in your MJCF: same rotation, or remember the offset)")
  return {j: d for j, d in offsets.items() if model.body_jntnum[model.jnt_bodyid[j]] == 1}


def _knee_bends(model, data, base, legs, root) -> dict[int, float | None]:
  """Per knee joint: the sign of the angle it flexes to (+1/-1), from its
  range (one-sided) or the existing keyframe; None if it can't tell."""
  bends = {}
  for leg in legs:
    k = leg.joints[2]
    sign = None
    if model.jnt_limited[k]:
      lo, hi = model.jnt_range[k]
      if hi < ONE_SIDED and lo < -ONE_SIDED:
        sign = -1.0
      elif lo > -ONE_SIDED and hi > ONE_SIDED:
        sign = 1.0
    if sign is None and model.nkey:
      q = model.key_qpos[0][model.jnt_qposadr[k]]
      if abs(q) > ONE_SIDED:
        sign = float(np.sign(q))
    bends[k] = sign
  return bends


def _layout(model, data, base, legs, bends, facing) -> tuple[bool | None, bool | None]:
  """(front knees bend forward, rear knees bend forward) facing `facing`; None
  where the ranges can't tell."""
  rot = _base_frame(facing) @ data.xmat[base].reshape(3, 3).T
  work = mujoco.MjData(model)
  result: dict[str, list[bool]] = {"F": [], "R": []}
  for leg in legs:
    a, h, k = leg.joints
    sign = bends[k]
    end = "F" if (rot @ (data.xanchor[a] - data.xpos[base]))[0] > 0 else "R"
    if sign is None:
      continue
    # Flex the knee 1 rad, swing the thigh back by half: is the knee ahead of
    # the hip-foot line?
    work.qpos[:] = model.qpos0
    work.qpos[model.jnt_qposadr[k]] = sign * 1.0
    work.qpos[model.jnt_qposadr[h]] = 0.0
    mujoco.mj_kinematics(model, work)
    hip, knee = rot @ work.xanchor[h], rot @ work.xanchor[k]
    fb, flocal = _foot_tip_local(model, data, leg)
    foot = rot @ (work.xpos[fb] + work.xmat[fb].reshape(3, 3) @ flocal)
    line = hip + (foot - hip) * ((knee - hip) @ (foot - hip)) / max((foot - hip) @ (foot - hip), 1e-12)
    result[end].append(bool((knee - line)[0] > 0))
  front = None if not result["F"] else all(result["F"])
  rear = None if not result["R"] else all(result["R"])
  return front, rear


def _own_torque_limits(model) -> dict[int, float]:
  """Joint -> the smallest torque limit the MJCF sets: joint actuatorfrcrange,
  actuator forcerange, or a motor's ctrlrange x gear."""
  limits: dict[int, list[float]] = {}
  for j in range(model.njnt):
    if model.jnt_actfrclimited[j]:
      limits.setdefault(j, []).append(float(np.abs(model.jnt_actfrcrange[j]).max()))
  for a in range(model.nu):
    if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
      continue
    j = int(model.actuator_trnid[a, 0])
    gear = abs(float(model.actuator_gear[a, 0]))
    if model.actuator_forcelimited[a]:
      limits.setdefault(j, []).append(gear * float(np.abs(model.actuator_forcerange[a]).max()))
    elif (model.actuator_ctrllimited[a] and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE
          and model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED):
      limits.setdefault(j, []).append(
        gear * float(model.actuator_gainprm[a, 0]) * float(np.abs(model.actuator_ctrlrange[a]).max()))
  return {j: min(v) for j, v in limits.items()}


def _standing_torques(model, data, legs, mass) -> dict[int, float]:
  """Static torque per leg joint holding mass/4 on each foot (data posed)."""
  out = {}
  weight = np.array([0.0, 0.0, mass * 9.81 / 4])
  for leg in legs:
    point, _ = foot_point(model, data, leg.foot)
    for j in leg.joints:
      out[j] = float(abs(np.cross(point - data.xanchor[j], weight) @ data.xaxis[j]))
  return out
