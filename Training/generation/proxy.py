"""Proxies: real quadrupeds rebuilt in the generated robots' model structure.

mjwarp batches every env against one model structure (see bank.py), and a
real robot's MJCF (Spot, Go2, ...) has its own bodies, geoms and meshes, so it
can't share a batch with the generated robots. A proxy keeps the generated
structure (create_robot's bodies, joints and primitive geoms) and takes the
real robot's numbers, measured in its zero pose after adapt.py has put it in
the training convention:

  * joint positions: hip (abduction), thigh and knee anchors, foot centre;
  * link inertia: each proxy link gets the mass, centre of mass and inertia of
    the real bodies moved by its joint (base: everything not in a leg);
  * collision: base box spanning the real base's collision geoms, foot sphere
    of the real foot radius, thigh/calf cylinders along the real links.

Proxies are used by train.py --generalise-proxies True; by default real
robots are trained on their own models instead (tasks/velocity/rl/multi_env.py).
Each proxy carries the real robot's own joint ranges and default pose (bank).

Limits of the approximation, forced by the shared structure: joint axes stay
the generated ones (x, y, y; see `check`), inertia tensors lose
their off-diagonal terms (principal axes aligned with the link), the base box
is centred on the base origin, and visual geoms are the generated primitives.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .adapt import adapt_quadruped
from .bank import DEFAULT_MAX_JOINT_VELOCITY, _compile, _extract, foot_point
from .catalog import default_pose, find_robot, robot_defaults
from .mjcf_builder import make_robot_xml

LEGS = ("FR", "FL", "RR", "RL")
ROLES = ("hip", "thigh", "calf")
TEMPLATE_SEED = 0
MIN_FOOT_RADIUS = 0.015
"""Foot sphere radius [m] for robots whose foot is a mesh (no radius)."""


@dataclass
class ProxyReport:
  name: str
  total_mass: float
  real_total_mass: float
  notes: list[str] = field(default_factory=list)


def _fmt(v) -> str:
  return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _compile_adapted(xml_path: str, forward: str):
  robot = adapt_quadruped(xml_path, forward)
  with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "robot.xml"
    path.write_text(robot.xml)
    meshdir = mujoco.MjSpec.from_string(robot.xml).meshdir
    for name, data in robot.assets.items():
      p = Path(d) / meshdir / name
      p.parent.mkdir(parents=True, exist_ok=True)
      p.write_bytes(data)
    model = mujoco.MjModel.from_xml_path(str(path))
  return robot, model


def _subtree(model, b: int) -> set[int]:
  found = {b}
  for c in range(b + 1, model.nbody):
    if model.body_parentid[c] in found:
      found.add(c)
  return found


def _lump(model, data, bodies, origin) -> tuple[float, np.ndarray, np.ndarray]:
  """Mass, COM (relative to `origin`) and diagonal inertia about the COM, in
  world axes (the proxy links are unrotated in the zero pose)."""
  bodies = [b for b in bodies if model.body_mass[b] > 0]
  mass = float(sum(model.body_mass[b] for b in bodies))
  if mass <= 0:
    return 0.0, np.zeros(3), np.zeros(3)
  com = sum(model.body_mass[b] * data.xipos[b] for b in bodies) / mass
  inertia = np.zeros((3, 3))
  for b in bodies:
    rot = data.ximat[b].reshape(3, 3)
    inertia += rot @ np.diag(model.body_inertia[b]) @ rot.T
    r = data.xipos[b] - com
    inertia += model.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
  return mass, com - origin, np.maximum(np.diag(inertia), 1e-7)


def check(model, data) -> list[str]:
  """Differences from the generated structure the proxy can't represent."""
  notes = []
  want = {"hip": np.array([1.0, 0, 0]), "thigh": np.array([0, 1.0, 0]),
          "calf": np.array([0, 1.0, 0])}
  for leg in LEGS:
    for role in ROLES:
      j = model.joint(f"{leg}_{role}_joint").id
      dev = np.degrees(np.arccos(np.clip(abs(data.xaxis[j] @ want[role]), -1, 1)))
      if dev > 5:
        notes.append(f"{leg}_{role}_joint axis {np.round(data.xaxis[j], 2)} is "
                     f"{dev:.0f} deg off the generated {want[role]}")
  return notes


def make_proxy(name: str, train_pose: dict[str, float], clearance: float):
  """Bank entry for a proxy of real robot `name` (see bank.generate_bank).

  Returns (source, seed, xml, scalars, fields, structure, notes).
  """
  path = find_robot(name)
  forward = robot_defaults(path).get("forward", "auto")
  xml, pose, report = make_proxy_xml(name, str(path), train_pose, forward)
  model = _compile(xml)
  fields, structure, scalars = _extract(model, pose, clearance)
  scalars.update(total_mass=float(model.body_subtreemass[1]),
                 max_joint_velocity=DEFAULT_MAX_JOINT_VELOCITY)
  notes = [f"mass {report.total_mass:.2f} kg (real {report.real_total_mass:.2f}), "
           f"leg length {scalars['leg_length']:.3f} m, foot radius "
           f"{scalars['foot_radius']:.3f} m"] + report.notes
  return name, -1, xml, scalars, fields, structure, notes


def make_proxy_xml(name: str, xml_path: str, train_pose: dict[str, float],
                   forward: str = "auto") -> tuple[str, dict[str, float], ProxyReport]:
  """Generated-structure MJCF with the numbers of the real robot at xml_path,
  and the robot's default pose (keyframe, else the training pose fitted)."""
  robot, model = _compile_adapted(xml_path, forward)
  pose, log = default_pose(robot, Path(xml_path), "auto", train_pose)
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  data.qpos[:3] = 0.0
  data.qpos[3:7] = (1, 0, 0, 0)
  mujoco.mj_kinematics(model, data)
  report = ProxyReport(name, 0.0, float(model.body_subtreemass[1]))
  report.notes += [line for line in log if not line.startswith("  ")]
  report.notes += check(model, data)

  template_xml, _, _ = make_robot_xml(TEMPLATE_SEED)
  root = ET.fromstring(template_xml)
  root.set("model", f"proxy_{name}")
  bodies = {b.get("name"): b for b in root.iter("body")}

  def set_inertial(body_name: str, mass: float, com: np.ndarray, diag: np.ndarray):
    inertial = bodies[body_name].find("inertial")
    inertial.set("mass", f"{max(mass, 1e-4):.6g}")
    inertial.set("diaginertia", _fmt(diag))
    if inertial.get("pos") != "0 0 0":  # Bodies whose COM sits at the origin keep it.
      inertial.set("pos", _fmt(com))

  base_id = 1
  leg_bodies: set[int] = set()
  anchors = {}
  for leg in LEGS:
    joints = [model.joint(f"{leg}_{role}_joint").id for role in ROLES]
    pts = [data.xanchor[j].copy() for j in joints]
    foot, radius = foot_point(model, data, model.geom(f"{leg}_foot_collision").id)
    anchors[leg] = (*pts, foot, max(radius, MIN_FOOT_RADIUS))
    leg_bodies |= _subtree(model, int(model.jnt_bodyid[joints[0]]))

  # Base: everything not in a leg (IMU, payload, ...).
  base = [b for b in _subtree(model, base_id) if b not in leg_bodies]
  mass, com, diag = _lump(model, data, base, data.xpos[base_id])
  set_inertial("base_link", mass, com, diag)
  report.total_mass += mass
  # Base box: half-extents of the real base's collision geoms about the origin.
  base_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in base
                and (model.geom_contype[g] or model.geom_conaffinity[g])]
  if base_geoms:
    ext = np.zeros(3)
    for g in base_geoms:
      center, half = model.geom_aabb[g].reshape(2, 3)
      rot = data.geom_xmat[g].reshape(3, 3)
      for s in np.ndindex(2, 2, 2):
        corner = data.geom_xpos[g] + rot @ (center + half * (2 * np.array(s) - 1))
        ext = np.maximum(ext, np.abs(corner - data.xpos[base_id]))
    for geom in bodies["base_link"].findall("geom"):
      geom.set("size", _fmt(ext))

  for leg in LEGS:
    hip, thigh, knee, foot, radius = anchors[leg]
    base_pos = data.xpos[base_id]
    joints = [model.joint(f"{leg}_{role}_joint").id for role in ROLES]
    # Real bodies moved by each joint but not by the next one down the leg.
    sub = [_subtree(model, int(model.jnt_bodyid[j])) for j in joints]
    own = [sub[0] - sub[1], sub[1] - sub[2], sub[2]]
    for role, members, origin in zip(ROLES, own, (hip, thigh, knee)):
      mass, com, diag = _lump(model, data, members, origin)
      set_inertial(f"{leg}_{role}", mass, com, diag)
      report.total_mass += mass
    # The hip_fixed and foot links of the template keep a token mass.
    for extra in (f"{leg}_hip_fixed", f"{leg}_foot"):
      inertial = bodies[extra].find("inertial")
      inertial.set("mass", "0.001")
      inertial.set("diaginertia", "1e-07 1e-07 1e-07")
      report.total_mass += 0.001

    bodies[f"{leg}_hip"].set("pos", _fmt(hip - base_pos))
    offset = thigh - hip
    bodies[f"{leg}_hip_fixed"].set("pos", _fmt(offset))
    bodies[f"{leg}_thigh"].set("pos", _fmt(offset))
    bodies[f"{leg}_calf"].set("pos", _fmt(knee - thigh))
    bodies[f"{leg}_foot"].set("pos", _fmt(foot - knee))
    for geom in bodies[f"{leg}_hip"].findall("geom"):
      if geom.get("fromto"):
        geom.set("fromto", f"0 0 0 {_fmt(offset)}")

    for role, start, end in (("thigh", thigh, knee), ("calf", knee, foot)):
      link = end - start
      length = float(np.linalg.norm(link))
      for geom in bodies[f"{leg}_{role}"].findall("geom"):
        r, half = (float(x) for x in geom.get("size").split())
        geom.set("pos", _fmt(link / 2))
        if geom.get("name").endswith("_visual"):
          geom.set("size", _fmt([r, length / 2]))
        else:  # Template collision cylinder: a tenth of the link long.
          geom.set("size", _fmt([r, length / 10]))
    for geom in bodies[f"{leg}_foot"].findall("geom"):
      geom.set("size", _fmt(radius))
    for site in bodies[f"{leg}_foot"].findall("site"):
      site.set("size", _fmt(radius))

  # The real joint ranges (unlimited joints keep the generated range).
  for joint in root.iter("joint"):
    real = robot.joint_ranges.get(joint.get("name"))
    if real is not None:
      joint.set("range", _fmt(real))

  return ET.tostring(root, encoding="unicode"), pose, report
