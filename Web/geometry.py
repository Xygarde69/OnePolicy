"""Robot geometry for the browser's 3D views (three.js).

robot_geometry() compiles a robot in the training convention (generated, or a
real MJCF fixed + adapted like play.py does), puts it in its default pose and
returns every visible geom in world coordinates, tagged with the part it
belongs to (base, or <leg>_<hip|thigh|calf>), plus its meshes as base64
float32 vertices / int32 faces.
"""

from __future__ import annotations

import base64
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from assets.morph_quadruped.constants import INIT_JOINT_POS
from generation.bank import _leg_dimensions, _resolved_pose
from generation.catalog import load_robot
from generation.conditions import robot_variables
from generation.knees import is_x_knee, x_knee_pose
from generation.mjcf_builder import make_robot_xml, mesh_assets

GEOM_TYPES = {mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
              mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
              mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_MESH: "mesh"}
LEGS = ("FR", "FL", "RR", "RL")
ROLES = ("hip", "thigh", "calf")
# Meshes above this many faces are thinned (every n-th face) to keep the page fast.
MAX_FACES = 60_000


def _b64(array: np.ndarray, dtype) -> str:
  return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def _parts(model: mujoco.MjModel) -> list[str]:
  """Part of every body: 'world', 'base' or '<leg>_<role>' (feet count as calf)."""
  starts = {}
  for leg in LEGS:
    for role in ROLES:
      j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{role}_joint")
      if j >= 0:
        starts[int(model.jnt_bodyid[j])] = f"{leg}_{role}"
  parts = ["world"]
  for b in range(1, model.nbody):
    parent = int(model.body_parentid[b])
    parts.append(starts.get(b, "base" if parent == 0 else parts[parent]))
  return parts


def _geometry(model: mujoco.MjModel, pose: dict[str, float], collision_only: bool = False) -> dict:
  """collision_only: skip visual geoms (the generator's are A1 decorations of a
  fixed size, not scaled with the robot)."""
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  for name, value in pose.items():
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j >= 0:
      data.qpos[model.jnt_qposadr[j]] = value
  mujoco.mj_kinematics(model, data)
  parts = _parts(model)

  # Per body: its visual-only geoms if it has any, else all its geoms.
  def visual(g):
    return model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
  has_visual = {int(model.geom_bodyid[g]) for g in range(model.ngeom) if visual(g)}

  geoms, meshes = [], {}
  for g in range(model.ngeom):
    body, kind = int(model.geom_bodyid[g]), GEOM_TYPES.get(int(model.geom_type[g]))
    if collision_only and visual(g):
      continue
    if body == 0 or kind is None or (body in has_visual and not visual(g) and not collision_only):
      continue
    if model.geom_rgba[g][3] == 0:
      continue
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
    geom = {"type": kind, "part": parts[body], "size": model.geom_size[g].tolist(),
            "pos": data.geom_xpos[g].tolist(), "quat": quat.tolist(),
            "rgba": model.geom_rgba[g].tolist()}
    if kind == "mesh":
      m = int(model.geom_dataid[g])
      geom["mesh"] = m
      if m not in meshes:
        va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
        fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
        faces = model.mesh_face[fa:fa + fn]
        if fn > MAX_FACES:
          faces = faces[::int(np.ceil(fn / MAX_FACES))]
        meshes[m] = {"vertices": _b64(model.mesh_vert[va:va + vn], np.float32),
                     "faces": _b64(faces, np.int32)}
    geoms.append(geom)

  lo = np.min([g["pos"] for g in geoms], axis=0) if geoms else np.zeros(3)
  hi = np.max([g["pos"] for g in geoms], axis=0) if geoms else np.zeros(3)
  return {"geoms": geoms, "meshes": {str(k): v for k, v in meshes.items()},
          "bounds": [lo.tolist(), hi.tolist()]}


def generated_geometry(seed: int, params: dict | None = None) -> dict:
  xml, total_mass, max_vel = make_robot_xml(seed, params=params)
  model = mujoco.MjModel.from_xml_string(xml, mesh_assets())
  x_knee = is_x_knee(ET.fromstring(xml).get("model", ""))
  pose = x_knee_pose(INIT_JOINT_POS) if x_knee else INIT_JOINT_POS
  out = _geometry(model, _resolved_pose(model, pose), collision_only=True)
  out["info"] = _info(model, total_mass, max_vel, x_knee)
  out["name"] = f"generated #{seed}"
  return out


def file_geometry(xml_path: Path, forward: str = "auto", x_knee: bool | None = None) -> dict:
  """A real robot MJCF, fixed and adapted as play.py does."""
  loaded = load_robot(xml_path, forward, "auto", x_knee=x_knee)
  with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    path = out_dir / "robot.xml"
    path.write_text(loaded.robot.xml)
    meshdir = mujoco.MjSpec.from_string(loaded.robot.xml).meshdir
    for name, data in loaded.robot.assets.items():
      asset = out_dir / meshdir / name
      asset.parent.mkdir(parents=True, exist_ok=True)
      asset.write_bytes(data)
    model = mujoco.MjModel.from_xml_path(str(path))
  out = _geometry(model, loaded.pose)
  x_knee = bool(loaded.pose.get("RR_calf_joint", -1.0) > 0)
  out["info"] = _info(model, float(model.body_subtreemass[1]), None, x_knee)
  out["name"] = xml_path.stem
  out["fixes"] = loaded.fixed.summary().splitlines()
  return out


def _info(model: mujoco.MjModel, mass: float, max_vel: float | None, x_knee: bool) -> dict:
  """Key numbers shown next to the robot (generation/conditions.py variables)."""
  leg_length, foot_radius = _leg_dimensions(model)
  try:
    variables = robot_variables(model, {
      "total_mass": mass, "leg_length": leg_length, "foot_radius": foot_radius,
      "base_above_feet": 0.0, "max_joint_velocity": max_vel or 0.0, "x_knee": float(x_knee)})
  except (KeyError, ValueError):
    variables = {"mass": mass, "leg_length": leg_length}
  variables.pop("stand_height", None)
  if max_vel is None:
    variables.pop("max_joint_velocity", None)
  return variables
