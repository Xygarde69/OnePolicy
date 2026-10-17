"""Turn one GenLoco random quadruped into a Go2-compatible MJCF robot.

The morphology comes from GenLoco's `quadruped_generator.create_robot`
(unchanged). It builds the robot through PyBullet's `createMultiBody`, so we:

  1. run `create_robot` against a headless PyBullet client wrapped in a
     recorder that captures every shape / multibody call it makes,
  2. read back the resulting masses, inertias and friction from PyBullet,
  3. translate that multibody 1:1 into an MJCF robot (no floor, lights or
     actuators; mjlab's scene and actuator configs provide those).

Every element is named after its Unitree Go2 counterpart, so any config written
for Go2 (rewards, sensors, contact patterns, joint regexes) works unchanged:

  bodies   base_link, {leg}_hip, {leg}_thigh, {leg}_calf, {leg}_foot
  joints   floating_base_joint, {leg}_hip_joint, {leg}_thigh_joint, {leg}_calf_joint
  geoms    *_collision (collision), *_visual (visual), {leg}_foot_collision
  sites    imu, FL / FR / RL / RR (feet)
  sensors  imu_ang_vel, imu_lin_vel, imu_lin_acc, root_angmom
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import pybullet

from . import quadruped_generator
from .knees import X_KNEE_TAG

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(GENERATION_DIR, "robot_descriptions", "a1_description", "meshes")

LEGS = ("FR", "FL", "RR", "RL")  # create_robot's leg order.

# create_robot link labels -> Go2 body names, in create_robot's link order.
GO2_BODY_NAMES = ["imu_link"] + [
  f"{leg}_{part}"
  for leg in LEGS
  for part in ("hip", "hip_fixed", "thigh", "calf", "foot")
]
# Revolute joints in create_robot order (per leg: abduction, hip, knee).
GO2_JOINT_NAMES = [
  f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")
]

# GenLoco ACTION_CONFIG bounds (random_quadruped.py), per leg: abduction, hip, knee.
JOINT_RANGES = [
  (-0.802851455917, 0.802851455917),
  (-1.0471975512, 4.18879020479),
  (-2.69653369433, -0.916297857297),
] * 4


def joint_ranges(x_knee: bool) -> list[tuple[float, float]]:
  """JOINT_RANGES in GO2_JOINT_NAMES order; for a "><" robot (knees.py) the
  rear thigh and calf ranges are mirrored, so the rear knees bend forward."""
  return [
    (-hi, -lo) if x_knee and name.startswith("R") and "_hip_" not in name else (lo, hi)
    for name, (lo, hi) in zip(GO2_JOINT_NAMES, JOINT_RANGES)
  ]

# Each link's inertia is multiplied by uniform(1 - d, 1 + d) (d scaled by
# delta_scale), all three axes alike so it stays physically valid: mass spread
# further from or closer to the link's COM than a uniform-density solid.
INERTIA_DELTA = 0.3

PYBULLET_FIXED = 4
PYBULLET_REVOLUTE = 0


class RecordingClient:
  """Pass-through PyBullet client that records shape and body creation."""

  def __init__(self, client):
    self._client = client
    self.collision_shapes = {}
    self.visual_shapes = {}
    self.multibody_args = None
    self.max_joint_velocities = {}  # PyBullet joint index -> rad/s

  def __getattr__(self, name):
    return getattr(self._client, name)

  def createCollisionShape(self, **kwargs):
    idx = self._client.createCollisionShape(**kwargs)
    self.collision_shapes[idx] = kwargs
    return idx

  def createVisualShape(self, **kwargs):
    idx = self._client.createVisualShape(**kwargs)
    self.visual_shapes[idx] = kwargs
    return idx

  def createMultiBody(self, baseMass, **kwargs):
    uid = self._client.createMultiBody(baseMass, **kwargs)
    self.multibody_args = dict(kwargs, baseMass=baseMass)
    return uid

  def changeDynamics(self, bodyUniqueId, linkIndex, **kwargs):
    if "maxJointVelocity" in kwargs:
      self.max_joint_velocities[linkIndex] = kwargs["maxJointVelocity"]
    return self._client.changeDynamics(bodyUniqueId, linkIndex, **kwargs)


def fmt(values):
  return " ".join(f"{float(v):.6g}" for v in np.atleast_1d(values))


def mj_quat(xyzw):
  x, y, z, w = xyzw
  return (w, x, y, z)


def generate_morphology(seed: int, delta_scale: float = 1.0, x_knee_fraction: float = 0.5,
                        params: dict | None = None):
  """Run GenLoco's generator in headless PyBullet; return recorded data and
  whether the robot is "><" (rear knees forward, probability x_knee_fraction).

  params: values that replace random draws (quadruped_generator.PARAM_KEYS),
  plus "x_knee" (bool) to pick the knee layout."""
  params = dict(params or {})
  forced_x_knee = params.pop("x_knee", None)
  # create_robot draws from numpy's global RNG.
  np.random.seed(seed)
  cid = pybullet.connect(pybullet.DIRECT)
  try:
    pybullet.setAdditionalSearchPath(GENERATION_DIR, physicsClientId=cid)
    rec = RecordingClient(pybullet)
    uid = quadruped_generator.create_robot(rec, delta_scale=delta_scale, params=params)
    dynamics = {
      i: pybullet.getDynamicsInfo(uid, i, physicsClientId=cid)
      for i in range(-1, pybullet.getNumJoints(uid, physicsClientId=cid))
    }
  finally:
    pybullet.disconnect(cid)
  # PyBullet's inertia assumes uniform density; randomise it per link.
  d = INERTIA_DELTA * delta_scale
  for i, dyn in dynamics.items():
    factor = np.random.uniform(1 - d, 1 + d)
    dynamics[i] = (dyn[0], dyn[1], tuple(np.asarray(dyn[2]) * factor), *dyn[3:])
  x_knee = bool(np.random.uniform() < x_knee_fraction)
  if forced_x_knee is not None:
    x_knee = bool(forced_x_knee)
  return rec, dynamics, x_knee


def shape_to_geom(shape, frame_prefix, length_key):
  """Translate PyBullet createCollision/VisualShape kwargs to MJCF geom attrs."""
  stype = shape["shapeType"]
  attrs = {
    "pos": fmt(shape.get(f"{frame_prefix}FramePosition", (0, 0, 0))),
    "quat": fmt(mj_quat(shape.get(f"{frame_prefix}FrameOrientation", (0, 0, 0, 1)))),
  }
  if stype == pybullet.GEOM_BOX:
    attrs.update(type="box", size=fmt(shape.get("halfExtents", (0.5, 0.5, 0.5))))
  elif stype == pybullet.GEOM_SPHERE:
    attrs.update(type="sphere", size=fmt(shape.get("radius", 0.5)))
  elif stype == pybullet.GEOM_CYLINDER:
    half_len = shape.get(length_key, 1.0) / 2.0
    attrs.update(type="cylinder", size=fmt((shape.get("radius", 0.5), half_len)))
  elif stype == pybullet.GEOM_MESH:
    # Only the hip uses a mesh (hip.obj, unit scale) in create_robot.
    attrs.update(type="mesh", mesh=os.path.splitext(os.path.basename(shape["fileName"]))[0])
  else:
    raise ValueError(f"Unsupported PyBullet shape type {stype}")
  return attrs


def add_inertial(body, dyn):
  mass, _, inertia, ipos, iorn = dyn[:5]
  ET.SubElement(body, "inertial", {
    "pos": fmt(ipos),
    "quat": fmt(mj_quat(iorn)),
    "mass": fmt(mass),
    # Links without a collision shape get zero inertia in PyBullet;
    # MuJoCo needs a strictly positive value.
    "diaginertia": fmt(np.maximum(inertia, 1e-7)),
  })


def add_geoms(body, name, rec, col_idx, vis_idx, friction):
  if vis_idx >= 0:
    vis = rec.visual_shapes[vis_idx]
    attrs = shape_to_geom(vis, "visual", "length")
    attrs.update(name=f"{name}_visual", rgba=fmt(vis.get("rgbaColor", (1, 1, 1, 1))),
                 **{"class": "visual"})
    ET.SubElement(body, "geom", attrs)
  if col_idx >= 0:
    col = rec.collision_shapes[col_idx]
    attrs = shape_to_geom(col, "collision", "height")
    attrs.update(name=f"{name}_collision", friction=fmt((friction, 0.005, 0.0001)),
                 **{"class": "collision"})
    ET.SubElement(body, "geom", attrs)


def build_mjcf(rec, dynamics, seed: int, x_knee: bool = False) -> ET.Element:
  """Build the Go2-named MJCF robot from a recorded create_robot call. A
  "><" robot gets mirrored rear knee ranges and X_KNEE_TAG in its model name."""
  mb = rec.multibody_args

  model_name = f"morph_quadruped_{seed}" + (f"_{X_KNEE_TAG}" if x_knee else "")
  root = ET.Element("mujoco", model=model_name)
  ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false",
                autolimits="true")

  default = ET.SubElement(root, "default")
  ET.SubElement(default, "joint", damping="0.1", armature="0.01")
  vis_cls = ET.SubElement(default, "default", {"class": "visual"})
  ET.SubElement(vis_cls, "geom", contype="0", conaffinity="0", density="0", group="2")
  col_cls = ET.SubElement(default, "default", {"class": "collision"})
  ET.SubElement(col_cls, "geom", group="3")
  ET.SubElement(default, "site", rgba="1 0 0 1", group="5")

  asset = ET.SubElement(root, "asset")
  ET.SubElement(asset, "mesh", name="hip", file="hip.obj")

  world = ET.SubElement(root, "worldbody")
  base = ET.SubElement(world, "body", name="base_link", pos="0 0 0")
  ET.SubElement(base, "freejoint", name="floating_base_joint")
  add_inertial(base, dynamics[-1])
  add_geoms(base, "base", rec, mb["baseCollisionShapeIndex"],
            mb["baseVisualShapeIndex"], dynamics[-1][1])

  # PyBullet linkParentIndices: 0 = base, k = link k-1.
  bodies = {0: base}
  joint_iter = iter(zip(GO2_JOINT_NAMES, joint_ranges(x_knee)))
  for i, name in enumerate(GO2_BODY_NAMES):
    parent = bodies[mb["linkParentIndices"][i]]
    body = ET.SubElement(parent, "body", name=name,
                         pos=fmt(mb["linkPositions"][i]),
                         quat=fmt(mj_quat(mb["linkOrientations"][i])))
    bodies[i + 1] = body
    jtype = mb["linkJointTypes"][i]
    if jtype == PYBULLET_REVOLUTE:
      jname, (lo, hi) = next(joint_iter)
      ET.SubElement(body, "joint", name=jname, axis=fmt(mb["linkJointAxis"][i]),
                    range=fmt((lo, hi)))
    elif jtype != PYBULLET_FIXED:
      raise ValueError(f"Unsupported PyBullet joint type {jtype}")
    add_inertial(body, dynamics[i])
    add_geoms(body, name, rec, mb["linkCollisionShapeIndices"][i],
              mb["linkVisualShapeIndices"][i], dynamics[i][1])

    if name.endswith("_hip"):
      # Visual only: create_robot gives the hip link no shape spanning its
      # (randomised) length. Radius and colour follow the thigh's visual.
      thigh_pos = mb["linkPositions"][i + 2]
      thigh_visual = rec.visual_shapes[mb["linkVisualShapeIndices"][i + 2]]
      ET.SubElement(body, "geom", {
        "name": f"{name}_link_visual",
        "type": "cylinder",
        "fromto": fmt((0, 0, 0, *thigh_pos)),
        "size": fmt(thigh_visual["radius"]),
        "rgba": fmt(thigh_visual.get("rgbaColor", (1, 1, 1, 1))),
        "class": "visual",
      })
    elif name == "imu_link":
      ET.SubElement(body, "site", name="imu", pos="0 0 0")
    elif name.endswith("_foot"):
      leg = name.split("_")[0]
      ET.SubElement(body, "site", name=leg, type="sphere", pos="0 0 0",
                    size=fmt(rec.collision_shapes[mb["linkCollisionShapeIndices"][i]]["radius"]))

  sensor = ET.SubElement(root, "sensor")
  ET.SubElement(sensor, "gyro", name="imu_ang_vel", site="imu")
  ET.SubElement(sensor, "velocimeter", name="imu_lin_vel", site="imu")
  ET.SubElement(sensor, "accelerometer", name="imu_lin_acc", site="imu")
  ET.SubElement(sensor, "subtreeangmom", name="root_angmom", body="base_link")

  ET.indent(root)
  return root


def mesh_assets() -> dict[str, bytes]:
  """Mesh bytes keyed by file name, for compiling from a string or spec."""
  return {
    f: open(os.path.join(MESH_DIR, f), "rb").read()
    for f in os.listdir(MESH_DIR)
    if f.endswith(".obj")
  }


def make_robot_xml(seed: int, delta_scale: float = 1.0, x_knee_fraction: float = 0.5,
                   params: dict | None = None) -> tuple[str, float, float]:
  """Generate one random quadruped.

  Returns (MJCF string, total mass [kg], max joint velocity [rad/s]). MJCF has
  no joint velocity limit, so the velocity create_robot gives every revolute
  joint (PyBullet maxJointVelocity) is returned for the actuators to enforce.
  """
  rec, dynamics, x_knee = generate_morphology(seed, delta_scale, x_knee_fraction, params)
  total_mass = float(sum(d[0] for d in dynamics.values()))
  max_joint_velocity = float(max(rec.max_joint_velocities.values()))
  xml = ET.tostring(build_mjcf(rec, dynamics, seed, x_knee), encoding="unicode")
  return xml, total_mass, max_joint_velocity
