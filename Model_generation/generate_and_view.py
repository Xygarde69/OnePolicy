#!/usr/bin/env python3
"""Procedurally generate a random quadruped (GenLoco) and visualise it in MuJoCo.

The morphology itself is produced by GenLoco's own generator,
`quadruped_generator.create_robot` (copied verbatim from
GenLoco/motion_imitation/utilities/quadruped_generator.py). That generator
builds the robot through PyBullet's `createMultiBody` API, so this script:

  1. runs `create_robot` against a headless PyBullet client wrapped in a
     recorder that captures every shape / multibody call it makes,
  2. reads back the resulting masses, inertias and friction from PyBullet,
  3. translates that multibody 1:1 into an MJCF file, and
  4. opens it in the MuJoCo viewer.

No randomisation happens here; every random draw is made inside
`create_robot`. A fresh seed is drawn each run, so each run yields a new robot.

Usage:
    python generate_and_view.py                  # new random robot + viewer
    python generate_and_view.py --seed 1234      # reproduce a robot
    python generate_and_view.py --no-view        # only write the XML
"""

import argparse
import os
import secrets
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pybullet  # noqa: E402
import quadruped_generator  # noqa: E402

MESH_DIR = os.path.join(SCRIPT_DIR, "robot_descriptions", "a1_description", "meshes")
OUT_DIR = os.path.join(SCRIPT_DIR, "generated")

# Link order of create_robot (see the "Format of value below" comment there).
LINK_LABELS = ["imu"] + [
    f"{leg}_{part}"
    for leg in ("FR", "FL", "RR", "RL")
    for part in ("hip", "hip_fixed", "upper", "lower", "toe")
]

# Constants taken from GenLoco/motion_imitation/robots/random_quadruped.py,
# the robot class that consumes create_robot during training.
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * 4)
ABDUCTION_P_GAIN, ABDUCTION_D_GAIN = 100.0, 1.0
HIP_P_GAIN, HIP_D_GAIN = 100.0, 2.0
KNEE_P_GAIN, KNEE_D_GAIN = 100.0, 2.0
DEFAULT_MASS = 12.458
JOINT_RANGES = [  # ACTION_CONFIG bounds, per leg: abduction, hip, knee
    (-0.802851455917, 0.802851455917),
    (-1.0471975512, 4.18879020479),
    (-2.69653369433, -0.916297857297),
] * 4
TIME_STEP = 0.001
SPAWN_HEIGHT = 0.3  # lowest toe above the floor at spawn/reset [m]
MAX_VEL_PREFIX = "max_velocity_"

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


def shape_to_geom(shape, frame_prefix, length_key, mesh_names):
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
        file_name = os.path.basename(shape["fileName"])
        scale = tuple(shape.get("meshScale", (1, 1, 1)))
        key = (file_name, scale)
        if key not in mesh_names:
            mesh_names[key] = f"{os.path.splitext(file_name)[0]}_{len(mesh_names)}"
        attrs.update(type="mesh", mesh=mesh_names[key])
    else:
        raise ValueError(f"Unsupported PyBullet shape type {stype}")
    return attrs


def generate_morphology(delta_scale):
    """Run GenLoco's generator in headless PyBullet; return recorded data."""
    cid = pybullet.connect(pybullet.DIRECT)
    try:
        pybullet.setAdditionalSearchPath(SCRIPT_DIR)
        rec = RecordingClient(pybullet)
        uid = quadruped_generator.create_robot(rec, delta_scale=delta_scale)
        dynamics = {
            i: pybullet.getDynamicsInfo(uid, i)
            for i in range(-1, pybullet.getNumJoints(uid))
        }
        joint_names = {
            i: pybullet.getJointInfo(uid, i)[1].decode()
            for i in range(pybullet.getNumJoints(uid))
        }
    finally:
        pybullet.disconnect(cid)
    return rec, dynamics, joint_names


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


def add_geoms(body, rec, col_idx, vis_idx, friction, mesh_names):
    if vis_idx >= 0:
        vis = rec.visual_shapes[vis_idx]
        attrs = shape_to_geom(vis, "visual", "length", mesh_names)
        attrs.update(rgba=fmt(vis.get("rgbaColor", (1, 1, 1, 1))), **{"class": "visual"})
        ET.SubElement(body, "geom", attrs)
    if col_idx >= 0:
        col = rec.collision_shapes[col_idx]
        attrs = shape_to_geom(col, "collision", "height", mesh_names)
        attrs.update(friction=fmt((friction, 0.005, 0.0001)), **{"class": "collision"})
        ET.SubElement(body, "geom", attrs)


def add_hip_link_visual(hip_body, rec, mb, hip_index):
    """Draw the hip link as a cylinder from the hip joint to the thigh joint.

    Visual only: create_robot gives the hip link no shape spanning its
    (randomised) length. Radius and colour follow the thigh's visual.
    """
    thigh_pos = mb["linkPositions"][hip_index + 2]
    thigh_visual = rec.visual_shapes[mb["linkVisualShapeIndices"][hip_index + 2]]
    ET.SubElement(hip_body, "geom", {
        "type": "cylinder",
        "fromto": fmt((0, 0, 0, *thigh_pos)),
        "size": fmt(thigh_visual["radius"]),
        "rgba": fmt(thigh_visual.get("rgbaColor", (1, 1, 1, 1))),
        "class": "visual",
    })


def build_mjcf(rec, dynamics, joint_names, seed, out_dir):
    mb = rec.multibody_args
    total_mass = sum(d[0] for d in dynamics.values())
    gain_scale = total_mass / DEFAULT_MASS

    root = ET.Element("mujoco", model=f"genloco_random_quadruped_{seed}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false",
                  meshdir=os.path.relpath(MESH_DIR, out_dir))
    ET.SubElement(root, "option", timestep=fmt(TIME_STEP))

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.1", armature="0.01")
    vis_cls = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(vis_cls, "geom", contype="0", conaffinity="0", group="1")
    col_cls = ET.SubElement(default, "default", {"class": "collision"})
    # contype=1/conaffinity=0: robot collides with the floor but not itself,
    # matching PyBullet's createMultiBody default (no self-collision).
    ET.SubElement(col_cls, "geom", contype="1", conaffinity="0", group="3",
                  rgba="0.2 0.6 1 0.4")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")
    ET.SubElement(visual, "global", azimuth="140", elevation="-20")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="512")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid",
                  texrepeat="8 8", reflectance="0.2")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  material="grid", contype="0", conaffinity="1")

    mesh_names = {}
    base = ET.SubElement(world, "body", name="trunk", pos="0 0 0")
    ET.SubElement(base, "freejoint", name="root")
    add_inertial(base, dynamics[-1])
    add_geoms(base, rec, mb["baseCollisionShapeIndex"], mb["baseVisualShapeIndex"],
              dynamics[-1][1], mesh_names)

    # PyBullet linkParentIndices: 0 = base, k = link k-1.
    bodies = {0: base}
    actuated = []
    for i, label in enumerate(LINK_LABELS):
        parent = bodies[mb["linkParentIndices"][i]]
        body = ET.SubElement(parent, "body", name=label,
                             pos=fmt(mb["linkPositions"][i]),
                             quat=fmt(mj_quat(mb["linkOrientations"][i])))
        bodies[i + 1] = body
        jtype = mb["linkJointTypes"][i]
        if jtype == PYBULLET_REVOLUTE:
            lo, hi = JOINT_RANGES[len(actuated)]
            ET.SubElement(body, "joint", name=joint_names[i], type="hinge",
                          axis=fmt(mb["linkJointAxis"][i]), range=fmt((lo, hi)))
            actuated.append(joint_names[i])
        elif jtype != PYBULLET_FIXED:
            raise ValueError(f"Unsupported PyBullet joint type {jtype}")
        add_inertial(body, dynamics[i])
        add_geoms(body, rec, mb["linkCollisionShapeIndices"][i],
                  mb["linkVisualShapeIndices"][i], dynamics[i][1], mesh_names)
        if label.endswith("_hip"):
            add_hip_link_visual(body, rec, mb, i)

    for (file_name, scale), name in mesh_names.items():
        ET.SubElement(asset, "mesh", name=name, file=file_name, scale=fmt(scale))

    # MuJoCo has no joint velocity limit attribute; store it as custom data
    # (enforced by clamp_joint_velocities, like PyBullet's maxJointVelocity).
    custom = ET.SubElement(root, "custom")
    for i, vmax in sorted(rec.max_joint_velocities.items()):
        ET.SubElement(custom, "numeric", name=f"{MAX_VEL_PREFIX}{joint_names[i]}", data=fmt(vmax))

    gains = [(ABDUCTION_P_GAIN, ABDUCTION_D_GAIN), (HIP_P_GAIN, HIP_D_GAIN),
             (KNEE_P_GAIN, KNEE_D_GAIN)] * 4
    actuator = ET.SubElement(root, "actuator")
    for name, (kp, kd), (lo, hi) in zip(actuated, gains, JOINT_RANGES):
        ET.SubElement(actuator, "position", name=f"{name}_motor", joint=name,
                      kp=fmt(kp * gain_scale), kv=fmt(kd * gain_scale),
                      ctrlrange=fmt((lo, hi)))
    return root


def joint_velocity_limits(model):
    """(dof addresses, limits) from the max-velocity custom numerics."""
    import mujoco

    dofs, limits = [], []
    for n in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, n)
        if name.startswith(MAX_VEL_PREFIX):
            joint = model.joint(name[len(MAX_VEL_PREFIX):])
            dofs.append(joint.dofadr[0])
            limits.append(model.numeric_data[model.numeric_adr[n]])
    return np.array(dofs, dtype=int), np.array(limits)


def clamp_joint_velocities(model, data, dofs, limits):
    """Cap joint speeds without changing the robot's total momentum.

    Clipping joint velocities alone would inject spin into the floating base,
    so the base velocity is re-solved to keep the spatial momentum
    (first 6 rows of M @ qvel) what it was before the clip.
    """
    import mujoco

    clipped = np.clip(data.qvel[dofs], -limits, limits)
    if np.array_equal(clipped, data.qvel[dofs]):
        return
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass_matrix)
    momentum = mass_matrix[:6] @ data.qvel
    data.qvel[dofs] = clipped
    data.qvel[:6] = np.linalg.solve(mass_matrix[:6, :6],
                                    momentum - mass_matrix[:6, 6:] @ data.qvel[6:])


def set_spawn_height(root):
    """Lift the robot so its lowest toe is SPAWN_HEIGHT above the floor.

    Applies to both the default pose (qpos0, what a viewer reset restores)
    and the 'stand' keyframe in GenLoco's init pose.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"),
                                           assets=_mesh_assets(root))
    data = mujoco.MjData(model)
    toes = [g for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]]

    def lowest_toe(joint_angles):
        data.qpos[:] = model.qpos0
        data.qpos[7:] = joint_angles
        mujoco.mj_forward(model, data)
        return min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in toes)

    root.find("worldbody/body[@name='trunk']").set(
        "pos", fmt((0, 0, SPAWN_HEIGHT - lowest_toe(model.qpos0[7:]))))

    stand_z = SPAWN_HEIGHT - lowest_toe(INIT_MOTOR_ANGLES)
    qpos = data.qpos.copy()
    qpos[2] = stand_z
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="stand", qpos=fmt(qpos), ctrl=fmt(INIT_MOTOR_ANGLES))


def _mesh_assets(root):
    """Mesh bytes for compiling from a string (meshdir is relative to the file)."""
    return {
        m.get("file"): open(os.path.join(MESH_DIR, m.get("file")), "rb").read()
        for m in root.iter("mesh")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: fresh seed every run)")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="create_robot randomisation range scale (GenLoco default 1)")
    parser.add_argument("--out", default=None, help="output XML path")
    parser.add_argument("--no-view", action="store_true", help="do not open the viewer")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    np.random.seed(seed)  # create_robot draws from numpy's global RNG

    rec, dynamics, joint_names = generate_morphology(args.delta_scale)

    out_path = os.path.abspath(args.out or os.path.join(OUT_DIR, f"random_quadruped_{seed}.xml"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = build_mjcf(rec, dynamics, joint_names, seed, os.path.dirname(out_path))
    set_spawn_height(root)
    ET.indent(root)
    ET.ElementTree(root).write(out_path, encoding="unicode")

    total_mass = sum(d[0] for d in dynamics.values())
    print(f"seed: {seed}")
    print(f"total mass: {total_mass:.3f} kg")
    print(f"hip link length (hip -> thigh joint): {abs(rec.multibody_args['linkPositions'][2][1]):.4f} m")
    print(f"max joint velocity: {next(iter(rec.max_joint_velocities.values())):.3f} rad/s")
    print(f"wrote: {out_path}")

    if args.no_view:
        return

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(out_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    dofs, limits = joint_velocity_limits(model)
    print("Viewer: press '3' to toggle collision geoms; reset (Backspace) restarts from the 'stand' pose.")
    frame_dt = 1 / 60
    steps_per_frame = max(1, round(frame_dt / model.opt.timestep))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_time = data.time
        while viewer.is_running():
            frame_start = time.time()
            with viewer.lock():
                # The viewer's reset restores qpos0 (all joints at 0, knees
                # outside their range); restart from the 'stand' keyframe instead.
                if data.time < last_time:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                    clamp_joint_velocities(model, data, dofs, limits)
                last_time = data.time
            viewer.sync()
            time.sleep(max(0.0, frame_dt - (time.time() - frame_start)))


if __name__ == "__main__":
    main()
