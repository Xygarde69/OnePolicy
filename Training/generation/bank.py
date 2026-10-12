"""Generate and compile one random quadruped per environment.

mjwarp simulates every env against a single model *structure*, so all robots
must share one kinematic tree (same bodies, joints, geoms and geom types).
create_robot guarantees that: it only randomises numbers (lengths, radii,
masses, inertias). Each robot is compiled into its own MjModel, checked against
env 0, and its numeric fields are stored per env. At env startup, the
`apply_morphologies` event writes them into the batched mjwarp model, so env i
simulates robot i.
"""

from __future__ import annotations

import os
import re
import secrets
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import mujoco
import numpy as np

from .mjcf_builder import make_robot_xml, mesh_assets

# Per-env numeric fields copied from each compiled robot into its env.
BODY_FIELDS = ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos",
               "body_iquat")
GEOM_FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_aabb")
SITE_FIELDS = ("site_pos", "site_quat")
MORPHOLOGY_FIELDS = BODY_FIELDS + GEOM_FIELDS + SITE_FIELDS
# Per-env joint data, over the hinge joints in model order (bank "joint_names"):
# each robot's own joint limits and default pose (e.g. knee-forward legs).
JOINT_ARRAYS = ("jnt_range", "default_joint_pos")

# Fields that define structure; they must be identical across all robots.
STRUCTURE_FIELDS = ("body_parentid", "body_sameframe", "body_simple", "jnt_type",
                    "jnt_bodyid", "dof_simplenum", "geom_type", "geom_bodyid",
                    "geom_sameframe", "geom_contype", "geom_conaffinity",
                    "site_bodyid", "site_sameframe")

BANK_FILE = "morphologies.npz"

# Joint velocity limit [rad/s] for robots not made by create_robot (adapt.py):
# the same fixed max_joint_velocity create_robot gives every robot.
DEFAULT_MAX_JOINT_VELOCITY = 6 * np.pi


@dataclass(frozen=True)
class MorphologyGenCfg:
  """Procedural generation settings."""

  seed: int | None = None
  """Base seed; env i uses seed + i. None draws a fresh one."""
  delta_scale: float = 1.0
  """create_robot randomisation range scale (GenLoco default 1)."""
  foot_clearance: float = 0.02
  """Gap between the lowest foot and the floor at spawn [m]."""
  workers: int = 0
  """Generation processes. 0 uses all CPUs."""


def _names(model: mujoco.MjModel, obj: mujoco.mjtObj, n: int) -> list[str]:
  return [mujoco.mj_id2name(model, obj, i) for i in range(n)]


def _compile(xml: str) -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(xml, mesh_assets())


def _resolved_pose(model: mujoco.MjModel, init_joint_pos: dict[str, float]) -> dict[str, float]:
  """Joint name -> value of a {regex or name: value} pose, over the hinges."""
  pose = {}
  for j in range(1, model.njnt):
    name = model.joint(j).name
    for pattern, value in init_joint_pos.items():
      if re.fullmatch(pattern, name):
        pose[name] = value
        break
  return pose


def _posed(model: mujoco.MjModel, init_joint_pos: dict[str, float]) -> mujoco.MjData:
  """Kinematics of the robot in `init_joint_pos` with its base at z = 0."""
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  data.qpos[2] = 0.0
  for name, value in _resolved_pose(model, init_joint_pos).items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
  mujoco.mj_kinematics(model, data)
  return data


def foot_point(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> tuple[np.ndarray, float]:
  """World "foot centre" of foot geom g and the radius below it.

  Sphere: its centre and radius (the generated robots' and Go2's convention).
  Capsule / cylinder: centre of its lower end cap and its radius. Mesh: its
  lowest vertex, radius 0 (a calf mesh ending in the foot). Other shapes: the
  lowest point of the bounding box below the geom centre, radius 0. Assumes
  the legs hang down (zero / default pose).
  """
  pos = data.geom_xpos[g].copy()
  rot = data.geom_xmat[g].reshape(3, 3)
  kind = model.geom_type[g]
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    return pos, float(model.geom_size[g][0])
  if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
    ends = [pos + s * model.geom_size[g][1] * rot[:, 2] for s in (-1.0, 1.0)]
    return min(ends, key=lambda p: p[2]), float(model.geom_size[g][0])
  if kind == mujoco.mjtGeom.mjGEOM_MESH:
    m = model.geom_dataid[g]
    verts = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    world = pos + verts @ rot.T
    return world[np.argmin(world[:, 2])], 0.0
  center, half = model.geom_aabb[g].reshape(2, 3)
  corners = np.array([center + half * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])
  low = (pos + corners @ rot.T)[:, 2].min()
  return np.array([pos[0], pos[1], low]), 0.0


def _base_above_feet(model: mujoco.MjModel, init_joint_pos: dict[str, float]) -> float:
  """Base height above the mean foot-site height in the default pose [m].

  The base_height reward's target: measured against the feet rather than the
  floor, so it holds on any terrain.
  """
  data = _posed(model, init_joint_pos)
  feet = [model.site(leg).id for leg in ("FL", "FR", "RL", "RR")]
  return float(-np.mean([data.site_xpos[s][2] for s in feet]))


def _standing_height(model: mujoco.MjModel, init_joint_pos: dict[str, float],
                     clearance: float) -> float:
  """Base height that puts the lowest foot `clearance` above the floor."""
  data = _posed(model, init_joint_pos)
  feet = [model.geom(f"{leg}_foot_collision").id for leg in ("FL", "FR", "RL", "RR")]

  def bottom(g):
    point, radius = foot_point(model, data, g)
    return point[2] - radius

  return clearance - min(bottom(g) for g in feet)


def _leg_dimensions(model: mujoco.MjModel) -> tuple[float, float]:
  """Mean leg length (thigh joint -> knee -> foot centre) and foot radius [m].

  The foot centre is `foot_point`'s (a calf mesh's lowest vertex, not its
  middle).

  Measured in the zero pose, so it works for any robot in the Go2 naming
  convention (generated or adapted).
  """
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  lengths, radii = [], []
  for leg in ("FL", "FR", "RL", "RR"):
    hip = data.xanchor[model.joint(f"{leg}_thigh_joint").id]
    knee = data.xanchor[model.joint(f"{leg}_calf_joint").id]
    foot, radius = foot_point(model, data, model.geom(f"{leg}_foot_collision").id)
    lengths.append(np.linalg.norm(knee - hip) + np.linalg.norm(foot - knee))
    radii.append(radius)
  return float(np.mean(lengths)), float(np.mean(radii))


def _extract(model: mujoco.MjModel, init_joint_pos: dict[str, float],
             clearance: float):
  """Per-env fields, structure fields and per-robot scalars of one compiled robot."""
  fields = {f: getattr(model, f)[1:] if f in BODY_FIELDS else getattr(model, f)
            for f in MORPHOLOGY_FIELDS}
  hinges = range(1, model.njnt)  # Joint 0 is the free joint.
  fields["jnt_range"] = model.jnt_range[1:].copy()
  pose = _resolved_pose(model, init_joint_pos)
  fields["default_joint_pos"] = np.array([pose.get(model.joint(j).name, 0.0)
                                          for j in hinges])
  structure = {f: getattr(model, f).copy() for f in STRUCTURE_FIELDS}
  leg_length, foot_radius = _leg_dimensions(model)
  scalars = {
    "root_height": _standing_height(model, init_joint_pos, clearance),
    "base_above_feet": _base_above_feet(model, init_joint_pos),
    "leg_length": leg_length,
    "foot_radius": foot_radius,
  }
  return fields, structure, scalars


def _make_one(args):
  seed, delta_scale, init_joint_pos, clearance = args
  xml, total_mass, max_vel = make_robot_xml(seed, delta_scale)
  fields, structure, scalars = _extract(_compile(xml), init_joint_pos, clearance)
  scalars.update(total_mass=total_mass, max_joint_velocity=max_vel)
  return seed, xml, scalars, fields, structure


def _save_bank(out_dir: Path, reference: mujoco.MjModel, seeds,
               scalars: dict[str, list], fields, sources=None) -> Path:
  """`scalars`: total_mass, max_joint_velocity, root_height, base_above_feet,
  leg_length, foot_radius, one value per robot."""
  bank_path = out_dir / BANK_FILE
  np.savez(
    bank_path,
    seeds=np.array(seeds),
    sources=np.array(sources if sources is not None else ["generated"] * len(seeds)),
    **{k: np.array(v) for k, v in scalars.items()},
    body_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_BODY, reference.nbody)[1:]),
    geom_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_GEOM, reference.ngeom)),
    site_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_SITE, reference.nsite)),
    joint_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_JOINT, reference.njnt)[1:]),
    **{f: np.stack(v) for f, v in fields.items()},
  )
  return bank_path


def single_robot_bank(xml: str, assets: dict[str, bytes], out_dir: Path,
                      init_joint_pos: dict[str, float],
                      clearance: float = 0.02,
                      max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY,
                      copies: int = 1, source: str = "generated") -> Path:
  """Bank holding `copies` envs of one existing robot (see adapt.py): one for
  play, many for training on a real robot's own model (train.py --generalise).

  `assets` are written under the MJCF's meshdir, next to env_00000.xml.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  xml_path = out_dir / "env_00000.xml"
  xml_path.write_text(xml)
  meshdir = mujoco.MjSpec.from_string(xml).meshdir
  for name, data in assets.items():
    path = out_dir / meshdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  fields, _, scalars = _extract(model, init_joint_pos, clearance)
  scalars.update(total_mass=model.body_subtreemass[1],
                 max_joint_velocity=max_joint_velocity)
  return _save_bank(out_dir, model, [-1] * copies,
                    {k: [v] * copies for k, v in scalars.items()},
                    {f: [v] * copies for f, v in fields.items()}, [source] * copies)


def generate_bank(num_envs: int, out_dir: Path, cfg: MorphologyGenCfg,
                  init_joint_pos: dict[str, float],
                  proxies: dict[str, int] | None = None) -> Path:
  """Generate, compile and validate one robot per env.

  The first num_envs - sum(proxies) envs get random generated robots; then
  `proxies` (real robot name -> number of envs) adds that many copies of a
  proxy of each real robot (proxy.py: the real robot's numbers in the
  generated structure, with its own joint ranges and default pose).

  Writes `out_dir/env_XXXXX.xml` (robot of env i) and `out_dir/morphologies.npz`
  (per-env fields; "sources" says "generated" or the real robot's name).
  Returns the path of the .npz.
  """
  proxies = {k: v for k, v in (proxies or {}).items() if v > 0}
  num_generated = num_envs - sum(proxies.values())
  if num_generated < 1:
    raise ValueError(f"{num_envs} envs leave no room for generated robots "
                     f"next to {sum(proxies.values())} proxies.")
  out_dir.mkdir(parents=True, exist_ok=True)
  base_seed = cfg.seed if cfg.seed is not None else secrets.randbits(31)
  jobs = [(base_seed + i, cfg.delta_scale, init_joint_pos, cfg.foot_clearance)
          for i in range(num_generated)]
  workers = cfg.workers or os.cpu_count() or 1

  print(f"[INFO] Generating {num_generated} random quadrupeds "
        f"(base seed {base_seed}, {workers} workers)...", flush=True)
  if workers == 1 or num_generated == 1:
    results = map(_make_one, jobs)
  else:
    pool = ProcessPoolExecutor(max_workers=workers)
    results = pool.map(_make_one, jobs,
                       chunksize=max(1, num_generated // (workers * 8)))

  def generated():
    for result in results:
      yield ("generated", *result)

  def proxy_results():
    from .proxy import make_proxy  # Imports adapt.py; only needed here.
    for name, count in proxies.items():
      *result, notes = make_proxy(name, init_joint_pos, cfg.foot_clearance)
      print(f"[INFO] Proxy of {name} x {count}" + "".join(f"\n[INFO]   {n}" for n in notes),
            flush=True)
      yield from [tuple(result)] * count

  reference = None
  seeds = []
  sources = []
  scalars: dict[str, list] = {}
  fields: dict[str, list] = {}
  ref_structure = None
  for i, (source, seed, xml, env_scalars, env_fields, structure) in enumerate(
      chain(generated(), proxy_results())):
    if ref_structure is None:
      ref_structure = structure
      reference = _compile(xml)
    for f in STRUCTURE_FIELDS:
      if not np.array_equal(structure[f], ref_structure[f]):
        raise RuntimeError(
          f"Robot of env {i} ({source}, seed {seed}) differs from env 0 in '{f}'; "
          "all envs must share one model structure.")
    (out_dir / f"env_{i:05d}.xml").write_text(xml)
    seeds.append(seed)
    sources.append(source)
    for k, v in env_scalars.items():
      scalars.setdefault(k, []).append(v)
    for f, v in env_fields.items():
      fields.setdefault(f, []).append(v)
    if (i + 1) % 1024 == 0 or i + 1 == num_envs:
      print(f"[INFO]   compiled {i + 1}/{num_envs}", flush=True)
  if workers > 1 and num_generated > 1:
    pool.shutdown()

  bank_path = _save_bank(out_dir, reference, seeds, scalars, fields, sources)

  def span(key: str, fmt: str) -> str:
    return f"{min(scalars[key]):{fmt}}-{max(scalars[key]):{fmt}}"

  print(f"[INFO] Wrote {num_envs} robots to {out_dir} "
        f"(mass {span('total_mass', '.1f')} kg, "
        f"leg length {span('leg_length', '.2f')} m, "
        f"max joint velocity {span('max_joint_velocity', '.1f')} rad/s, "
        f"spawn height {span('root_height', '.2f')} m)", flush=True)
  return bank_path
