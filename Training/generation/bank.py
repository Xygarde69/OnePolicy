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


def _posed(model: mujoco.MjModel, init_joint_pos: dict[str, float]) -> mujoco.MjData:
  """Kinematics of the robot in `init_joint_pos` with its base at z = 0."""
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  data.qpos[2] = 0.0
  for j in range(1, model.njnt):
    name = model.joint(j).name
    for pattern, value in init_joint_pos.items():
      if re.fullmatch(pattern, name):
        data.qpos[model.jnt_qposadr[j]] = value
  mujoco.mj_kinematics(model, data)
  return data


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
    # Exact for spheres; the bounding sphere is a safe bound for other shapes.
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
      return data.geom_xpos[g][2] - model.geom_size[g][0]
    return data.geom_xpos[g][2] - model.geom_rbound[g]

  return clearance - min(bottom(g) for g in feet)


def _leg_dimensions(model: mujoco.MjModel) -> tuple[float, float]:
  """Mean leg length (thigh joint -> knee -> foot centre) and foot radius [m].

  Measured in the zero pose, so it works for any robot in the Go2 naming
  convention (generated or adapted).
  """
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  lengths, radii = [], []
  for leg in ("FL", "FR", "RL", "RR"):
    hip = data.xanchor[model.joint(f"{leg}_thigh_joint").id]
    knee = data.xanchor[model.joint(f"{leg}_calf_joint").id]
    g = model.geom(f"{leg}_foot_collision").id
    lengths.append(np.linalg.norm(knee - hip) + np.linalg.norm(data.geom_xpos[g] - knee))
    radii.append(model.geom_size[g][0] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                 else model.geom_rbound[g])
  return float(np.mean(lengths)), float(np.mean(radii))


def _extract(model: mujoco.MjModel, init_joint_pos: dict[str, float],
             clearance: float):
  """Per-env fields, structure fields and per-robot scalars of one compiled robot."""
  fields = {f: getattr(model, f)[1:] if f in BODY_FIELDS else getattr(model, f)
            for f in MORPHOLOGY_FIELDS}
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
               scalars: dict[str, list], fields) -> Path:
  """`scalars`: total_mass, max_joint_velocity, root_height, base_above_feet,
  leg_length, foot_radius, one value per robot."""
  bank_path = out_dir / BANK_FILE
  np.savez(
    bank_path,
    seeds=np.array(seeds),
    **{k: np.array(v) for k, v in scalars.items()},
    body_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_BODY, reference.nbody)[1:]),
    geom_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_GEOM, reference.ngeom)),
    site_names=np.array(_names(reference, mujoco.mjtObj.mjOBJ_SITE, reference.nsite)),
    **{f: np.stack(v) for f, v in fields.items()},
  )
  return bank_path


def single_robot_bank(xml: str, assets: dict[str, bytes], out_dir: Path,
                      init_joint_pos: dict[str, float],
                      clearance: float = 0.02,
                      max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY) -> Path:
  """Bank holding one existing robot (see adapt.py), for play.

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
  return _save_bank(out_dir, model, [-1], {k: [v] for k, v in scalars.items()},
                    {f: [v] for f, v in fields.items()})


def generate_bank(num_envs: int, out_dir: Path, cfg: MorphologyGenCfg,
                  init_joint_pos: dict[str, float]) -> Path:
  """Generate, compile and validate one robot per env.

  Writes `out_dir/env_XXXXX.xml` (robot of env i) and `out_dir/morphologies.npz`
  (per-env fields). Returns the path of the .npz.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  base_seed = cfg.seed if cfg.seed is not None else secrets.randbits(31)
  jobs = [(base_seed + i, cfg.delta_scale, init_joint_pos, cfg.foot_clearance)
          for i in range(num_envs)]
  workers = cfg.workers or os.cpu_count() or 1

  print(f"[INFO] Generating {num_envs} random quadrupeds "
        f"(base seed {base_seed}, {workers} workers)...", flush=True)
  if workers == 1 or num_envs == 1:
    results = map(_make_one, jobs)
  else:
    pool = ProcessPoolExecutor(max_workers=workers)
    results = pool.map(_make_one, jobs, chunksize=max(1, num_envs // (workers * 8)))

  reference = None
  seeds = []
  scalars: dict[str, list] = {}
  fields = {f: [] for f in MORPHOLOGY_FIELDS}
  ref_structure = None
  for i, (seed, xml, env_scalars, env_fields, structure) in enumerate(results):
    if ref_structure is None:
      ref_structure = structure
      reference = _compile(xml)
    for f in STRUCTURE_FIELDS:
      if not np.array_equal(structure[f], ref_structure[f]):
        raise RuntimeError(
          f"Robot of env {i} (seed {seed}) differs from env 0 in '{f}'; "
          "all envs must share one model structure.")
    (out_dir / f"env_{i:05d}.xml").write_text(xml)
    seeds.append(seed)
    for k, v in env_scalars.items():
      scalars.setdefault(k, []).append(v)
    for f in MORPHOLOGY_FIELDS:
      fields[f].append(env_fields[f])
    if (i + 1) % 1024 == 0 or i + 1 == num_envs:
      print(f"[INFO]   compiled {i + 1}/{num_envs}", flush=True)
  if workers > 1 and num_envs > 1:
    pool.shutdown()

  bank_path = _save_bank(out_dir, reference, seeds, scalars, fields)

  def span(key: str, fmt: str) -> str:
    return f"{min(scalars[key]):{fmt}}-{max(scalars[key]):{fmt}}"

  print(f"[INFO] Wrote {num_envs} robots to {out_dir} "
        f"(mass {span('total_mass', '.1f')} kg, "
        f"leg length {span('leg_length', '.2f')} m, "
        f"max joint velocity {span('max_joint_velocity', '.1f')} rad/s, "
        f"spawn height {span('root_height', '.2f')} m)", flush=True)
  return bank_path
