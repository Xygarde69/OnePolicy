"""Procedurally generated quadruped velocity environment configurations.

Copied from the Unitree Go2 configuration; only the robot differs. Every env
simulates its own generated robot (see `set_morphologies`).
"""

import re
from dataclasses import dataclass
from typing import Literal

from pathlib import Path

from assets.morph_quadruped.constants import (
  NOMINAL_LEG_LENGTH,
  NOMINAL_MASS,
  get_morph_robot_cfg,
  get_spec,
)
from generation.bank import BANK_FILE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)

import tasks.velocity.mdp as morph_mdp
from tasks.velocity.mdp import UniformVelocityCommandCfg
from tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]

# One downward ray per foot site, named FOOT_RAY_PREFIX + site name: the ground
# height under each foot, so foot clearance is measured from the terrain, not
# from z = 0 (mdp.foot_heights_above_ground).
FOOT_RAY_PREFIX = "foot_ray_"


@dataclass
class _RaisedRayCfg(GridPatternCfg):
  """A single downward ray starting `height` [m] above its frame.

  A foot sunk into a stair box would start its ray inside the box, which the
  ray then ignores. The terrain has no overhangs, so starting well above the
  foot hits the surface under it.
  """

  height: float = 1.0

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] = self.height
    return offsets, directions


def morph_quadruped_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create generated-quadruped rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # mujoco_warp's EPA scratch scales with num_envs * nconmax * ccd_iterations
  # (~130 B each): 500 iterations needs ~9.5 GB at 4096 envs, 50 needs ~1 GB.
  cfg.sim.mujoco.ccd_iterations = 50
  # Generated robots can start with more contacts than the Go1's 35.
  cfg.sim.nconmax = 64
  cfg.sim.contact_sensor_maxmatch = 500

  # Robot structure and per-env robots are filled in by set_morphologies().
  cfg.scene.entities = {"robot": get_morph_robot_cfg(template_xml="")}
  # Must run before the other startup events (base_com reads body_ipos).
  cfg.events = {
    "morphology": EventTermCfg(
      func=morph_mdp.apply_morphologies,
      mode="startup",
      params={
        "bank_file": "",
        "nominal_mass": NOMINAL_MASS,
        "nominal_leg_length": NOMINAL_LEG_LENGTH,
      },
    ),
    **cfg.events,
  }

  # Set raycast sensor frame to base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms, and the shin geoms ending in them that
      # robots adapted for play may have (see generation/adapt.py).
      exclude=(*geom_names, r"(FR|FL|RR|RL)_shin_\d+_collision$"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  foot_ray_cfgs = tuple(
    RayCastSensorCfg(
      name=f"{FOOT_RAY_PREFIX}{site}",
      frame=ObjRef(type="site", name=site, entity="robot"),
      pattern=_RaisedRayCfg(size=(0.0, 0.0)),
      ray_alignment="world",
      max_distance=21.0,
      include_geom_groups=(0,),  # Terrain only: the foot starts inside its calf's visual.
    )
    for site in site_names
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    *foot_ray_cfgs,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
  cfg.observations["critic"].terms["foot_height"].params["ray_prefix"] = FOOT_RAY_PREFIX

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["ray_prefix"] = FOOT_RAY_PREFIX
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_contact_velocity"].params["asset_cfg"].site_names = site_names
  cfg.rewards["base_height"].params["asset_cfg"].site_names = site_names
  cfg.rewards["stance_width"].params["hip_cfg"].body_names = tuple(
    f"{name}_thigh" for name in foot_names)
  cfg.rewards["stance_width"].params["foot_cfg"].site_names = site_names

  # Go2's threshold [N], scaled per robot by mass.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=morph_mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    # No curriculum in play, so commands use their full range.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.initial_level = 1.0
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def morph_quadruped_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create generated-quadruped flat terrain velocity configuration."""
  cfg = morph_quadruped_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove the raycast sensors and height scan (no terrain to scan); foot
  # heights are above z = 0.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ())
    if s.name != "terrain_scan" and not s.name.startswith(FOOT_RAY_PREFIX)
  )
  cfg.observations["critic"].terms["foot_height"].params["ray_prefix"] = None
  cfg.rewards["foot_clearance"].params["ray_prefix"] = None
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def set_morphologies(
  cfg: ManagerBasedRlEnvCfg,
  morphology_dir: Path,
  init_joint_pos: dict[str, float] | None = None,
) -> None:
  """Point the env at a robot bank (see generation/bank.py).

  Env 0's robot provides the model structure; every env then gets its own
  robot from the bank at startup.
  """
  cfg.scene.entities["robot"] = get_morph_robot_cfg(
    morphology_dir / "env_00000.xml", init_joint_pos
  )
  cfg.events["morphology"].params["bank_file"] = str(morphology_dir / BANK_FILE)


def drop_body_contact_if_unmatched(env_cfg, xml_path: Path) -> None:
  """Remove non-foot ground contact sensing if the robot has nothing to sense.

  Robots whose only collision geoms are feet and shins (e.g. a model with a
  visual-only base and thighs) leave the nonfoot_ground_touch sensor with no
  geoms, which mjlab rejects. Play then drops the sensor, the illegal_contact
  termination and the nonfoot_contact reward.
  """
  sensor = next(s for s in env_cfg.scene.sensors if s.name == "nonfoot_ground_touch")
  patterns = sensor.primary.pattern
  patterns = (patterns,) if isinstance(patterns, str) else patterns
  excludes = sensor.primary.exclude
  # get_spec supplies the meshes training uses (generated robots' hip.obj is
  # not written next to their XML) as well as an adapted robot's own.
  model = get_spec(str(xml_path)).compile()
  names = [model.geom(i).name for i in range(model.ngeom)]
  matched = [n for n in names if any(re.fullmatch(p, n) for p in patterns)
             and not any(re.fullmatch(e, n) for e in excludes)]
  if matched:
    return
  env_cfg.scene.sensors = tuple(s for s in env_cfg.scene.sensors if s is not sensor)
  env_cfg.terminations.pop("illegal_contact", None)
  env_cfg.rewards.pop("nonfoot_contact", None)
  print("[WARN]: The robot's only collision geoms are feet and shins: base/thigh "
        "ground contact is not detected (no illegal_contact termination).")
