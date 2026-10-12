"""Procedurally generated quadruped constants.

Mirrors Unitree Go2's constants (actuators, collisions, initial state) so the
generated robots are drop-in replacements: `get_morph_robot_cfg()` takes the
place of `get_go2_robot_cfg()`. Names follow Go2 (see generation/mjcf_builder.py).
"""

from dataclasses import replace
from functools import partial
from pathlib import Path

import mujoco

from generation.mjcf_builder import mesh_assets
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##


def get_spec(xml_path: str) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(xml_path)
  assets = mesh_assets()
  # Robots adapted for play (generation/adapt.py) bundle their own files.
  asset_dir = Path(xml_path).parent / spec.meshdir
  if spec.meshdir and asset_dir.is_dir():
    for f in asset_dir.rglob("*"):
      if f.is_file():
        assets[f"{spec.meshdir}/{f.relative_to(asset_dir)}"] = f.read_bytes()
  spec.assets = assets
  return spec


##
# Actuator config (Go2 values).
##

# Go2 reference anatomy. Every Go2-tuned threshold (rewards, terminations,
# commands, randomisation) and the actuators below are rescaled per robot by
# its size relative to these (see tasks/velocity/mdp/scaling.py).
NOMINAL_MASS = 15.206
"""Go2 total mass [kg]."""
NOMINAL_LEG_LENGTH = 0.426
"""Go2 thigh joint -> knee -> foot centre [m] (0.213 + 0.213)."""
NOMINAL_FOOT_RADIUS = 0.022
"""Go2 foot sphere radius [m]."""
NOMINAL_BASE_ABOVE_FEET = 0.30
"""Placeholder base height above the feet [m] until the `morphology` event sets
each robot's own (generation/bank.py, _base_above_feet)."""

# DC motors rather than MuJoCo position actuators, so that each robot's joint
# velocity limit (create_robot's maxJointVelocity) holds: torque falls linearly
# from effort_limit at rest to zero at velocity_limit. The `morphology` startup
# event overwrites velocity_limit with each robot's own value.
PLACEHOLDER_VELOCITY_LIMIT = 30.0

MORPH_ACTUATOR_HIP = DcMotorActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=PLACEHOLDER_VELOCITY_LIMIT,
  armature=0.01,
)
MORPH_ACTUATOR_THIGH = DcMotorActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=PLACEHOLDER_VELOCITY_LIMIT,
  armature=0.01,
)
MORPH_ACTUATOR_CALF = DcMotorActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  saturation_effort=45,
  velocity_limit=PLACEHOLDER_VELOCITY_LIMIT,
  armature=0.02,
)

##
# Keyframes.
##

INIT_JOINT_POS = {
  ".*thigh_joint": 0.9,
  ".*calf_joint": -1.8,
  ".*R_hip_joint": 0.1,
  ".*L_hip_joint": -0.1,
}

# The base height here is a placeholder: each env's spawn height is set from
# its own leg lengths by the `morphology` startup event.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos=INIT_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collision config (Go2 values).
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

MORPH_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MORPH_ACTUATOR_HIP,
    MORPH_ACTUATOR_THIGH,
    MORPH_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_morph_robot_cfg(
  template_xml: str | Path, init_joint_pos: dict[str, float] | None = None
) -> EntityCfg:
  """Get a fresh generated-quadruped configuration instance.

  `template_xml` is any generated robot (env 0's by convention). It fixes the
  model structure; per-env geometry and inertia come from the `morphology`
  startup event. `init_joint_pos` overrides the default pose (INIT_JOINT_POS).
  """
  init_state = INIT_STATE
  if init_joint_pos is not None:
    init_state = replace(INIT_STATE, joint_pos=init_joint_pos)
  return EntityCfg(
    init_state=init_state,
    collisions=(FULL_COLLISION,),
    spec_fn=partial(get_spec, str(template_xml)),
    articulation=MORPH_ARTICULATION,
  )
