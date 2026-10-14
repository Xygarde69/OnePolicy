"""Knee configuration: ">>" or "><".

">>": every knee bends backward (Go2, A1, Spot). "><": the rear knees bend
forward, towards the front legs (ANYmal). A robot is "><" when its name
carries X_KNEE_TAG (or a literal "><"): a real robot's MJCF file name
(quadrupeds/anybotics_anymal_c/anymal_c_xknee.xml), a generated robot's
MJCF model name. Its default pose is then the training pose with the rear
thigh and calf angles negated, which mirrors the rear legs front-to-back.
"""

from __future__ import annotations

import re

X_KNEE_TAG = "xknee"


def is_x_knee(name: str) -> bool:
  """Whether a robot's name marks it "><"."""
  return X_KNEE_TAG in name.lower() or "><" in name


def x_knee_pose(pose: dict[str, float]) -> dict[str, float]:
  """A {regex or name: value} ">>" pose turned "><": rear thigh and calf
  negated. The rear patterns come first so they win the first match."""
  rear = {}
  for part in ("thigh", "calf"):
    value = next(v for p, v in pose.items() if re.fullmatch(p, f"RR_{part}_joint"))
    rear[f"R[RL]_{part}_joint"] = -value
  return {**rear, **pose}
