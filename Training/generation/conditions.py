"""Conditions on a generated robot's morphology (MorphologyGenCfg.condition).

A condition is a Python-like expression over the VARIABLES below, e.g.

  leg_length > 0.3 and mass < 40
  abs(com_x) < 0.02 and kp_calf < 200
  thigh_length / calf_length > 1.1

Only numbers, the variable names, arithmetic, comparisons, and/or/not,
if/else and the functions in FUNCTIONS are allowed: it is parsed and checked
here, never passed to eval().
"""

from __future__ import annotations

import ast
import math
import operator

import mujoco
import numpy as np

# name -> what it is. Lengths in m, masses in kg, gains in Nm/rad and Nm s/rad.
VARIABLES = {
  "mass": "total mass [kg]",
  "base_mass": "base (trunk) mass [kg]",
  "leg_length": "mean thigh joint -> knee -> foot length [m]",
  "thigh_length": "thigh joint -> knee [m]",
  "calf_length": "knee -> foot centre [m]",
  "hip_link": "hip joint -> thigh joint [m]",
  "base_length": "base box length (x) [m]",
  "base_width": "base box width (y) [m]",
  "base_height": "base box height (z) [m]",
  "foot_radius": "foot sphere radius [m]",
  "stand_height": "base above the feet in the default pose [m]",
  "com_x": "whole-robot centre of mass in the base frame, x [m]",
  "com_y": "whole-robot centre of mass in the base frame, y [m]",
  "com_z": "whole-robot centre of mass in the base frame, z [m]",
  "s_L": "leg length / Go2's",
  "s_m": "mass / Go2's",
  "kp_hip": "hip PD stiffness after size scaling [Nm/rad]",
  "kp_thigh": "thigh PD stiffness after size scaling [Nm/rad]",
  "kp_calf": "knee PD stiffness after size scaling [Nm/rad]",
  "kd_hip": "hip PD damping after size scaling [Nm s/rad]",
  "kd_calf": "knee PD damping after size scaling [Nm s/rad]",
  "torque_hip": "hip / thigh torque limit after size scaling [Nm]",
  "torque_calf": "knee torque limit after size scaling [Nm]",
  "max_joint_velocity": "joint speed limit [rad/s]",
  "x_knee": "1 if rear knees bend forward (><), else 0",
}

FUNCTIONS = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log}

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}


class ConditionError(ValueError):
  pass


def parse_condition(text: str) -> ast.Expression:
  """Parse and check `text`; raises ConditionError naming the problem."""
  try:
    tree = ast.parse(text.strip(), mode="eval")
  except SyntaxError as e:
    raise ConditionError(f"Syntax error in condition: {e.msg} (column {e.offset})")
  for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id not in VARIABLES and node.id not in FUNCTIONS:
      raise ConditionError(f"Unknown name '{node.id}'. Variables: {', '.join(VARIABLES)}")
    if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                           and node.func.id in FUNCTIONS and not node.keywords):
      raise ConditionError(f"Only {', '.join(FUNCTIONS)} can be called.")
    allowed = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp,
               ast.Call, ast.Name, ast.Constant, ast.Load, ast.And, ast.Or,
               *_BINARY, *_COMPARE, *_UNARY)
    if not isinstance(node, allowed):
      raise ConditionError(f"'{type(node).__name__}' is not allowed in a condition.")
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
      raise ConditionError("Only numbers are allowed as constants.")
  return tree


def evaluate(tree: ast.Expression, variables: dict[str, float]) -> bool:
  def ev(node):
    if isinstance(node, ast.Expression):
      return ev(node.body)
    if isinstance(node, ast.Constant):
      return node.value
    if isinstance(node, ast.Name):
      return variables[node.id] if node.id in VARIABLES else FUNCTIONS[node.id]
    if isinstance(node, ast.BoolOp):
      values = (ev(v) for v in node.values)
      return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.BinOp):
      return _BINARY[type(node.op)](ev(node.left), ev(node.right))
    if isinstance(node, ast.UnaryOp):
      return _UNARY[type(node.op)](ev(node.operand))
    if isinstance(node, ast.Compare):
      left = ev(node.left)
      for op, comparator in zip(node.ops, node.comparators):
        right = ev(comparator)
        if not _COMPARE[type(op)](left, right):
          return False
        left = right
      return True
    if isinstance(node, ast.IfExp):
      return ev(node.body) if ev(node.test) else ev(node.orelse)
    if isinstance(node, ast.Call):
      return ev(node.func)(*(ev(a) for a in node.args))
    raise ConditionError(f"'{type(node).__name__}' is not allowed in a condition.")

  try:
    return bool(ev(tree))
  except (ArithmeticError, ValueError) as e:
    raise ConditionError(f"Condition failed to evaluate: {e}")


def robot_variables(model: mujoco.MjModel, scalars: dict[str, float]) -> dict[str, float]:
  """VARIABLES of one compiled robot in the Go2 naming convention; `scalars`
  are bank._extract's plus total_mass and max_joint_velocity."""
  from assets.morph_quadruped.constants import (
    MORPH_ACTUATOR_CALF,
    MORPH_ACTUATOR_HIP,
    MORPH_ACTUATOR_THIGH,
    NOMINAL_LEG_LENGTH,
    NOMINAL_MASS,
  )

  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)

  def anchor(joint: str) -> np.ndarray:
    return data.xanchor[model.joint(joint).id]

  foot = data.geom_xpos[model.geom("FR_foot_collision").id]
  base = 1
  box = next((g for g in range(model.ngeom) if model.geom_bodyid[g] == base
              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX), None)
  extents = 2 * model.geom_size[box] if box is not None else np.zeros(3)
  com = data.xmat[base].reshape(3, 3).T @ (data.subtree_com[base] - data.xpos[base])

  mass, leg_length = float(scalars["total_mass"]), float(scalars["leg_length"])
  s_m, s_L = mass / NOMINAL_MASS, leg_length / NOMINAL_LEG_LENGTH
  torque, damping = s_m * s_L, s_m * s_L**1.5
  return {
    "mass": mass,
    "base_mass": float(model.body_mass[base]),
    "leg_length": leg_length,
    "thigh_length": float(np.linalg.norm(anchor("FR_calf_joint") - anchor("FR_thigh_joint"))),
    "calf_length": float(np.linalg.norm(foot - anchor("FR_calf_joint"))),
    "hip_link": float(np.linalg.norm(anchor("FR_thigh_joint") - anchor("FR_hip_joint"))),
    "base_length": float(extents[0]),
    "base_width": float(extents[1]),
    "base_height": float(extents[2]),
    "foot_radius": float(scalars["foot_radius"]),
    "stand_height": float(scalars["base_above_feet"]),
    "com_x": float(com[0]), "com_y": float(com[1]), "com_z": float(com[2]),
    "s_L": s_L, "s_m": s_m,
    "kp_hip": MORPH_ACTUATOR_HIP.stiffness * torque,
    "kp_thigh": MORPH_ACTUATOR_THIGH.stiffness * torque,
    "kp_calf": MORPH_ACTUATOR_CALF.stiffness * torque,
    "kd_hip": MORPH_ACTUATOR_HIP.damping * damping,
    "kd_calf": MORPH_ACTUATOR_CALF.damping * damping,
    "torque_hip": MORPH_ACTUATOR_HIP.effort_limit * torque,
    "torque_calf": MORPH_ACTUATOR_CALF.effort_limit * torque,
    "max_joint_velocity": float(scalars["max_joint_velocity"]),
    "x_knee": float(scalars["x_knee"]),
  }
