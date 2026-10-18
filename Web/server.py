"""OnePolicy web harness: pick, generate or upload a quadruped, train it, walk it.

  cd Web && ../.venv/bin/python server.py          # then open http://localhost:8000

Runs on the user's machine: it uses the Training/ code and the GPU. The page
is the Next.js app in ui/ (built into ui/out with `npm run build`); without a
build, the old single-file page (index.html) is served. /legacy always serves
the old page.

Two flows (ui/):
  Train your own robot: upload an MJCF, check it against the convention,
    tweak reward weights / limits / morphology conditions, run train.py here,
    watch its stats live, download the checkpoints (or import a .pt).
  OnePolicy: a bundled, generated (sliders) or uploaded robot, walked by
    Web/OnePolicy/OnePolicy.pt (run 2026-10-17_17-48-07, model_10000) in
    play.py's viser viewer, driven with WASD + arrows (play.py --command-port).
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB = Path(__file__).resolve().parent
REPO = WEB.parent
TRAINING = REPO / "Training"
WORKSPACE = WEB / "workspace"
UI_BUILD = WEB / "ui" / "out"
ONEPOLICY = WEB / "OnePolicy" / "OnePolicy.pt"
RUNS = TRAINING / "logs" / "rsl_rl" / "morph_quadruped_velocity"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(WEB))

import onboard  # noqa: E402
from generation.catalog import available_robots, find_robot  # noqa: E402
from generation.knees import is_x_knee  # noqa: E402

PY = sys.executable
CHECK_TIMEOUT_S = 900
# Warp/MuJoCo-Warp kernel-load chatter, dropped from logs shown in the page.
NOISE = re.compile(r"^(Module |Warp \d|   (CUDA|Devices|Kernel|\"cpu\"|\"cuda)|     )|warn")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Training logs keep rsl_rl's indented stats; only kernel-load lines are dropped.
TRAIN_NOISE = re.compile(r"^(Module |Warp \d)")
# Only flat ground is supported by the UI (rough is untested with OnePolicy).
TERRAINS = ("flat",)

_play = {"proc": None, "log": None, "args": None, "port": None, "stopped": False}
_play_lock = threading.Lock()
_train = {"proc": None, "log": None, "args": None, "started": None}
_train_lock = threading.Lock()
_cache: dict = {}
_drive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class ApiError(Exception):
  pass


# ---------------------------------------------------------------- helpers

def checkpoints() -> list[str]:
  """OnePolicy first, then every model_*.pt of the training runs, newest first."""
  ckpts = sorted(RUNS.glob("*/model_*.pt"),
                 key=lambda p: (p.parent.name, int(re.findall(r"\d+", p.stem)[-1])),
                 reverse=True)
  return ([str(ONEPOLICY)] if ONEPOLICY.is_file() else []) + [str(p) for p in ckpts]


def upload_dir(upload: str) -> Path:
  if not re.fullmatch(r"[0-9a-f]{16}", upload or ""):
    raise ApiError("Bad upload id.")
  path = WORKSPACE / upload
  if not path.is_dir():
    raise ApiError("Upload not found (server restarted?). Upload again.")
  return path


def inside(base: Path, rel: str) -> Path:
  """base / rel, refusing anything that escapes base."""
  path = (base / rel).resolve()
  if base.resolve() not in path.parents:
    raise ApiError(f"Path escapes the workspace: {rel}")
  return path


def robot_xml(body: dict) -> Path:
  if body.get("kind") == "robot":
    return find_robot(body["robot"])
  return inside(upload_dir(body.get("upload")), body.get("xml", ""))


def clean_log(text: str) -> str:
  return "\n".join(l for l in ANSI.sub("", text).splitlines() if not NOISE.search(l))


def policy_args(body: dict) -> list[str]:
  policy = body.get("policy") or (checkpoints() or [None])[0]
  if policy is None:
    return []
  if not Path(policy).is_file():
    raise ApiError(f"Policy not found: {policy}")
  return ["--policy", policy]


def gen_params(body: dict) -> str:
  params = body.get("params") or {}
  if not isinstance(params, dict):
    raise ApiError("params must be an object.")
  return json.dumps(params) if params else ""


def source_args(body: dict, play: bool) -> list[str]:
  source = body.get("source")
  if source == "random":
    seed = str(int(body.get("seed") or secrets.randbits(31)))
    args = ["--random", "True", "--seed", seed] if play else ["--random", "--seed", seed]
    if play and gen_params(body):
      args += ["--gen-params", gen_params(body)]
    return args
  if source == "upload":
    if not play:
      raise ApiError("Install the robot before the headless check.")
    args = ["--robot-xml", str(robot_xml({**body, "kind": "upload"}))]
    if body.get("forward") not in (None, "", "auto"):
      args += ["--forward", body["forward"]]
    return args
  return ["--robot", find_robot(body["robot"]).stem]


def free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def stop_proc(proc) -> None:
  if proc is not None and proc.poll() is None:
    proc.terminate()
    try:
      proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
      proc.kill()


# ------------------------------------------------------- robots / inspect

def api_state(_: dict) -> dict:
  return {"robots": list(available_robots()), "checkpoints": checkpoints(),
          "onepolicy": str(ONEPOLICY) if ONEPOLICY.is_file() else None,
          "terrains": TERRAINS}


def api_upload(body: dict) -> dict:
  """Files of a robot folder: [{path: 'my_dog/robot.xml', data: <base64>}]."""
  files = body.get("files") or []
  if not files:
    raise ApiError("No files.")
  upload = secrets.token_hex(8)
  base = WORKSPACE / upload
  base.mkdir(parents=True)
  for f in files:
    out = inside(base, f["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(f["data"]))
  xmls = sorted(str(p.relative_to(base)) for p in base.rglob("*.xml")
                if not p.name.endswith(".temp.xml"))
  # The robot file first: scenes <include> it, MJX variants are for MJX.
  xmls.sort(key=lambda x: (Path(x).stem.startswith("scene") or x.endswith("_mjx.xml"), x))
  if not xmls:
    raise ApiError("No .xml file in the upload.")
  return {"upload": upload, "xmls": xmls}


def api_inspect(body: dict) -> dict:
  path = robot_xml(body)
  forward = body.get("forward") or "auto"
  if body.get("kind") == "robot" and (path.parent / "play.json").is_file():
    forward = json.loads((path.parent / "play.json").read_text()).get("forward", forward)
  xknee = body.get("xknee")
  if xknee is None:
    xknee = is_x_knee(path.stem)
  out: dict = {"forward": forward, "xknee": xknee, "file": path.name}
  try:
    out["table"] = onboard.joint_rows(path, forward)
  except Exception as e:  # noqa: BLE001
    out["table"] = None
    out["table_error"] = str(e)
  errors, robot, pose, notes = onboard.run_checks(path, forward, xknee)
  out["errors"] = errors
  out["report"] = (robot.report if robot else []) + notes
  out["pose"] = pose
  if body.get("kind") != "robot":
    out["text"] = path.read_text()
    out["default_name"] = re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") \
      .replace("_xknee", "")
  return out


def api_save(body: dict) -> dict:
  path = robot_xml(body)
  backup = path.with_name(path.name + ".orig")
  if not backup.exists():
    shutil.copy2(path, backup)
  path.write_text(body["text"])
  return api_inspect(body)


def api_install(body: dict) -> dict:
  path = robot_xml(body)
  name = re.sub(r"[^a-z0-9_]+", "_", (body.get("name") or "").lower()).strip("_")
  if not name:
    raise ApiError("Give the robot a name.")
  forward, xknee = body.get("forward") or "auto", bool(body.get("xknee"))
  errors, *_ = onboard.run_checks(path, forward, xknee)
  if errors:
    raise ApiError("Fix the convention errors first.")
  try:
    installed = onboard.install(path, name, xknee, forward, bool(body.get("overwrite")),
                                auto=True)
  except SystemExit as e:
    raise ApiError(str(e) if str(e) != "Not installed." else
                   f"A robot folder '{name}' already exists: tick overwrite or rename.")
  return {"robot": installed.stem, "path": str(installed)}


def api_geometry(body: dict) -> dict:
  """Visible geoms of a robot in its default pose, for the 3D views (geometry.py)."""
  import geometry  # Imports mujoco + mjlab constants; only needed here.

  source = body.get("source") or "robot"
  if source == "random":
    seed = int(body.get("seed") if body.get("seed") not in (None, "") else secrets.randbits(31))
    params = body.get("params") or None
    key = ("random", seed, json.dumps(params, sort_keys=True))
    if key not in _cache:
      try:
        _cache[key] = {**geometry.generated_geometry(seed, params), "seed": seed}
      except ValueError as e:
        raise ApiError(str(e))
  else:
    path = find_robot(body["robot"]) if source == "robot" else robot_xml({**body, "kind": "upload"})
    forward = body.get("forward") or "auto"
    key = ("file", str(path), path.stat().st_mtime, forward, body.get("xknee"))
    if key not in _cache:
      try:
        _cache[key] = geometry.file_geometry(path, forward, body.get("xknee"))
      except Exception as e:  # noqa: BLE001
        raise ApiError(f"Cannot build {path.name}: {e}")
  if len(_cache) > 64:
    _cache.pop(next(iter(_cache)))
  return _cache[key]


def api_check(body: dict) -> dict:
  """Headless: the policy standing 3 s, then walking forward 5 s."""
  cmd = [PY, "onboard.py", *source_args(body, play=False), *policy_args(body),
         "--yes", "--no-viewer", "--command", str(float(body.get("command") or 0.5))]
  try:
    run = subprocess.run(cmd, cwd=TRAINING, capture_output=True, text=True,
                         timeout=CHECK_TIMEOUT_S)
  except subprocess.TimeoutExpired:
    raise ApiError(f"Check took longer than {CHECK_TIMEOUT_S} s.")
  log = clean_log(run.stdout + run.stderr)
  return {"ok": run.returncode == 0, "log": log, "command": " ".join(cmd[1:])}


# ------------------------------------------------------------------- play

def api_play(body: dict) -> dict:
  terrain = body.get("terrain") or "flat"
  if terrain not in TERRAINS:
    raise ApiError(f"Terrain '{terrain}' is under construction; only flat works.")
  cmd = [PY, "play.py", *source_args(body, play=True), *policy_args(body),
         "--viewer", "viser", "--task", terrain]
  if not policy_args(body):
    cmd += ["--agent", "zero"]
  port = None
  if body.get("drive"):
    port = free_port()
    cmd += ["--command-port", str(port)]
  elif body.get("command") not in (None, ""):
    cmd += ["--command", f"{float(body['command'])},0,0"]
  with _play_lock:
    stop_proc(_play["proc"])
    WORKSPACE.mkdir(exist_ok=True)
    log = WORKSPACE / "play.log"
    _play.update(log=log, args=" ".join(cmd[1:]), port=port, stopped=False)
    _play["proc"] = subprocess.Popen(cmd, cwd=TRAINING, stdout=log.open("w"),
                                     stderr=subprocess.STDOUT, start_new_session=True)
  return api_play_status({})


def api_play_status(_: dict) -> dict:
  proc, log = _play["proc"], _play["log"]
  text = log.read_text(errors="replace") if log and log.exists() else ""
  port = None
  for m in re.finditer(r"https?://[\w.\-]+:(\d+)", text):
    port = int(m.group(1))
  return {"running": proc is not None and proc.poll() is None,
          "exit_code": None if proc is None else proc.poll(),
          "viser_port": port, "drive": _play["port"] is not None, "stopped": _play["stopped"],
          "command": _play["args"],
          "log": "\n".join(clean_log(text).splitlines()[-40:])}


def api_stop(_: dict) -> dict:
  with _play_lock:
    stop_proc(_play["proc"])
    _play["stopped"] = _play["proc"] is not None
  return api_play_status({})


def api_drive(body: dict) -> dict:
  """Velocity command from the keyboard (Go2 units) -> play.py's UDP port."""
  port = _play["port"]
  if port is None or _play["proc"] is None or _play["proc"].poll() is not None:
    raise ApiError("No robot is being driven.")
  values = [float(body.get(k) or 0.0) for k in ("forward", "left", "yaw")]
  if not all(abs(v) <= 10 for v in values):
    raise ApiError("Command out of range.")
  _drive.sendto(" ".join(f"{v:.4f}" for v in values).encode(), ("127.0.0.1", port))
  return {"ok": True}


# ---------------------------------------------------------------- training

# Training settings shown as sliders: the reward terms come from the env config.
LIMITS = [
  # key, label, tyro flag, default, min, max, step, unit
  ("num_envs", "Robots (parallel envs)", "--num-envs", 4096, 64, 8192, 64, ""),
  ("max_iterations", "PPO iterations", "--agent.max-iterations", 3000, 1, 20000, 10, ""),
  ("save_interval", "Checkpoint every", "--agent.save-interval", 100, 10, 1000, 10, "it"),
  ("learning_rate", "Learning rate", "--agent.algorithm.learning-rate", 1e-3, 1e-5, 1e-2, 1e-5, ""),
  ("episode_length_s", "Episode length", "--env.episode-length-s", 20.0, 5.0, 60.0, 1.0, "s"),
  ("fell_over_angle", "Fell-over tilt limit", "--env.terminations.fell-over.params.limit-angle",
   1.2217, 0.3, 1.57, 0.01, "rad"),
  ("illegal_contact_force", "Illegal contact force", "--env.terminations.illegal-contact.params.force-threshold",
   10.0, 1.0, 200.0, 1.0, "N (Go2)"),
  ("delta_scale", "Randomisation range", "--morphology.delta-scale", 1.0, 0.0, 1.0, 0.05, "x"),
  ("x_knee_fraction", "Share of >< knees", "--morphology.x-knee-fraction", 0.5, 0.0, 1.0, 0.05, ""),
]


def _reward_table() -> dict:
  if "rewards" in _cache:
    return _cache["rewards"]
  import inspect

  from generation.conditions import FUNCTIONS, VARIABLES
  from generation.quadruped_generator import PARAM_KEYS
  from tasks.velocity.config.morph_quadruped import morph_quadruped_flat_env_cfg

  cfg = morph_quadruped_flat_env_cfg()
  staged = {c.params["reward_name"]: c.params["weight_stages"]
            for c in cfg.curriculum.values() if "weight_stages" in (c.params or {})}
  terms = []
  for name, term in cfg.rewards.items():
    doc = (inspect.getdoc(term.func) or "").strip().split("\n\n")[0].replace("\n", " ")
    params = {k: v for k, v in (term.params or {}).items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    terms.append({"name": name, "weight": term.weight, "func": term.func.__name__,
                  "doc": doc, "params": params, "curriculum": staged.get(name)})
  limits = [dict(zip(("key", "label", "flag", "value", "min", "max", "step", "unit"), l))
            for l in LIMITS]
  _cache["rewards"] = {"terms": terms, "limits": limits, "variables": VARIABLES,
                       "functions": list(FUNCTIONS), "gen_params": list(PARAM_KEYS)}
  return _cache["rewards"]


def api_rewards(_: dict) -> dict:
  return _reward_table()


def api_condition_test(body: dict) -> dict:
  """How many of `samples` generated robots pass the condition (seeds 0..n-1)."""
  from assets.morph_quadruped.constants import INIT_JOINT_POS
  from generation.bank import _compile, _extract
  from generation.conditions import ConditionError, evaluate, parse_condition, robot_variables
  from generation.knees import x_knee_pose
  from generation.mjcf_builder import make_robot_xml

  text = (body.get("condition") or "").strip()
  try:
    tree = parse_condition(text) if text else None
  except ConditionError as e:
    raise ApiError(str(e))
  params = body.get("params") or None
  samples = max(4, min(int(body.get("samples") or 24), 64))
  kept, rows = 0, []
  for seed in range(samples):
    xml, mass, max_vel = make_robot_xml(seed, params=params)
    model = _compile(xml)
    x_knee = is_x_knee(ET.fromstring(xml).get("model", ""))
    pose = x_knee_pose(INIT_JOINT_POS) if x_knee else INIT_JOINT_POS
    _, _, scalars = _extract(model, pose, 0.02)
    scalars.update(total_mass=mass, max_joint_velocity=max_vel)
    variables = robot_variables(model, scalars)
    try:
      ok = evaluate(tree, variables) if tree else True
    except ConditionError as e:
      raise ApiError(str(e))
    kept += ok
    rows.append({"seed": seed, "ok": ok, "mass": variables["mass"],
                 "leg_length": variables["leg_length"]})
  return {"kept": kept, "samples": samples, "rows": rows}


def _resume_args(policy: str | None) -> list[str]:
  """Start from a checkpoint: it must sit in a run folder under RUNS."""
  if not policy:
    return []
  path = Path(policy).resolve()
  if not path.is_file():
    raise ApiError(f"Checkpoint not found: {policy}")
  if path.parent.parent != RUNS.resolve() or not re.fullmatch(r"model_\d+\.pt", path.name):
    run = RUNS / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_from_{path.stem}"
    run.mkdir(parents=True)
    n = (re.findall(r"\d+", path.stem) or ["0"])[-1]
    path = shutil.copy2(path, run / f"model_{n}.pt")
    path = Path(path)
  return ["--agent.resume", "True", "--agent.load-run", path.parent.name,
          "--agent.load-checkpoint", path.name]


def api_train(body: dict) -> dict:
  table = _reward_table()
  cmd = [PY, "train.py", "--plot"]
  values = {l["key"]: l["value"] for l in table["limits"]}
  values.update({k: v for k, v in (body.get("limits") or {}).items() if k in values})
  for l in table["limits"]:
    value = values[l["key"]]
    if not isinstance(value, (int, float)) or not l["min"] <= value <= l["max"]:
      raise ApiError(f"{l['label']} must be within [{l['min']}, {l['max']}].")
    cmd += [l["flag"], str(int(value) if isinstance(l["value"], int) else float(value))]

  known = {t["name"]: t for t in table["terms"]}
  for name, change in (body.get("rewards") or {}).items():
    if name not in known:
      raise ApiError(f"Unknown reward term {name}.")
    flag = f"--env.rewards.{name.replace('_', '-')}"
    if "weight" in change and change["weight"] != known[name]["weight"]:
      if known[name]["curriculum"]:
        raise ApiError(f"{name}'s weight is set by the curriculum; it can't be changed here.")
      cmd += [f"{flag}.weight", str(float(change["weight"]))]
    for param, value in (change.get("params") or {}).items():
      if param not in known[name]["params"]:
        raise ApiError(f"{name} has no parameter {param}.")
      if value != known[name]["params"][param]:
        cmd += [f"{flag}.params.{param.replace('_', '-')}", str(float(value))]

  if gen_params(body):
    cmd += ["--morphology.params", gen_params(body)]
  condition = (body.get("condition") or "").strip()
  if condition:
    from generation.conditions import ConditionError, parse_condition
    try:
      parse_condition(condition)
    except ConditionError as e:
      raise ApiError(str(e))
    cmd += ["--morphology.condition", condition]

  robot = body.get("robot")
  if robot:
    fraction = float(body.get("fraction") or 0.5)
    if not 0.0 < fraction < 1.0:
      raise ApiError("Share of envs on your robot must be in (0, 1).")
    cmd += ["--generalise", str(fraction), "--generalise-robots", find_robot(robot).stem]
  cmd += _resume_args(body.get("policy"))
  # ONNX only on request: the run is tried out in the sim from its .pt files.
  if not body.get("export_onnx"):
    cmd += ["--export-onnx", "False"]

  with _train_lock:
    if _train["proc"] is not None and _train["proc"].poll() is None:
      raise ApiError("A training run is already going; stop it first.")
    with _play_lock:  # One GPU job at a time.
      stop_proc(_play["proc"])
      _play["proc"] = None
    WORKSPACE.mkdir(exist_ok=True)
    log = WORKSPACE / "train.log"
    _train.update(log=log, args=" ".join(cmd[1:]), started=time.time())
    _train["proc"] = subprocess.Popen(cmd, cwd=TRAINING, stdout=log.open("w"),
                                      stderr=subprocess.STDOUT, start_new_session=True,
                                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
  return api_train_status({})


STAT_LINE = re.compile(r"^\s*([A-Za-z_][\w /.-]*?):\s+(-?[\d.]+(?:e[-+]?\d+)?)s?\s*$")


def _parse_train_log(text: str) -> dict:
  blocks = re.split(r"Learning iteration (\d+)/(\d+)", text)
  history, latest, total = [], {}, None
  for i in range(1, len(blocks) - 2, 3):
    it, total = int(blocks[i]), int(blocks[i + 1])
    stats = {}
    for line in blocks[i + 2].splitlines():
      m = STAT_LINE.match(line)
      if m:
        stats[m.group(1).strip()] = float(m.group(2))
    latest = {"iteration": it, **stats}
    history.append({"iteration": it,
                    "reward": stats.get("Mean reward"),
                    "episode_length": stats.get("Mean episode length"),
                    "std": stats.get("Mean action std"),
                    "fps": stats.get("Steps per second")})
  etas = re.findall(r"ETA:\s*(\d+:\d\d:\d\d)", text)
  log_dir = None
  m = re.search(r"Logging experiment in directory: (\S+)", text)
  if m:
    log_dir = (TRAINING / m.group(1)).resolve()
  return {"history": history[-3000:], "latest": latest, "total": total, "log_dir": log_dir,
          "eta": etas[-1] if etas else None}


def api_train_status(_: dict) -> dict:
  proc, log = _train["proc"], _train["log"]
  raw = ANSI.sub("", log.read_text(errors="replace")) if log and log.exists() else ""
  # Parsed before clean_log: rsl_rl right-aligns its stats, which NOISE drops.
  parsed = _parse_train_log(raw)
  text = "\n".join(l for l in raw.splitlines() if not TRAIN_NOISE.search(l))
  running = proc is not None and proc.poll() is None
  if proc is None:
    phase = "idle"
  elif parsed["history"]:
    phase = "training" if running else ("done" if proc.poll() == 0 else "stopped")
  elif running:
    phase = "generating" if "Generating" in text and "Building" not in text else "building"
  else:
    phase = "failed"
  log_dir = parsed.pop("log_dir")
  ckpts = []
  if log_dir is not None and log_dir.is_dir():
    ckpts = sorted((str(p) for p in log_dir.glob("model_*.pt")),
                   key=lambda p: int(re.findall(r"\d+", Path(p).stem)[-1]))
    onnx = log_dir / "policy.onnx"
    ckpts += [str(onnx)] if onnx.is_file() else []
  return {"running": running, "phase": phase, "exit_code": None if proc is None else proc.poll(),
          "command": _train["args"], "started": _train["started"],
          "log_dir": str(log_dir) if log_dir else None, "checkpoints": ckpts,
          "log": "\n".join(text.splitlines()[-60:]), **parsed}


def api_train_stop(_: dict) -> dict:
  with _train_lock:
    proc = _train["proc"]
    if proc is not None and proc.poll() is None:
      os.killpg(proc.pid, 2)  # Ctrl+C: train.py saves the plot and cleans up.
      try:
        proc.wait(timeout=30)
      except subprocess.TimeoutExpired:
        stop_proc(proc)
  return api_train_status({})


def api_import_policy(body: dict) -> dict:
  """A .pt from the user's disk -> its own run folder, usable to play / resume."""
  name = Path(body.get("name") or "policy.pt").name
  if not name.endswith(".pt"):
    raise ApiError("Import a .pt checkpoint.")
  data = base64.b64decode(body.get("data") or "")
  if not data:
    raise ApiError("Empty file.")
  stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem)
  run = RUNS / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_imported_{stem}"
  run.mkdir(parents=True)
  n = (re.findall(r"\d+", stem) or ["0"])[-1]
  path = run / f"model_{n}.pt"
  path.write_bytes(data)
  return {"policy": str(path), "checkpoints": checkpoints()}


def download(query: dict) -> tuple[bytes, str, str]:
  """A checkpoint, ONNX or plot from the runs or Web/OnePolicy/: (data, type, name)."""
  path = Path((query.get("path") or [""])[0]).resolve()
  roots = (RUNS.resolve(), ONEPOLICY.parent.resolve())
  if not any(r in path.parents for r in roots) or path.suffix not in (".pt", ".onnx", ".csv", ".png"):
    raise ApiError("Only checkpoints, ONNX files and plots of the runs can be downloaded.")
  if not path.is_file():
    raise ApiError("File not found.")
  name = path.name if path.parent.parent != RUNS.resolve() else f"{path.parent.name}_{path.name}"
  return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream", name


# -------------------------------------------------------------- endpoints

ROUTES = {
  "/api/state": api_state, "/api/upload": api_upload, "/api/inspect": api_inspect,
  "/api/save": api_save, "/api/install": api_install, "/api/check": api_check,
  "/api/geometry": api_geometry,
  "/api/play": api_play, "/api/play_status": api_play_status, "/api/stop": api_stop,
  "/api/drive": api_drive,
  "/api/rewards": api_rewards, "/api/condition_test": api_condition_test,
  "/api/train": api_train, "/api/train_status": api_train_status,
  "/api/train_stop": api_train_stop, "/api/import_policy": api_import_policy,
}
QUIET = ("/api/play_status", "/api/train_status", "/api/drive")


def static_file(path: str) -> Path | None:
  """File of the Next.js export for a URL path (trailingSlash: dirs hold index.html)."""
  if not UI_BUILD.is_dir():
    return None
  rel = path.lstrip("/")
  for candidate in (rel, rel + "index.html", rel + "/index.html", rel + ".html"):
    try:
      file = inside(UI_BUILD, candidate) if candidate else UI_BUILD / "index.html"
    except ApiError:
      return None
    if file.is_file():
      return file
  return None


class Handler(BaseHTTPRequestHandler):
  def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    for k, v in (extra or {}).items():
      self.send_header(k, v)
    self.end_headers()
    if self.command != "HEAD":
      self.wfile.write(body)

  def do_HEAD(self) -> None:  # noqa: N802
    self.do_GET()

  def do_GET(self) -> None:  # noqa: N802
    url = urlparse(self.path)
    if url.path == "/api/download":
      try:
        data, ctype, name = download(parse_qs(url.query))
      except ApiError as e:
        self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
        return
      self._send(200, data, ctype, {"Content-Disposition": f'attachment; filename="{name}"'})
    elif url.path in ROUTES:
      self._api(url.path, {})
    elif url.path in ("/legacy", "/legacy/") or (url.path in ("/", "/index.html")
                                                  and not UI_BUILD.is_dir()):
      self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
    elif (file := static_file(url.path)) is not None:
      ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
      self._send(200, file.read_bytes(), ctype)
    elif (file := static_file("/404")) is not None:
      self._send(404, file.read_bytes(), "text/html; charset=utf-8")
    else:
      self._send(404, b"not found", "text/plain")

  def do_POST(self) -> None:  # noqa: N802
    path = urlparse(self.path).path
    if path not in ROUTES:
      self._send(404, b"not found", "text/plain")
      return
    length = int(self.headers.get("Content-Length") or 0)
    try:
      body = json.loads(self.rfile.read(length) or b"{}")
    except json.JSONDecodeError:
      self._send(400, b'{"error": "bad JSON"}', "application/json")
      return
    self._api(path, body)

  def _api(self, path: str, body: dict) -> None:
    try:
      out, code = ROUTES[path](body), 200
    except (ApiError, ValueError, KeyError, FileNotFoundError) as e:
      out, code = {"error": str(e)}, 400
    except Exception as e:  # noqa: BLE001
      out, code = {"error": f"{type(e).__name__}: {e}"}, 500
    self._send(code, json.dumps(out).encode(), "application/json")

  def log_message(self, fmt, *args) -> None:
    if not any(q in str(args[0] if args else "") for q in QUIET):
      super().log_message(fmt, *args)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1",
                      help="127.0.0.1 = this machine only; 0.0.0.0 = whole network")
  parser.add_argument("--port", type=int, default=8000)
  args = parser.parse_args()
  WORKSPACE.mkdir(exist_ok=True)
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"OnePolicy harness on http://localhost:{args.port}"
        + ("" if UI_BUILD.is_dir() else " (no ui/out build: serving the old page)"), flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    stop_proc(_play["proc"])
    stop_proc(_train["proc"])


if __name__ == "__main__":
  main()
