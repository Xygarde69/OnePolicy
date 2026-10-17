"""OnePolicy web harness: pick or upload a quadruped, fix it, make it walk.

  cd Web && ../.venv/bin/python server.py          # then open http://localhost:8000

Runs on the user's machine: it uses the Training/ code (generation/adapt.py,
play.py) and the GPU for the simulation. Inference only, never training.

Flow (index.html):
  1. Policy: pick a checkpoint (default: newest model_*.pt in Training/logs).
  2. Robot: an existing one (../quadrupeds), a random generated one, or your
     own robot folder (MJCF + meshes), uploaded into Web/workspace/<id>/.
  3. Your own robot: joint table + convention check; edit the XML in the page,
     save, re-check until it passes; set forward axis / knee layout; install
     into ../quadrupeds/<name>/ (Training/onboard.py does the work).
  4. Headless check (onboard.py: stands with zero actions? walks with the
     policy?), then Walk: play.py with the browser viewer (viser), embedded.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
TRAINING = WEB.parent / "Training"
WORKSPACE = WEB / "workspace"
sys.path.insert(0, str(TRAINING))

import onboard  # noqa: E402
from generation.catalog import available_robots, find_robot  # noqa: E402
from generation.knees import is_x_knee  # noqa: E402

PY = sys.executable
CHECK_TIMEOUT_S = 900
# Warp/MuJoCo-Warp kernel-load chatter, dropped from logs shown in the page.
NOISE = re.compile(r"^(Module |Warp \d|   (CUDA|Devices|Kernel|\"cpu\"|\"cuda)|     )|warn")

_play = {"proc": None, "log": None, "args": None}
_play_lock = threading.Lock()


class ApiError(Exception):
  pass


# ---------------------------------------------------------------- helpers

def checkpoints() -> list[str]:
  runs = TRAINING / "logs" / "rsl_rl" / "morph_quadruped_velocity"
  ckpts = sorted(runs.glob("*/model_*.pt"),
                 key=lambda p: (p.parent.name, int(re.findall(r"\d+", p.stem)[-1])),
                 reverse=True)
  return [str(p) for p in ckpts]


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
  return "\n".join(l for l in text.splitlines() if not NOISE.search(l))


def policy_args(body: dict) -> list[str]:
  policy = body.get("policy") or (checkpoints() or [None])[0]
  if policy is None:
    return []
  if not Path(policy).is_file():
    raise ApiError(f"Policy not found: {policy}")
  return ["--policy", policy]


def source_args(body: dict, play: bool) -> list[str]:
  if body.get("source") == "random":
    seed = str(int(body.get("seed") or secrets.randbits(31)))
    return ["--random", "True", "--seed", seed] if play else ["--random", "--seed", seed]
  return ["--robot", find_robot(body["robot"]).stem]


# -------------------------------------------------------------- endpoints

def api_state(_: dict) -> dict:
  return {"robots": list(available_robots()), "checkpoints": checkpoints()}


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
  xmls = sorted(str(p.relative_to(base)) for p in base.rglob("*.xml"))
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


def api_check(body: dict) -> dict:
  """Headless: zero actions 2 s, then the policy walking forward 5 s."""
  cmd = [PY, "onboard.py", *source_args(body, play=False), *policy_args(body),
         "--yes", "--no-viewer", "--command", str(float(body.get("command") or 0.5))]
  try:
    run = subprocess.run(cmd, cwd=TRAINING, capture_output=True, text=True,
                         timeout=CHECK_TIMEOUT_S)
  except subprocess.TimeoutExpired:
    raise ApiError(f"Check took longer than {CHECK_TIMEOUT_S} s.")
  log = clean_log(run.stdout + run.stderr)
  return {"ok": run.returncode == 0, "log": log, "command": " ".join(cmd[1:])}


def _stop_play() -> None:
  proc = _play["proc"]
  if proc is not None and proc.poll() is None:
    proc.terminate()
    try:
      proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
      proc.kill()
  _play["proc"] = None


def api_play(body: dict) -> dict:
  cmd = [PY, "play.py", *source_args(body, play=True), *policy_args(body),
         "--viewer", "viser"]
  if not policy_args(body):
    cmd += ["--agent", "zero"]
  if body.get("command") not in (None, ""):
    speed = float(body["command"])
    cmd += ["--command", f"{speed},0,0"]
  with _play_lock:
    _stop_play()
    WORKSPACE.mkdir(exist_ok=True)
    log = WORKSPACE / "play.log"
    _play["log"] = log
    _play["args"] = " ".join(cmd[1:])
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
          "viser_port": port, "command": _play["args"],
          "log": "\n".join(clean_log(text).splitlines()[-40:])}


def api_stop(_: dict) -> dict:
  with _play_lock:
    _stop_play()
  return api_play_status({})


ROUTES = {
  "/api/state": api_state, "/api/upload": api_upload, "/api/inspect": api_inspect,
  "/api/save": api_save, "/api/install": api_install, "/api/check": api_check,
  "/api/play": api_play, "/api/play_status": api_play_status, "/api/stop": api_stop,
}


class Handler(BaseHTTPRequestHandler):
  def _send(self, code: int, body: bytes, ctype: str) -> None:
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # noqa: N802
    if self.path in ("/", "/index.html"):
      self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
    elif self.path in ROUTES:
      self._api({})
    else:
      self._send(404, b"not found", "text/plain")

  def do_POST(self) -> None:  # noqa: N802
    if self.path not in ROUTES:
      self._send(404, b"not found", "text/plain")
      return
    length = int(self.headers.get("Content-Length") or 0)
    try:
      body = json.loads(self.rfile.read(length) or b"{}")
    except json.JSONDecodeError:
      self._send(400, b'{"error": "bad JSON"}', "application/json")
      return
    self._api(body)

  def _api(self, body: dict) -> None:
    try:
      out, code = ROUTES[self.path](body), 200
    except (ApiError, ValueError, KeyError, FileNotFoundError) as e:
      out, code = {"error": str(e)}, 400
    except Exception as e:  # noqa: BLE001
      out, code = {"error": f"{type(e).__name__}: {e}"}, 500
    self._send(code, json.dumps(out).encode(), "application/json")

  def log_message(self, fmt, *args) -> None:
    if "/api/play_status" not in (args[0] if args else ""):
      super().log_message(fmt, *args)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1",
                      help="127.0.0.1 = this machine only; 0.0.0.0 = whole network")
  parser.add_argument("--port", type=int, default=8000)
  args = parser.parse_args()
  WORKSPACE.mkdir(exist_ok=True)
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"OnePolicy harness on http://localhost:{args.port}", flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    _stop_play()


if __name__ == "__main__":
  main()
