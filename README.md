# OnePolicy

**One locomotion policy for many quadrupeds.**

OnePolicy trains a single velocity-tracking policy on thousands of procedurally
generated quadrupeds at once. Every parallel environment simulates a
*different* robot, with its own body size, leg lengths, link masses and
inertias. Because the policy never sees the same body twice, it learns to walk
across morphologies instead of memorizing one robot.

The same policy then runs on robots it was never trained on: Unitree Go2 / Go1
/ A1, Boston Dynamics Spot, ANYbotics ANYmal, Google Barkour, or your own MJCF.
At play time, each robot is rewritten into the training convention
automatically.

```
                 ┌───────────────────────────── train.py ──────────────────────────────┐
  GenLoco        │  generate N robots ─► compile N MjModels ─► check same structure    │
  generator ────►│        │                                        │                   │
  (PyBullet)     │        ▼                                        ▼                   │
                 │  env_00000.xml … env_N.xml          morphologies.npz (per-env data) │
                 │        │                                        │                   │
                 │        ▼                                        ▼                   │
                 │  one batched MuJoCo-Warp model ◄─ startup event: env i = robot i    │
                 │        │                                                            │
                 │        ▼                                                            │
                 │  PPO (rsl_rl) on the GPU  ─────────────────────────►  model_*.pt    │
                 └─────────────────────────────────────────────────────────────────────┘
                 ┌───────────────────────────── play.py ───────────────────────────────┐
  any quadruped  │  adapt.py: find legs, rename to Go2 names, reorder, fix axes,       │
  MJCF     ─────►│  rotate base if needed ─► 1 env ─► same policy ─► viewer            │
                 └─────────────────────────────────────────────────────────────────────┘
```

## The policy: `OnePolicy/OnePolicy.pt`

**`OnePolicy/OnePolicy.pt` is OnePolicy**: the one trained policy in this
repository, and the only one published. Use it for everything (playing,
onboarding a robot, the web harness) unless you train your own.

| | |
|---|---|
| File | `OnePolicy/OnePolicy.pt` (rsl_rl checkpoint, actor + critic + optimizer) and `OnePolicy/OnePolicy.onnx` (actor only, for deployment) |
| Source run | `2026-10-17_05-05-36`, checkpoint `model_1200.pt` (1200 PPO iterations) |
| Code | commit `40fec60` |
| Training | 4096 generated quadrupeds (1/4-4x Go2, `>>` and `><` knees), flat ground, robot seed `1437162221` (`OnePolicy/morphology_seed.txt`) |
| Config | `OnePolicy/params/env.yaml`, `OnePolicy/params/agent.yaml` |
| Observations | 51 actor inputs (with `knee_direction`); older checkpoints don't load |

```bash
cd Training
$PY play.py --robot spot --policy ../OnePolicy/OnePolicy.pt
```

Results on the bundled robots are in [Robot compatibility](#robot-compatibility).
It is an early checkpoint: it walks every bundled robot at 0.5 m/s but has not
seen the later command curriculum (fast commands, from iteration 5000).

## Run OnePolicy (minimum)

Needs Linux, an NVIDIA GPU, git, [uv](https://docs.astral.sh/uv/) and
Node.js >= 20.

```bash
git clone https://github.com/Xygarde69/OnePolicy.git && cd OnePolicy
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
cd Web/ui && npm install && npm run build && cd ..
../.venv/bin/python server.py
```

Open http://localhost:8000, choose **OnePolicy**, pick a robot and drive it
with W/A/S/D and ←/→.

---

## Contents

- [Run OnePolicy (minimum)](#run-onepolicy-minimum)

- [The policy](#the-policy-onepolicyonepolicypt)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Training](#training)
- [Playing](#playing)
- [Robot convention and the adapter](#robot-convention-and-the-adapter)
- [Robot compatibility](#robot-compatibility)
- [Adding your own robot](#adding-your-own-robot)
- [Web UI](#web-ui)
- [Morphology viewer](#morphology-viewer)
- [How per-environment robots work](#how-per-environment-robots-work)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Credits and licenses](#credits-and-licenses)

---

## Repository layout

```
Morphology/
├── OnePolicy/                        # THE policy: OnePolicy.pt (+ .onnx, params/)
├── Training/                         # OnePolicy: training and playing
│   ├── train.py                      # generate N robots, then train one policy
│   ├── play.py                       # play the policy on one robot
│   ├── generation/
│   │   ├── quadruped_generator.py    # GenLoco's random quadruped generator (Go2 nominal values)
│   │   ├── mjcf_builder.py           # GenLoco robot -> Go2-named MJCF
│   │   ├── bank.py                   # generate + compile + validate N robots
│   │   ├── adapt.py                  # any quadruped MJCF -> training convention
│   │   ├── robot_descriptions/…/hip.obj   # the one mesh the generator uses
│   │   └── LICENSE_GenLoco.txt
│   ├── assets/morph_quadruped/
│   │   └── constants.py              # robot config (mirrors Unitree Go2's)
│   └── tasks/velocity/               # velocity task (from unitree_rl_mjlab)
│       ├── velocity_env_cfg.py       # observations, rewards, terminations, …
│       ├── mdp/                      # reward/obs/command terms
│       │   └── morphology.py         # startup event: env i gets robot i
│       ├── rl/runner.py              # PPO runner (+ ONNX export)
│       └── config/morph_quadruped/   # env + PPO config (Go2's, robot swapped)
├── Web/                              # browser UI: train your robot, or walk any robot with OnePolicy
│   ├── server.py                     # stdlib HTTP server + JSON API (runs train.py / play.py here)
│   ├── geometry.py                   # robot geoms -> three.js for the 3D previews
│   └── ui/                           # Next.js + TypeScript + Tailwind + shadcn/ui front end
├── Model_generation/                 # stand-alone generator + MuJoCo viewer
│   ├── generate_and_view.py
│   └── quadruped_generator.py
└── quadrupeds/                       # MuJoCo Menagerie robots to play on
    ├── unitree_go2/  unitree_go1/  unitree_a1/
    ├── boston_dynamics_spot/
    ├── anybotics_anymal_b/  anybotics_anymal_c/
    └── google_barkour_v0/  google_barkour_vb/
```

The training code is built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo-Warp + manager-based RL) and the Go2 velocity task from
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab). Only the
quadruped velocity task was copied. The Unitree humanoid configs are not part of
this project.

---

## Setup

OnePolicy has its own Python environment in `.venv`, pinned in
`requirements.txt` (138 packages, frozen from the environment it was developed
and tested in). It provides mjlab, MuJoCo-Warp, PyTorch, rsl_rl and PyBullet.

| Package     | Version              |
|-------------|----------------------|
| Python      | 3.11                 |
| mjlab       | 1.2.0                |
| mujoco      | 3.5.0                |
| mujoco-warp | 3.5.0                |
| warp-lang   | 1.12.1               |
| torch       | 2.14.1 (CUDA 13.0)   |
| rsl-rl-lib  | 5.0.1                |
| pybullet    | 3.2.7                |

```bash
cd ~/Documents/Projects/Morphology

# Create the environment (skip if .venv already exists)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt

# Either activate it …
source .venv/bin/activate          # then use `python` below
# … or use this alias, as the rest of this README does
export PY=~/Documents/Projects/Morphology/.venv/bin/python
```

You need an NVIDIA GPU for training. On CUDA driver ≥ 12.4, MuJoCo-Warp uses
CUDA graphs for faster stepping. Older drivers fall back to normal stepping.

All commands below are run from `Training/`:

```bash
cd ~/Documents/Projects/Morphology/Training
```

---

## Quick start

```bash
# Train on 4096 random quadrupeds (one per env), flat ground
$PY train.py --num-envs 4096

# Play OnePolicy on Spot
$PY play.py --robot spot --policy ../OnePolicy/OnePolicy.pt

# … or the newest checkpoint of your own training runs
$PY play.py --robot spot

# … on a fresh random robot, with a specific checkpoint
$PY play.py --random True --policy logs/rsl_rl/morph_quadruped_velocity/<run>/model_3000.pt

# … on your own robot
$PY play.py --robot-xml ~/Documents/Projects/Byte01_Quadraped/Model/V2/V2.xml
```

---

## Training

### What `train.py` does

1. **Generates** one random quadruped per environment with GenLoco's
   generator. Env `i` uses seed `base_seed + i`. Generation runs in parallel
   on all CPU cores (about 27 s for 4096 robots).
2. **Compiles** each robot into its own `MjModel` and checks that all robots
   share the same model structure (see
   [How per-environment robots work](#how-per-environment-robots-work)).
3. **Saves** each robot as `env_XXXXX.xml` and the per-env numbers as
   `morphologies.npz`.
4. **Builds** the batched simulation. A startup event writes robot `i`'s
   geometry, masses, inertias, spawn height and PD gains into env `i`.
5. **Trains** PPO with rsl_rl.

### Command-line options

```bash
$PY train.py --num-envs 4096                    # number of envs = number of robots
$PY train.py --num-envs 4096 --task rough       # rough terrain (default: flat)
$PY train.py --num-envs 4096 --morphology.seed 7          # reproducible robots
$PY train.py --num-envs 4096 --morphology.delta-scale 0.5 # less extreme robots
$PY train.py --num-envs 4096 --agent.max-iterations 3000
$PY train.py --num-envs 4096 --agent.resume True --agent.load-run <run_dir>
$PY train.py --help                              # all --env.* / --agent.* options
```

| Option | Default | Meaning |
|---|---|---|
| `--num-envs` | `env.scene.num_envs` | Parallel envs, one robot each |
| `--task` | `flat` | `flat` or `rough` terrain |
| `--morphology.seed` | random | Base seed. Env `i` uses `seed + i` |
| `--morphology.delta-scale` | `1.0` | Scales every randomization range (0 gives Go2-like robots) |
| `--morphology.foot-clearance` | `0.02` | Gap between lowest foot and floor at spawn [m] |
| `--morphology.x-knee-fraction` | `0.5` | Share of generated robots with `><` knees (rear knees forward); the rest are `>>` |
| `--morphology.workers` | all CPUs | Processes used for generation |
| `--morphology.params` | `""` | Generator values that replace random draws, as JSON, for every robot, e.g. `'{"scale": 1.5, "x_knee": true}'`. `scale` is size vs Go2; `base_length`, `base_width`, `base_height`, `density`, `thigh_length`, `calf_length`, `leg_radius`, `hip_link`, `link_mass` multiply Go2's value at that size |
| `--morphology.condition` | `""` | Keep only robots for which this is true, e.g. `"leg_length > 0.3 and mass < 40"`; failing seeds are skipped. Variables (mass, link lengths, COM, PD gains, torque limits, …) in `generation/conditions.py` |
| `--agent.max-iterations` | `10001` | PPO iterations |
| `--agent.logger` | `tensorboard` | `tensorboard` or `wandb` |
| `--video` | `False` | Record training videos |
| `--plot` | `False` | Plot mean episode reward per iteration, overall and per leg-length third, to `<log_dir>/mean_reward.png` (+ `.csv`) |
| `--symmetry` | `True` | Left-right mirror data augmentation in PPO |
| `--generalise` | `0` | Fraction of envs (bare flag: `0.5`; `0.33` = 33%) split equally between the `--generalise-robots`, each trained on its own model (adapted as in `play.py`) in its own batched sim next to the generated robots'; the rest are generated. Reward logged per robot type. About 1.9× slower per iteration at 0.5 with 4 robots (RTX 3060, 4096 envs) |
| `--generalise-proxies` | `False` | Train on proxies instead: real robots rebuilt in the generated model structure (`generation/proxy.py`), one sim, faster, but an approximation of the real model |
| `--generalise-robots` | `spot a1 barkour_v0 go2` | Real robots in `quadrupeds/` to train proxies of, by name or unique partial name (`--generalise-robots spot go1 anymal_c`) |
| `--gpu-ids` | `[0]` | GPUs to use (several → torchrunx) |

### Output

```
logs/rsl_rl/morph_quadruped_velocity/<YYYY-MM-DD_HH-MM-SS>/
├── morphologies/
│   ├── env_00000.xml … env_04095.xml   # every robot that was trained on
│   └── morphologies.npz                # per-env fields, seeds, masses, spawn heights
├── model_0.pt, model_100.pt, …         # checkpoints (every 100 iterations)
├── policy.onnx                         # latest policy, exported for deployment
├── params/env.yaml, params/agent.yaml  # full configs
└── events.out.tfevents.*               # TensorBoard logs
```

Watch training with `tensorboard --logdir logs/rsl_rl`. Each run's robots take
about 100 MB for 4096 envs.

### What is randomized

All randomization comes from GenLoco's `create_robot`
(`generation/quadruped_generator.py`, nominal values changed from A1's to
Unitree Go2's) plus a per-link inertia factor (`generation/mjcf_builder.py`).
Every range is uniform (no normal distributions), around a Go2, and scales
with `--morphology.delta-scale`:

| Property | Randomization |
|---|---|
| Overall scale | log-uniform ¼–4 × Go2 (as likely 2× as ½×; Go2 is the median). All lengths × scale, masses × scale³ |
| Body box (length, width, height) | Go2 dimensions ± 50 % |
| Body density → body mass | Go2 density ± 50 % |
| Lower leg (radius, length) | Go2 values ± 50 % (0.213 m long, foot radius 0.022 m) |
| Upper leg | lower leg × uniform 0.75–1.25 |
| Hip offset from body centre | scales with body size × 0.9–1.1 |
| Hip link length | Go2's 0.0955 m ± 50 % (uniform), × scale |
| Link masses | Go2 masses ± 50 %, × scale³ |
| Link inertia | PyBullet's uniform-density inertia of the link's shape and mass, × uniform 0.7–1.3 per link (same factor on all three axes: mass closer to or further from the COM) |
| Max joint velocity | 6π rad/s / √scale (Froude: angular rates ~ 1/√size) |
| Knee configuration | `>>` (all knees bend backward, Go2) or `><` (rear knees bend forward, ANYmal), 50/50 (`--morphology.x-knee-fraction`). Not scaled by delta-scale. See [Knee configuration](#knee-configuration--and-) |

With 4096 robots this gives total masses of about 0.2–970 kg (p1–p99, median
13 kg; Go2 ~15 kg) and leg lengths of about 0.07–2.1 m (p1–p99, median 0.39 m;
Go2 ~0.43 m). The per-part ±50 % ranges come on top of the overall scale, so
the extremes reach past ¼–4×. Link centre-of-mass offsets are fixed (GenLoco's
A1 values), not randomized. Each robot's spawn height is computed from its
own legs so that it starts just above the ground.

On top of the morphology, the Go2 task's domain randomization still applies:
foot friction 0.3–1.6, base COM offset ± 5 cm, encoder bias ± 0.015 rad,
random pushes every 5–6 s.

### Robot and task

The generated robots follow **Unitree Go2's naming exactly**, so the Go2
velocity task runs unchanged:

| Element | Names |
|---|---|
| Base body / free joint | `base_link` / `floating_base_joint` |
| Joints (per leg `FL FR RL RR`) | `{leg}_hip_joint` (abduction), `{leg}_thigh_joint`, `{leg}_calf_joint` |
| Feet | geom `{leg}_foot_collision`, site `{leg}` |
| IMU | site `imu`; sensors `imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc`, `root_angmom` |

`assets/morph_quadruped/constants.py` mirrors `go2_constants.py`, so swapping
`get_morph_robot_cfg()` for `get_go2_robot_cfg()` switches between random
robots and Go2.

- **Actuators:** Go2's position actuators (hip/thigh kp 20, kd 1, 23.5 Nm;
  calf kp 40, kd 2, 45 Nm). For each robot, kp, kd and the torque limit are
  scaled by `robot_mass / 15.2 kg` (Go2's mass), the same way GenLoco scales its
  gains. Without this, heavy robots couldn't stand.
- **Default pose:** hips ±0.1, thighs 0.9, calves −1.8 rad.
- **Control:** 50 Hz policy (0.005 s physics step × decimation 4), actions are
  joint position offsets × 0.25 around the default pose.
- **Actor observations (51):** base angular velocity, projected gravity,
  velocity command, gait phase, joint positions, joint velocities, knee
  direction (per leg ±1: which way the knee bends, from the default pose),
  last action. The critic also gets base linear velocity, foot heights, air time,
  contacts and contact forces.
- **Rewards:** velocity tracking (linear and yaw), posture, trot gait
  (FR/RL vs FL/RR), foot clearance (0.10 m), foot slip, soft landing, body
  orientation, angular velocity and momentum, joint acceleration, joint
  limits, action rate, stand-still, termination penalty.
- **Terminations:** 20 s time-out, tilt > 70°, any non-foot collision geom
  touching the ground.
- **Command curriculum:** forward speed −0.5…1.0 m/s, widened to
  −1.0…2.0 m/s after 5000 iterations.
- **PPO:** actor/critic MLP 512-256-128 (ELU), 24 steps per env per
  iteration, 5 epochs × 4 minibatches, adaptive LR (1e-3, KL 0.01).

---

## Playing

`play.py` runs one env (`num_envs = 1`) with the trained policy. Choose the
robot with **exactly one** of:

```bash
$PY play.py --robot spot                      # by name, from ../quadrupeds
$PY play.py --random True                     # a new random robot (as in training)
$PY play.py --random True --seed 42           # a specific random robot
$PY play.py --robot-xml path/to/robot.xml     # any MJCF file
```

`ROBOT_XML` at the top of `play.py` is a placeholder path, used when no robot
option is given.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--robot NAME` | – | Robot file name in `../quadrupeds` (`spot`, `go2`, `go1`, `a1`, `anymal_b`, `anymal_c`, `barkour_v0`, …). A unique partial name works; `--robot ?` lists all |
| `--random True` | `False` | Generate and play a random robot |
| `--seed N` | random | Seed of the `--random` robot (printed every run) |
| `--robot-xml PATH` | `ROBOT_XML` | Any robot MJCF |
| `--policy PATH` | latest | Checkpoint `model_*.pt`. Default: newest checkpoint of the newest run |
| `--forward` | `auto` | Base axis the robot faces: `auto`, `+x`, `-x`, `+y`, `-y` |
| `--init-pose` | `auto` | Standing pose: `keyframe`, `train`, or `auto` (keyframe if the XML has one) |
| `--task` | `flat` | `flat` or `rough` |
| `--agent` | `trained` | `zero` / `random` send zero / random actions (checks a robot without a policy) |
| `--scale-actuators-by-mass` | `True` | Scale gains by robot mass, as in training |
| `--viewer` | `auto` | `native` (MuJoCo window), `viser` (browser), or `auto` |
| `--video True` | `False` | Record a video into the checkpoint's folder |
| `--no-terminations True` | `False` | Never reset the robot |
| `--joystick True` | `False` | Drive with a DualSense: right stick up/down = forward/back, left/right = sideways; L1 / R1 = turn left / right. Read from `--joystick-device` (`/dev/input/js0`) |
| `--joystick-max-forward/-backward/-lateral/-yaw` | task range | Full-stick speeds and L1/R1 turn rate (Go2 m/s and rad/s, scaled to the robot) |
| `--joystick-scale` | `2.0` | Multiplies all four joystick speeds |
| `--command-port P` | none | Take the command from UDP datagrams `"forward left yaw"` (Go2 units) on `127.0.0.1:P`; no datagram for 0.5 s = stand still. Used by the web harness's WASD driving |
| `--gen-params JSON` | `""` | With `--random`: generator values, as `--morphology.params` |

Booleans take a value (`--random True`), the mjlab convention.

### Commands for every bundled robot

```bash
POLICY=logs/rsl_rl/morph_quadruped_velocity/<run>/model_<N>.pt

$PY play.py --robot go2        --policy $POLICY
$PY play.py --robot go1        --policy $POLICY
$PY play.py --robot a1         --policy $POLICY
$PY play.py --robot spot       --policy $POLICY
$PY play.py --robot anymal_b   --policy $POLICY
$PY play.py --robot anymal_c   --policy $POLICY
$PY play.py --robot barkour_v0 --policy $POLICY
$PY play.py --random True      --policy $POLICY
$PY play.py --robot-xml ~/Documents/Projects/Byte01_Quadraped/Model/V2/V2.xml --policy $POLICY
```

---

## Robot convention and the adapter

The policy only depends on a robot's **structure**, not its names.
`generation/adapt.py` takes any MJCF that follows this convention:

1. **One floating base**: a body with a free joint (`<freejoint/>` or
   `<joint type="free"/>`).
2. **Four legs of three hinge joints**, ordered from the body outwards:
   **abduction** (rotates about the body's long axis), **hip** and **knee**
   (both rotate about the sideways axis). No other joints.
3. **A collision geom on each foot**: the lowest collision geom below each
   knee is taken as the foot.
4. **A known standing pose**: either zero joint angles mean *legs straight
   down* (Go2, ANYmal, Spot, …), **or** the XML has a standing `<keyframe>`.
5. **The robot file itself**, not a `scene.xml` that `<include>`s it.

Names, leg order, joint sign conventions and which base axis points forward
don't matter. The adapter takes care of those:

| Step | What it does |
|---|---|
| Assets | Collects every mesh/texture file and bundles it with the adapted robot |
| Forward axis | Finds the long axis from the abduction joints. If the robot faces ±y (common in CAD/URDF exports), rotates the base frame so it faces +x |
| Legs | Finds the 4 three-joint chains below the base. Labels each leg FL/FR/RL/RR from where it attaches (front = +x, left = +y) |
| Axes | Flips any joint whose axis points along −x/−y (and its range), so a positive angle means the same motion on every robot |
| Names | Renames to Go2 names: `base_link`, `{leg}_hip/thigh/calf_joint`, `{leg}_foot_collision`, foot sites, `imu` site. Updates any references (contact excludes, light targets, …) |
| Order | Reorders legs to the training order FR, FL, RR, RL (e.g. Go2's own file lists FL, FR, RL, RR) |
| Sensors | Adds the IMU and angular-momentum sensors the policy needs |
| Cleanup | Removes the XML's actuators (training actuators are added instead), sensors, keyframes and floor geoms |
| Pose | Uses the keyframe (converted to training joint signs) or the training pose. A value outside a joint's range is mirrored if that fits (knees that bend the other way), else clipped |

The adapted robot is written to a temporary directory
(`/tmp/play_robot_*/env_00000.xml`). The path is printed so you can inspect it.

### The fixer: `<robot>.temp.xml`

Before the adapter runs, `generation/mjcf_fix.py` fixes what it can and writes
**`<robot>.temp.xml` next to your MJCF** (git-ignored). Play, onboarding and
training read that file, not yours. It keeps your names, so you can diff it
against your file. Each change is marked with an `<!-- OnePolicy: ... -->`
comment that says what to change in your own MJCF, and a header comment lists
all changes and suggestions. `play.py` and `onboard.py` print the same summary.

| Fixed automatically | Why |
|---|---|
| Scene that `<include>`s the robot → the robot file | The adapter needs the robot itself |
| Stale `<keyframe>` (wrong qpos size) → dropped | Otherwise MuJoCo refuses the file |
| No free joint → `<freejoint/>` on the body holding the legs | The torso must float |
| Extra joints (arm, head, tail, toes) → removed, with their actuators/sensors/tendons | OnePolicy drives 4 legs × 3 hinges |
| Leg without a foot collision geom, or with a mesh/box foot → sphere at the leg tip | Training feet are round. A mesh foot does not roll (Byte01: drifted while standing) |
| Zero pose not "legs straight down" → leg bodies re-zeroed, ranges shifted | The training pose and the `knee_direction` observation assume it. Barkour's zero pose folds the legs along the body, so its front knees looked like they bent forward, which no training robot does |
| Knee layout: rear knees forward (`><`) from the knee ranges; front knees forward → robot assumed to face the other way | Ranges show which way a knee bends. The file name (`xknee`) is used only when the ranges allow both |
| Keyframe `home` = the `>>`/`><` training pose | The pose the policy was trained around |
| Joint `actuatorfrcrange` → removed (`play.py --own-motor-limits True` keeps it) | It clamps the training actuators below what the policy expects (Byte01 v2: knees clamped at 36.5 Nm, the policy expects up to ~83 Nm) |
| Leg joint damping → 0.1 (the training value) | Go2/A1/Go1 ship damping 1-2. At 0.1, Go2 tracks a 0.5 m/s command at 0.46 m/s instead of 0.30 |

**Only reported** (the sim can't change your hardware): motors weaker than
~65 % of what OnePolicy was trained with at the robot's size and mass (Go2's
peak torque × mass ratio × leg length ratio), the torque needed to stand, joint
friction, heavy lower legs and size outside the training range. The onboard
sim check also prints the peak torque the policy actually used per joint
against your motors' limits.

The re-zeroing changes what joint angle 0 means. On a real robot, the policy's
angle is `q_yours − offset` (`FixedMjcf.joint_offsets`, printed per joint).

---

## Robot compatibility

The table comes from `onboard.py --robot <name> --yes --no-viewer` with
`2026-10-17_05-05-36/model_1200.pt`, i.e. `OnePolicy/OnePolicy.pt` (an early
checkpoint, 1200 iterations).
Each robot stands 3 s under the policy with a zero command, then walks 5 s at
0.5 m/s (Go2 units, scaled to the robot). "Walked" is the distance covered out
of the ideal. "Knee torque" is the peak knee torque the policy used against
the motor limit in the robot's own MJCF, where one is set.

| Robot | Walked | Knee torque used / MJCF limit | What the fixer changed |
|---|---|---|---|
| Unitree Go2 | 2.10 / 2.50 m | 17.5 / 45.4 Nm | damping 2 → 0.1 |
| Unitree Go1 | 2.21 / 2.50 m | 11.5 / 35.5 Nm | damping 1-2 → 0.1 |
| Unitree A1 | 2.08 / 2.42 m | 11.1 / 33.5 Nm | damping 1-2 → 0.1 |
| Boston Dynamics Spot | 2.34 / 3.11 m | (±1000 Nm placeholder) | actuatorfrcrange removed, damping 0 → 0.1 |
| Spot with arm (`spot_arm`) | 2.87 / 3.11 m | (placeholder) | 7 arm joints removed (rigid arm) |
| ANYbotics ANYmal B (`anymal_b_xknee`) | 1.88 / 2.94 m | **116 / 40 Nm** | `><` from the file name, knee re-zeroed 19° |
| ANYbotics ANYmal C (`anymal_c_xknee`) | 2.69 / 3.04 m | not captured / 80 Nm | `><` from the file name, knee re-zeroed 16° |
| Google Barkour v0 | 2.32 / 2.51 m | 10.8 Nm (no limit) | legs re-zeroed (89° hip, 166° knee), round feet |
| Google Barkour vB | 1.84 / 2.53 m | 14.8 / 18.0 Nm | legs re-zeroed, round feet, standing keyframe |
| Byte01 V1 | 0.43 m/s at 0.5 (diag) | | round feet, actuatorfrcrange removed. Faces +x (the old `play.json` −x made it walk backwards) |
| Byte01 V2 | 2.34 / 2.57 m | **43.6 / 36.5 Nm** | faces +y, round feet, actuatorfrcrange removed |
| Random generated robot | ✅ | | the training distribution |

Before the fixer, Barkour v0 collapsed (its front knees looked like they
bent forward to the policy), Barkour vB and Spot-with-arm failed to load, and
Byte01 V2 walked 1.43 m with its knees clamped at 36.5 Nm.

---

## Adding your own robot

1. Check your XML against the [convention](#robot-convention-and-the-adapter).
   In particular: 12 hinge joints, the abduction → hip → knee order, and foot
   collision geoms.
2. Point `play.py` at it, first without a policy, to check the robot itself:
   ```bash
   $PY play.py --robot-xml path/to/robot.xml --agent zero
   ```
   The printed `[INFO]` lines show what the fixer and the adapter did. Read
   `path/to/robot.temp.xml`: its header lists the changes and what to change in
   your own MJCF (see [the fixer](#the-fixer-robottempxml)).
3. If it faces backwards when commanded forward, add `--forward -y` (or `-x`).
   For a robot in `quadrupeds/`, put `{"forward": "-x"}` in a `play.json`
   next to its XML so `--robot <name>` picks it up.
4. Run the headless check, `$PY onboard.py path/to/robot.xml --yes --no-viewer`.
   It stands the robot with the policy, walks it, and compares the torques the
   policy used with your motors.
5. Play it with the policy: `$PY play.py --robot-xml path/to/robot.xml`.

To make it available by name, put it in `quadrupeds/<folder>/<name>.xml` and
use `--robot <name>`. Files named `scene*.xml` or `*_mjx.xml` are ignored.

### Knee configuration: `>>` and `><`

Quadrupeds come in two leg layouts, seen from the side:

- `>>`: every knee bends backward (Go2, A1, Go1, Spot, Barkour).
- `><`: the rear knees bend forward, so all knees point towards the middle
  of the body (ANYmal B and C).

Training covers both, 50/50 (`--morphology.x-knee-fraction`). A `><` robot
gets mirrored rear knee ranges and a default pose with the rear thigh and
calf angles negated (rear legs at -0.9 / +1.8 rad instead of 0.9 / -1.8). The
policy sees which way each knee bends (the `knee_direction` observation).

The fixer reads the layout from the knee joint ranges. A knee whose range
stays on one side of "straight" can only bend that way. When the ranges allow
both directions (ANYmal: ±540°), it uses the XML's keyframe, then the file
name. **Mark such a `><` robot by its file name**: put `xknee` in it (or a
literal `><`, though that is awkward in a shell), e.g. `anymal_c_xknee.xml`.
`--robot anymal_c` keeps working (partial name). Generated `><`
robots carry the tag in their MJCF model name (`morph_quadruped_<seed>_xknee`).
The logic lives in `generation/knees.py`.

---

## Web UI

A neo-brutalist browser front end for the whole project. Everything runs on
your machine: `Web/server.py` serves the page and starts `train.py` /
`play.py` on your GPU.

```bash
sudo pacman -S --needed npm            # Arch; any Node.js >= 20 with npm works
cd Web/ui && npm install && npm run build && cd ..   # once, and after UI changes
../.venv/bin/python server.py          # then open http://localhost:8000
```

Without a build, `server.py` serves the old single-file page (also always at
`/legacy`). To work on the UI, run `npm run dev` in `Web/ui` next to
`server.py`: the dev server (port 3000) proxies `/api` to port 8000.

**Train your own robot → Quadruped**

1. Upload the robot folder (drag and drop; an animated file tree shows what
   goes up).
2. Convention check: the fixer + adapter run on it; errors come with a FIX
   hint, the joint table marks bad joints, and the XML can be edited and
   re-checked in the page. On success a popup shows its base link, a thigh
   and a calf.
3. Rewards: every term of `velocity_env_cfg.py` with its weight, parameters
   and docstring; weights and parameters are editable (terms whose weight the
   curriculum sets are shown but locked).
4. Limits: envs, iterations, checkpoint interval, learning rate, episode
   length, termination thresholds, randomisation range, `><` share.
5. Conditions: a filter on the generated robots that train next to yours
   (`--morphology.condition`), over mass, link lengths, COM, PD gains, torque
   limits, … Test it on 32 seeds before training.
6. Train: installs the robot into `quadrupeds/<name>/` and runs
   `train.py --generalise <share> --generalise-robots <name> --plot …`, from
   scratch or fine-tuning a checkpoint (OnePolicy by default). Press **X** for
   live stats (reward, episode length, action std, per-term rewards, log).
7. Policy: download the run's `model_*.pt` / `policy.onnx`, or import a `.pt`.

**OnePolicy ;) → Quadruped**

1. Pick a bundled robot (all spinning in 3D), generate one (seed + sliders
   for size, body, legs, masses and knee layout; unset sliders keep the
   seed's random draw), or upload your own (with the convention in short).
2. Its parts popup, then a terrain (only flat works; the rest are marked
   under construction).
3. `play.py --viewer viser --command-port …` runs it with the chosen policy
   (`OnePolicy/OnePolicy.pt` by default); the viewer is embedded and you drive
   with **W/S** (forward/back), **A/D** (sideways) and **←/→** (turn). Speeds
   are Go2 units, scaled to the robot. Switch to camera mode to orbit the view.

"Custom morphology" in both flows is under construction: OnePolicy is
12-DoF quadrupeds of any size.

## Morphology viewer

`Model_generation/` is a stand-alone tool to look at single generated robots
in the MuJoCo viewer (it uses GenLoco's original naming, not Go2's):

```bash
cd ~/Documents/Projects/Morphology/Model_generation
$PY generate_and_view.py                 # new random robot + viewer
$PY generate_and_view.py --seed 1234     # reproduce a robot
$PY generate_and_view.py --no-view       # only write generated/random_quadruped_<seed>.xml
```

To look at a robot from a training run instead, open its
`logs/…/morphologies/env_XXXXX.xml`, or play it with
`play.py --robot-xml logs/…/morphologies/env_00042.xml`.

---

## How per-environment robots work

MuJoCo-Warp simulates all envs as "worlds" of **one** model: they share one
model structure (bodies, joints, geoms, their types and order), while numeric
fields can be expanded to one value per world. OnePolicy relies on this:

- GenLoco's generator only randomizes **numbers**, never the structure. Every
  robot has the same 21-link tree and 12 revolute joints. `bank.py` checks
  this on every compiled robot (parent ids, joint and geom types, frame
  flags, …) and stops if any robot differs.
- Env 0's robot is compiled into the scene as the template.
- The `morphology` startup event (`tasks/velocity/mdp/morphology.py`) runs
  before every other startup event. It expands the following fields to
  per-world arrays and writes robot `i`'s values into world `i`:
  - bodies: `body_pos`, `body_quat`, `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`
  - geoms: `geom_size`, `geom_pos`, `geom_quat`, `geom_rbound`, `geom_aabb`
  - sites: `site_pos`, `site_quat`
  - actuators: `actuator_gainprm`, `actuator_biasprm`, `actuator_forcerange` (mass scaling)
- mjlab then recomputes the derived constants (`body_subtreemass`,
  `body_invweight0`, `dof_invweight0`, …) for every world, and re-captures
  the CUDA graphs.
- Each env's `default_root_state` height is set to its own standing height,
  so resets spawn every robot correctly.

Checked on the GPU model: every env's subtree mass, link offsets, collision
sizes, spawn height and gains match that robot's own compiled `MjModel`.

`play.py` reuses the same path with a one-robot bank, so a played robot gets
exactly the treatment a training robot gets.

---

## Known limitations

- **Viewer shapes:** during training, the viewer and recorded videos draw
  every env with env 0's shapes. Physics is per-robot. `play.py` (one robot)
  is always drawn correctly.
- **Joint velocity limits:** GenLoco's per-robot max joint velocity is stored
  but not enforced, because mjlab has no hook for it.
- **Joint ranges** of generated robots are GenLoco's (from the A1), not Go2's.
- **Gain scaling uses mass only:** Go2's gains are scaled by robot mass, which
  may still be too weak or too strong for robots far from Go2's proportions.
  Joint torque limits defined in a played robot's own XML (e.g.
  `actuatorfrcrange`) still apply.
- **Front vs back:** for robots whose long axis is y, `--forward auto` assumes
  +y. Use `--forward -y` if that's wrong.
- **Out-of-distribution robots:** generated robots span about 0.2–970 kg, so
  Spot (50 kg) and ANYmal C (45 kg) are inside the size range, but their
  shapes and mass distributions differ. `--generalise` trains on their own
  models.
- **Old checkpoints:** policies trained before the knee_direction
  observation (47 actor inputs) can't be resumed or played with this code
  (51 inputs).
- **Temp files:** each `play.py` run leaves its adapted robot (and a copy of
  its meshes) in `/tmp/play_robot_*`.
- **No early stopping:** training runs for `--agent.max-iterations`. Stop it
  with Ctrl-C. The latest checkpoint is saved every 100 iterations.

---

## Troubleshooting

| Message | Cause / fix |
|---|---|
| `Expected 12 hinge joints (+ free joint)` | Robot has extra joints (arm, spine, wheels) or is fixed-base |
| `… axis … is not along …; joints must be ordered abduction, hip, knee` | Joint order or axes don't follow the convention |
| `zero joint angles do not put the legs straight down` | Add a standing `<keyframe>` to the XML |
| `MJCF uses <include>` | Pass the robot file (`go2.xml`), not `scene.xml` |
| `'anymal' matches several robots` | Use the full name (`anymal_b`) or `--robot ?` |
| `Pass only one of --random, --robot and --robot-xml` | Choose one robot source |
| wandb / pydantic `ValidationError` at start of training | Use `--agent.logger tensorboard` (the default) |
| Robot walks backwards | `--forward -y` (or `-x`) |
| Robot collapses on spawn | Check gains (`--scale-actuators-by-mass`), or add a keyframe |

---

## Credits and licenses

- **Procedural generator:** `quadruped_generator.py` is from
  [GenLoco](https://github.com/HybridRobotics/GenLoco) (Feng et al.,
  *GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots*,
  CoRL 2022), under the Apache 2.0 license, with its nominal dimensions and
  masses changed from A1's to Go2's. See
  `generation/LICENSE_GenLoco.txt`.
- **Simulation and RL:** [mjlab](https://github.com/mujocolab/mjlab),
  [MuJoCo](https://github.com/google-deepmind/mujoco) /
  [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp),
  [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
- **Velocity task and Go2 config:** adapted from
  [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).
- **Robot models in `quadrupeds/`:** from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
  Each folder keeps its own `LICENSE`.
- **A1 hip mesh** (`generation/robot_descriptions/`): from Unitree's A1
  description, as shipped with GenLoco.
