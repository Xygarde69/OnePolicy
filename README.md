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

---

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Training](#training)
- [Playing](#playing)
- [Robot convention and the adapter](#robot-convention-and-the-adapter)
- [Robot compatibility](#robot-compatibility)
- [Adding your own robot](#adding-your-own-robot)
- [Morphology viewer](#morphology-viewer)
- [How per-environment robots work](#how-per-environment-robots-work)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Credits and licenses](#credits-and-licenses)

---

## Repository layout

```
Morphology/
├── Training/                         # OnePolicy: training and playing
│   ├── train.py                      # generate N robots, then train one policy
│   ├── play.py                       # play the policy on one robot
│   ├── generation/
│   │   ├── quadruped_generator.py    # GenLoco's random quadruped generator (unchanged)
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

# Play the newest policy on Spot
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
| `--morphology.delta-scale` | `1.0` | Scales every randomization range (0 gives A1-like robots) |
| `--morphology.foot-clearance` | `0.02` | Gap between lowest foot and floor at spawn [m] |
| `--morphology.workers` | all CPUs | Processes used for generation |
| `--agent.max-iterations` | `10001` | PPO iterations |
| `--agent.logger` | `tensorboard` | `tensorboard` or `wandb` |
| `--video` | `False` | Record training videos |
| `--plot` | `False` | Plot mean episode reward per iteration, overall and per leg-length third, to `<log_dir>/mean_reward.png` (+ `.csv`) |
| `--symmetry` | `True` | Left-right mirror data augmentation in PPO |
| `--generalise` | `0` | Fraction of envs (bare flag: `0.5`; `0.33` = 33%) split equally between proxies of the `--generalise-robots` (real robots rebuilt in the generated model structure, `generation/proxy.py`); the rest are generated. Reward logged per robot type |
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
(`generation/quadruped_generator.py`). Ranges are around a Unitree A1 and
scale with `--morphology.delta-scale`:

| Property | Randomization |
|---|---|
| Overall scale | uniform 0.8–1.2 (applied to body and legs) |
| Body box (length, width, height) | A1 dimensions ± 50 % |
| Body density → body mass | A1 density ± 50 % |
| Lower leg (radius, length) | A1 values ± 50 % |
| Upper leg | lower leg × uniform 0.75–1.25 |
| Hip offset from body centre | scales with body size × 0.9–1.1 |
| Hip link length | truncated normal (mean 0.081 m, std 0.05, 0.01–0.5 m) |
| Link masses | A1 masses ± 50 %, × scale³ |
| Max joint velocity | uniform π–6π rad/s (recorded, not enforced, see limitations) |

With 4096 robots this gives total masses of about 4–40 kg and standing
heights of about 0.15–0.6 m. Each robot's spawn height is computed from its
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
- **Actor observations (47):** base angular velocity, projected gravity,
  velocity command, gait phase, joint positions, joint velocities, last
  action. The critic also gets base linear velocity, foot heights, air time,
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

---

## Robot compatibility

The table below comes from 2 s headless runs with a trained checkpoint. Each
robot builds, loads the policy and steps without falling. These runs test
**compatibility, not walking quality**. How well each robot walks depends on
how far training has progressed.

| Robot | Result | Notes |
|---|---|---|
| Unitree Go2 | ✅ runs | keyframe pose. Legs reordered from FL, FR, RL, RR |
| Unitree Go1 | ✅ runs | keyframe pose |
| Unitree A1 | ✅ runs | keyframe pose |
| Boston Dynamics Spot | ✅ runs | keyframe pose, 50 kg → gains × 3.3 |
| ANYbotics ANYmal B | ✅ runs | no keyframe → training pose (unlimited joints) |
| ANYbotics ANYmal C | ✅ runs | no keyframe → training pose |
| Google Barkour v0 | ✅ runs | knee-forward legs: 6 joint axes flipped, keyframe pose |
| Byte01 V1 (`--robot byte01_v1`) | ✅ runs | faces −x (`play.json`) → base rotated; calf meshes are the feet. Training pose |
| Byte01 V2 (`--robot byte01_v2`) | ✅ runs | faces +y → base rotated; calf meshes are the feet. Training pose |
| Byte01 V2 primitive (`--robot byte01_v2_primitive`) | ✅ runs | mesh-free V2; base/thighs are visual-only, so body–ground contact is not detected |
| Random generated robot | ✅ runs | the training distribution |
| Google Barkour vB | ❌ stops | zero pose has folded legs and there is no keyframe. Add a standing keyframe |
| Spot with arm (`spot_arm`) | ❌ stops | arm joints → more than 12 hinge joints |

---

## Adding your own robot

1. Check your XML against the [convention](#robot-convention-and-the-adapter).
   In particular: 12 hinge joints, the abduction → hip → knee order, and foot
   collision geoms.
2. Point `play.py` at it, first without a policy, to check the robot itself:
   ```bash
   $PY play.py --robot-xml path/to/robot.xml --agent zero
   ```
   The printed `[INFO]` lines show what the adapter did (flipped axes, rotated
   base, chosen pose).
3. If it faces backwards when commanded forward, add `--forward -y` (or `-x`).
   For a robot in `quadrupeds/`, put `{"forward": "-x"}` in a `play.json`
   next to its XML so `--robot <name>` picks it up (see `quadrupeds/byte01_v1`).
4. If it spawns badly or its legs fold, add a standing pose to the XML:
   ```xml
   <keyframe>
     <!-- base x y z, quat w x y z, then the 12 joint angles in the XML's order -->
     <key name="home" qpos="0 0 0.35  1 0 0 0  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8"/>
   </keyframe>
   ```
5. Play it with the policy: `$PY play.py --robot-xml path/to/robot.xml`.

To make it available by name, put it in `quadrupeds/<folder>/<name>.xml` and
use `--robot <name>`. Files named `scene*.xml` or `*_mjx.xml` are ignored.

---

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
- **Out-of-distribution robots:** Spot (50 kg) and ANYmal C (45 kg) are heavier
  than the 4–40 kg training robots. Expect weaker performance there unless
  training covers them.
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
  CoRL 2022), used unchanged under the Apache 2.0 license. See
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
