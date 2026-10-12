# CLAUDE.md

OnePolicy: one velocity-tracking policy trained (mjlab + MuJoCo-Warp, rsl_rl
PPO) on thousands of procedurally generated quadrupeds, one robot per env. See
README.md for layout, setup and usage. Code lives in `Training/`; run scripts
from there with the repo's `.venv` (`../.venv/bin/python`).

## Workflow

- Commit regularly, one logical change per commit, and push after committing:
  `git push -u origin main`.
- Don't commit `MUJOCO_LOG.TXT`, the reference PDFs, or `Training/logs/`.
- Smoke test without training: `python train_test.py --viewer False
  --num-robots 2 --max-seconds 3` (random actions; early illegal_contact
  terminations are expected, crashes/NaNs are not).

## Conventions

- Every Go2-tuned threshold is rescaled per robot by Froude similarity
  (`tasks/velocity/mdp/scaling.py`): s_L = leg length / Go2's, s_m = mass /
  Go2's, speed and time ~ sqrt(s_L), force ~ s_m, torque ~ s_m * s_L. New
  reward terms must normalise to Go2 units the same way, otherwise large or
  small robots dominate the loss. Compare commands against thresholds via
  `normalized_command`.
- Foot order in the contact sensors and sites is FR, FL, RR, RL.
- Actuators are mjlab `DcMotorActuator`s (`assets/morph_quadruped/constants.py`):
  torque is clipped to a linear torque-speed band, wider in the braking
  quadrants. This is already the "motor operating region" constraint from
  KAIST Hound (below); keep it when changing actuators.

## Reward design and why

Base rewards come from mjlab's Go2 velocity task. In Oct 2026 they were extended
following KAIST Hound, "Actuator-Constrained RL for High-Speed Quadrupedal
Locomotion" (arXiv 2312.17507, Table I). That paper's weights are relative to a
tracking weight of 6; ours are divided by 6 to sit relative to our tracking
weight of 1.

Added (`tasks/velocity/mdp/rewards.py`, weights in `velocity_env_cfg.py`):

| Term | Weight | Why |
|---|---|---|
| `foot_contact_velocity` | -0.5 | Squared foot vertical speed the step before touchdown. Cleaner impact signal than touchdown force sampled at 50 Hz. Divided by s_L. |
| `diagonal_gait` | +0.2 | Bonus when exactly one diagonal pair is in stance, no clock. The paper's fix for one motor saturating first; spreads torque over all legs. Only above 1.5 m/s (Go2 units) where the clocked `foot_gait` period caps stride frequency. |
| `joint_torques_l2` (`scaled_joint_torques_l2`) | -1e-4 | Torque cost was missing. Torque divided by s_m * s_L before squaring. |
| `joint_vel_l2` (`scaled_joint_vel_l2`) | -5e-5 | Joint speed cost was missing. Squared speed multiplied by s_L. |
| `action_acc_l2` | -0.01 | Second-order action smoothness. The paper shows this is what makes the policy *learn* to avoid torque commands outside the motor band (clipping alone hides them from the torque penalty); without it their robot vibrated. |

Changed:

- `soft_landing` -1e-3 -> -5e-4: impacts are now mostly handled by
  `foot_contact_velocity`.
- Curriculum `track_linear_velocity_weight` (1.0 -> 2.0) and
  `track_angular_velocity_weight` (1.0 -> 1.5), switching at step 5000*24
  alongside the command-range widening. The paper found that as speed grows the
  effort penalties grow too and the policy gives up on tracking fast commands.

Deliberately unchanged: `action_rate_l2` (-0.05, already comparable to the
paper's after the 0.25 action scale), `joint_acc_l2` (same order after unit
conversion), and the walking/standing shaping terms (`foot_gait`, `pose`,
`stand_still`, `is_terminated`), which the paper does not cover.

Tuning notes: the torque weight is an estimate (the paper's robot is 45 kg, ours
are Go2-sized); if low-speed tracking stalls early in training, halve
`joint_torques_l2` first. `diagonal_gait` only matters once the command
curriculum goes well past ~1.5 m/s.

Coupling: `tasks/velocity/rl/runner.py` boosts command ranges x5 when the mean
episode reward passes `YAW_BOOST_REWARD` / `LIN_BOOST_REWARD`. Any reward
change shifts the episode-reward scale (the new penalties lower it, the
tracking-weight ramp raises it), so re-check those thresholds against the
training curves whenever rewards or weights change.
