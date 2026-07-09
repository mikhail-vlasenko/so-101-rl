# autoresearch

Autonomous RL research loop for the SO-101 **lift** task.

## Goal

Refine an *already-trained* lift policy so the arm **grasps and lifts the cube on
the first try**. The current checkpoint eventually lifts, but it fumbles: it makes
many grasp attempts, often grips wrong, tips the sponge over, or pushes it away
before finally succeeding. We want a clean, confident, single-attempt lift.

The metric for "first try" is **mean episode length**: the lift episode
terminates the moment the arm holds a *grasped* cube at target height, so a policy
that lifts immediately ends episodes early (low length) while a fumbling or failing
policy runs the full 300 steps. **Lower mean episode length is better**, read
together with `success_rate` (length is only "fast" if the arm actually lifts).

We are **continuing training** from a fixed base checkpoint, not training from
scratch. Each experiment resumes the same checkpoint and gets a short fine-tuning
budget; the question every run answers is "does this change make the *already
competent* policy lift sooner and more reliably?"

## Base checkpoint (fixed)

```
logs/ppo_lift/stage2_no_tag_rotation.zip
```

Every experiment resumes from **this exact file** so reward/config changes are
comparable against the same starting policy. Training writes `final_model.zip` /
`best_model.zip`, never this file, so it stays intact — do not overwrite it. It was
trained with `marker_include_rot=false`, `prev_actions_n=1`, `history_taps=[0]`; those
change obs dim and **must not change** (a resume would crash on dim mismatch).

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun15`).
2. **Create the branch**: `git checkout -b autoresearch-lift-<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - `src/base_env.py` — base Gymnasium env (observation space, MuJoCo setup, contact/grasp detection, obs noise/bias/latency, prev-actions buffer, servo profile)
   - `src/lift_env.py` — lift task env (reward function, reset logic, success/termination criterion)
   - `src/train.py` — training script (algorithm setup, callbacks, resume, model construction)
   - `src/callbacks.py` — metric logging (incl. `EpisodeLengthCallback`, `LiftSuccessCallback`)
   - `src/networks.py` — policy network architecture
   - `conf/config.yaml` — shared hyperparameters (action_scale, use_servo_profile, prev_actions_n, train.*)
   - `conf/env/lift.yaml` — lift-specific config overrides (max_steps, cube spawn range, target_height)
   - `conf/dr/light.yaml`, `conf/shaping/full.yaml` — the DR / careful-behavior presets used below
4. **Initialize results.tsv**: create with just the header row (see *Logging results*).
5. **Confirm and go**.

## Experimentation

Each run fine-tunes the base checkpoint with a **fixed time budget of 10 minutes**.
Launch (from the repo root, `mujoco_env` conda env):

```bash
conda run -n mujoco_env python -m src.train \
    env=lift dr=light \
    resume=logs/ppo_lift/stage2_no_tag_rotation.zip \
    train.time_limit_minutes=10 \
    > run.log 2>&1
```

Launch it via the agent's background-execution mechanism (not a shell `&` —
`conda run`'s output buffering plus `&` has been flaky). Redirect everything; do
NOT let training output flood your context.

After each run, fetch the metrics:

```bash
conda run -n mujoco_env python -m src.fetch_wandb
```

This reads the most recent W&B run and prints last-minute averages for:
- `rollout/lift/mean_ep_length` — **PRIMARY, lower is better** (steps to a grasped lift; max 300)
- `rollout/lift/success_rate` — fraction of episodes ending in a grasped lift to target (higher is better)
- `rollout/lift/mean_max_cube_height` — diagnostic only

Judge a run on **mean_ep_length**, but only count it if `success_rate` did not drop:
shaving length while lifting *less often* is not progress.

**What you CAN modify:**
- `src/base_env.py` — shared base env (observation space, contact/grasp detection, obs pipeline)
- `src/lift_env.py` — reward shaping, observation space, reset logic, success/termination criterion
- `src/train.py` — algorithm choice, model construction, callbacks
- `src/callbacks.py` — logging, custom callbacks
- `src/networks.py` — policy/value network architecture
- `conf/config.yaml` — hyperparameters (LR, batch size, gamma, net arch, ent_coef, etc.)
- `conf/env/lift.yaml` — lift-specific config (max steps, cube spawn range, target_height, etc.)
- `conf/shaping/full.yaml` — careful-behavior shaping coefficients (kept on; see below)

**What you CANNOT modify / must hold fixed:**
- `src/fetch_wandb.py` — metric fetching (read-only; do not retarget the metric to flatter a run)
- `so101/` — robot model and scene XMLs (read-only)
- The base checkpoint file and the obs-dim-defining knobs (`marker_include_rot`,
  `prev_actions_n`, `history_taps`) — changing them breaks the resume.

**Non-negotiable sim-to-real constraints.** This policy deploys on the real arm.
Do NOT chase the metric by making the sim easier to transfer *from*:

- **`use_servo_profile` must stay `true`.** The baked `so101.xml` sysid fit assumes
  the firmware motion profile carries the actuator lag; turning it off makes the
  sim under-damped and unlearnable-on-real. Never set it false.
- **Domain randomization floor is `dr=light`.** `dr=light` is acceptable for this
  refine (it keeps full joint-encoder noise, eases vision noise). Do NOT use
  `dr=none` or otherwise reduce/zero the `dr=light` noise, bias, latency, or marker
  dropout. You MAY *add* new randomization (mass/friction jitter, etc.). A better
  metric obtained by weakening DR is a regression — mark it `discard`.
- **Do not make the arm physically faster or the task coarser to win.** Off-limits:
  raising `action_scale`, the servo speed/accel registers, the MuJoCo timestep,
  `n_substeps` / control frequency, or shrinking the cube-spawn range / lowering
  `target_height`. These shorten episodes without the policy actually getting
  better, and most break sim-to-real. Keep the careful-behavior `shaping=full`
  (default) on — it's part of why the real arm doesn't slam the table.

**The goal: minimize `rollout/lift/mean_ep_length` while keeping `success_rate`
high.** Everything within the constraints above is fair game — reward shaping is
explicitly encouraged. Ideas to seed the loop (combine, tune, or discard):
- Penalize *repeated* failed grasp attempts (gripper open→close cycles, or
  re-approaches after losing contact) so the arm commits to one clean grasp.
- Strengthen the pre-grasp "don't disturb the cube" terms (`CUBE_MOTION_COEFF`,
  `cube_tip_coeff`, `poke_force_coeff`) so it stops nudging/tipping the sponge.
- Penalize pushing the cube away from its spawn xy before grasping.
- Reward a fast, monotonic approach-then-grasp; discourage dithering near the cube.
- Tune the grasp gate / closedness shaping so a confident full close beats a
  tentative pinch.

**Simplicity criterion**: all else equal, simpler is better. A small improvement
that adds ugly complexity is not worth it. Removing code for equal-or-better
results is a win.

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do NOT commit this file.

Header and 6 columns:

```
commit	mean_ep_length	success_rate	timesteps_M	status	description
```

1. git commit hash (short, 7 chars)
2. mean_ep_length over the last minute (lower better, e.g. 142.3)
3. success_rate over the last minute (e.g. 0.78)
4. total timesteps fine-tuned in this run, in millions (e.g. 3.1)
5. status: `keep`, `discard`, or `crash`
6. short description of what was tried

Example:

```
commit	mean_ep_length	success_rate	timesteps_M	status	description
a1b2c3d	178.0	0.71	3.0	baseline	resume base, dr=light, no changes
b2c3d4e	152.4	0.74	3.0	keep	+ penalty on gripper open/close cycles
c3d4e5f	171.0	0.55	3.0	discard	cube_tip_coeff -0.05→-0.3: stops touching cube
d4e5f6g	0.0	0.0	0.0	crash	obs-dim mismatch (changed prev_actions_n)
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit).
2. Make a change — edit code or config with an experimental idea.
3. `git commit` the change.
4. Run the launch command above (resume from the **fixed base checkpoint**, 10-min budget).
   - **Timeout**: each run is ~10 min. If it exceeds 15 minutes, kill it and treat as failure.
5. Read results: `conda run -n mujoco_env python -m src.fetch_wandb`.
6. If fetch fails or the metric is missing, check `tail -50 run.log` for errors.
7. Record in `results.tsv`.
8. If `mean_ep_length` dropped **and** `success_rate` held → keep the commit, advance the branch.
9. If equal/worse, or success_rate dropped, or the gain came from weakening DR /
   making the arm faster → `git reset --hard HEAD~1` to revert.

**Crashes**: if it's a typo/easy fix, fix and re-run. If fundamentally broken, log
as crash, revert, move on. The most common crash here is an obs-dim mismatch on
resume — never change `marker_include_rot`, `prev_actions_n`, or `history_taps`.

**Baseline first**: the first run should be a no-change resume to establish the
base `mean_ep_length` / `success_rate` that every later run is judged against.

**NEVER STOP**: do not pause to ask the human. You are autonomous. If you run out
of ideas, re-read the in-scope files, combine previous near-misses, or try more
radical reward redesigns. The loop runs until the human interrupts you.
