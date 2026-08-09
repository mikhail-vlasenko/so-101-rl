# autoresearch

Autonomous RL research loop for the SO-101 **lift** task, stage 1 of the
curriculum, on the **tag-free dual-channel cube observation**.

## Goal

Make stage-1 lift **take off from scratch**. With the cube channels replacing the
cube AprilTag (obs 37 -> 90 dims, commit `5e2d3af`), a fresh curriculum run no
longer learns to lift: three chained 20-minute runs on 2026-07-25/26
(`qaa1dl7u`, `g44e901z`, `firg205n`) sat at exactly **0.000 success** with
`mean_max_cube_height` creeping 0.030 -> 0.039 m, i.e. the arm nudges the sponge
and never lifts it. The old 37-dim curriculum on the same stage-1 config reached
0.16 success in its second 20-minute chunk and 0.66 by the sixth (`gte4tw3r` ->
`ul0tdtfy`, 2026-07-05).

So the question every experiment answers is: **does this change get a
from-scratch policy to a grasped lift within the run budget?** Not "does it lift
sooner" — that was the previous loop's objective (see *Prior loop*) and it is
meaningless while success is zero.

There is deliberately **no base checkpoint**. Nothing competent exists at 90 dims,
and `src/distill.py` cannot migrate the old tag-obs teachers (its modes are
`identical` / `current` / `privileged`; the teacher would need a cube-tag obs that
no longer exists). Every run starts from random init, which also makes the runs
directly comparable.

## Fixed experiment protocol

Every run uses **exactly** this command — the config after `python -m src.train`
is the stage definition, not a tunable:

```bash
conda run -n mujoco_env python -m src.train \
    env=lift dr=none shaping=none \
    marker_always_visible=true cube_smallest_face_only=true cube_no_flat_spawns=true \
    seed=0 train.time_limit_minutes=25 \
    > run.log 2>&1
```

Launch it via the agent's background-execution mechanism (not a shell `&` —
`conda run`'s output buffering plus `&` has been flaky). Redirect everything; do
NOT let training output flood your context.

- **25 minutes** (~10M steps at 16 envs). Shorter cannot resolve takeoff: the
  historical 37-dim curriculum needed ~20-40 minutes before the first lifts.
- **`seed=0` pinned.** From-scratch runs are noisy; a fixed seed keeps the
  comparison about the change. Once a change looks like a winner, **re-run it at
  `seed=1`** before keeping it — a single-seed takeoff is a coin flip, not a
  result.
- **Timeout**: if a run exceeds 35 minutes, kill it and treat it as a failure.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jul26`).
2. **Create the branch**: `git checkout -b autoresearch-lift1-<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - `src/base_env.py` — base env: observation assembly, camera/object-channel
     ingestion, contact/grasp detection, spawn sampling, privileged tail
   - `src/shape_obs.py` — the dual-channel cube observation driver (**read-only**,
     see below)
   - `src/lift_env.py` — lift reward, success/termination criterion
   - `src/obs_norm.py` — the fixed affine that normalizes the 90-dim obs,
     including the new cube channels
   - `src/networks.py` — policy/value nets, `TakeFirst` actor/critic split
   - `src/train.py`, `src/callbacks.py` — algorithm setup, metric logging
   - `conf/config.yaml`, `conf/env/lift.yaml`, `conf/dr/none.yaml`,
     `conf/shaping/none.yaml` — the composed stage-1 config
4. **Initialize results.tsv**: create with just the header row (see *Logging
   results*). The previous loop's file is archived as
   `results_lift_first_try.tsv` — different task, different metric, do not append
   to it.
5. **Confirm and go.**

## Reading a run

```bash
conda run -n mujoco_env python -m src.fetch_wandb
```

It prints last-minute averages **and the peak over the run** for a ladder of
metrics, ordered by how early they move:

- `rollout/lift/success_rate` — **PRIMARY, higher is better.** Fraction of
  episodes ending in a grasped lift to `target_height`.
- `rollout/lift/ever_grasped` — fraction of episodes with at least one grasped
  step. Moves before success does; this is the metric that separates "not
  learning" from "learning the wrong half".
- `rollout/lift/grasp_ratio` — mean fraction of steps grasped (grasp stability).
- `rollout/lift/mean_max_cube_height` — moves before grasp does. Baseline ~0.030
  (the sponge resting on its smallest face); anything at 0.03x means the cube
  never left the table.
- `rollout/lift/mean_ep_length` — diagnostic only here; pinned at 300 until the
  policy starts terminating episodes.

**Judging a run against the baseline**, in order: higher `success_rate` wins.
If both are zero, higher `ever_grasped` wins. If those are also equal, higher
`mean_max_cube_height` wins — but treat a height-only gain as *weak* evidence and
never keep more than two consecutive height-only wins without a grasp appearing;
that is how a loop talks itself into a local optimum where the arm shoves the
sponge around forever.

If the fetch fails or a metric is missing, check `tail -50 run.log`.

**What you CAN modify:**
- `src/lift_env.py` — reward shaping, success/termination criterion
- `src/base_env.py` — obs assembly, grasp/contact detection, spawn sampling
- `src/obs_norm.py` — the fixed obs affine. The cube channels are new and their
  normalization constants were never validated against a trained policy; a
  mis-scaled input block is a live hypothesis for the stall. Changing it
  invalidates old checkpoints, which costs nothing here (we train from scratch),
  but the real rollout reads the same constants — so it must stay a single
  source, never a train-only tweak.
- `src/networks.py` — architecture, the actor/critic split
- `src/train.py`, `src/callbacks.py` — algorithm, callbacks, logging
- `conf/config.yaml` — hyperparameters (LR, batch, gamma, ent_coef, net_arch,
  n_steps, n_epochs, ...)
- `conf/env/lift.yaml` — `max_steps`, `target_height` is NOT free (see below)

**What you CANNOT modify / must hold fixed:**
- **The stage config in the launch command.** `dr=none`, `shaping=none`, all
  three crutch flags on, `seed=0`, 25 minutes. These *define* the experiment;
  changing them changes the question, and turning a crutch on that is already on
  is the only cheat available at this stage.
- `src/{shape_obs,bps}.py` — single-sourced live/static and precise hold state,
  pinned by `tests/real/test_object_twin.py`.
  Never fork it. If the object-channel *semantics* genuinely need to change, that
  is a repo-level decision, not an experiment.
- `conf/dr/*.yaml` cube sigmas — measured from the sponge dataset
  (`datasets/sponge_20260725_213859`), not difficulty knobs. Stage 1 runs at
  `dr=none` anyway; do not "help" later stages by shrinking them.
- `src/fetch_wandb.py` — read-only. Do not retarget the metric to flatter a run.
- `so101/` — robot model and scene XMLs.
- `marker_include_rot`, `prev_actions_n`, `history_taps` — they change obs dim.
  Nothing resumes here so nothing would crash, but the real rollout and every
  contract test assume the current layout, and a dim change makes every earlier
  row in `results.tsv` incomparable.

**Non-negotiable sim-to-real constraints.** This policy deploys on the real arm.
Do NOT chase the metric by making the sim easier to transfer from:

- **`use_servo_profile` must stay `true`.** The baked `so101.xml` sysid fit was
  refit *with* the profile; without it the sim is under-damped and
  unlearnable-on-real.
- **Do not make the arm faster or the task coarser.** Off-limits: raising
  `action_scale`, servo speed/accel registers, the MuJoCo timestep, `n_substeps` /
  control frequency, shrinking the cube-spawn range, or lowering `target_height`.
  A lower target height "solves" takeoff by redefining a lift.
- **The stage-1 crutches are already the easy setting.** `dr=none` +
  `marker_always_visible` + `cube_smallest_face_only` + `cube_no_flat_spawns` is
  as forgiving as the task gets. There is nothing left to weaken — if a change
  only works because it made the task easier in some new way, mark it `discard`.
- **Tests are a gate.** Before committing any change to `src/`, run
  `pytest tests/ -q`. A change that breaks the sim/real twin contracts
  (`tests/real/test_object_twin.py`, the marker hold-last-pose tests) is an
  automatic `discard` no matter what the metric did.

**Ideas to seed the loop** (combine, tune, or discard):
- The reward ladder was tuned against the *tag* cube obs, where cube position was
  effectively exact. The live channel is a visible-surface centroid, biased toward
  the camera-facing surface — `EE_CUBE_COEFF` now pulls the gripper toward a point
  that is not the cube center, and `_detect_grasp` is GT-based while the policy
  aims with a biased estimate. Check whether the shaping gradient still points at
  a graspable pose.
- The critic receives 39 dims of GT state and DR latents beyond the 103-dim
  default actor input. If it can trivially predict returns from GT cube state,
  the advantage signal the actor sees may have gone quiet — worth measuring
  before tuning anything else.
- The 64 BPS distances are clean under `dr=none`; verify ablations show the
  policy uses them before spending capacity on a wider network.
- Exploration: entropy, action noise, or a shorter `max_steps` at stage 1 so the
  policy sees more resets per unit of experience.
- LR / batch / `n_epochs`: the previous loop found `3e-4` unstable for fine-tuning
  a competent policy, which says nothing about from-scratch learning — retune here.

**Simplicity criterion**: all else equal, simpler is better. A small improvement
that adds ugly complexity is not worth it. Removing code for equal-or-better
results is a win.

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do NOT commit this file.

Header and 7 columns:

```
commit	success	ever_grasped	max_height	timesteps_M	status	description
```

1. git commit hash (short, 7 chars)
2. `success_rate` — peak over the run (e.g. 0.12); `0.000` is a normal result here
3. `ever_grasped` — peak over the run (e.g. 0.31)
4. `mean_max_cube_height` — last-minute average (e.g. 0.041)
5. total timesteps trained in this run, in millions (e.g. 10.2)
6. status: `keep`, `discard`, or `crash`
7. short description of what was tried

Example:

```
commit	success	ever_grasped	max_height	timesteps_M	status	description
a1b2c3d	0.000	0.021	0.039	10.1	baseline	from-scratch stage 1, no changes
b2c3d4e	0.000	0.180	0.052	10.0	keep	EE_CUBE target = precise center, not live centroid
c3d4e5f	0.000	0.004	0.031	10.2	discard	ent_coef 0.001->0.01: dithers, never approaches
d4e5f6g	0.000	0.000	0.030	0.0	crash	obs_norm shape mismatch on the sqrtM block
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit).
2. Make a change — edit code or config with an experimental idea.
3. Run `pytest tests/ -q`. If it fails, fix before launching.
4. `git commit` the change.
5. Launch the fixed command above (25-minute budget, `seed=0`).
6. Read results: `conda run -n mujoco_env python -m src.fetch_wandb`.
7. Record in `results.tsv`.
8. If the run beat the baseline on the ladder above → **re-run at `seed=1`** to
   confirm, log that row too, and only then keep the commit and advance the
   branch (the new result becomes the baseline to beat).
9. Otherwise → `git reset --hard HEAD~1` and move on.

**Crashes**: if it's a typo/easy fix, fix and re-run. If fundamentally broken, log
as crash, revert, move on.

**Baseline first**: the first run must be a no-change from-scratch run at `seed=0`,
establishing the `success` / `ever_grasped` / `max_height` every later run is
judged against. Expect it to reproduce ~0.000 / low / ~0.03x.

**A note on what "no progress" means here**: unlike the previous refine loop, a
result of 0.000 success is *expected* for most experiments. Do not respond by
loosening the objective. The ladder metrics exist precisely so that a run with
zero success still carries information.

**NEVER STOP**: do not pause to ask the human. You are autonomous. If you run out
of ideas, re-read the in-scope files, combine previous near-misses, or try more
radical redesigns of the reward or the observation scaling. The loop runs until
the human interrupts you.

## Prior loop

`results_lift_first_try.tsv` + `AUTORESEARCH_REPORT.md` hold the earlier loop,
which fine-tuned an already-competent **37-dim tag-obs** policy to lift on the
first try (metric: minimize `mean_ep_length`). It ended at 184 steps / 0.75
success and concluded the ceiling was grasp security, not shaping — twelve
experiments, of which only the LR drop (3e-4 -> 3e-5, for fine-tuning stability)
was kept. Those checkpoints and that metric no longer apply, but the negative
results are still worth reading before repeating them.
