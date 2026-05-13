# autoresearch

Autonomous RL research loop for the SO-101 **lift** task.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may14`).
2. **Create the branch**: `git checkout -b autoresearch-lift-<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - `base_env.py` — base Gymnasium env (observation space, MuJoCo setup, contact detection, obs noise/bias/latency, prev-actions buffer)
   - `lift_env.py` — lift task env (reward function, reset logic, success criterion)
   - `train.py` — training script (algorithm setup, callbacks, model construction)
   - `networks.py` — policy network architecture
   - `conf/config.yaml` — shared hyperparameters (incl. obs_noise, obs_bias, obs_latency, action_scale)
   - `conf/env/lift.yaml` — lift-specific config overrides
4. **Initialize results.tsv**: Create with just the header row.
5. **Confirm and go**.

## Experimentation

Training runs with a **fixed time budget of 20 minutes** (configured in `conf/config.yaml` under `train.time_limit_minutes`). Launch:

```bash
conda run -n mujoco_env python train.py env=lift > run.log 2>&1
```

After each run, fetch the key metric:

```bash
conda run -n mujoco_env python fetch_wandb.py
```

This prints the average `rollout/lift/mean_max_cube_height` over the last minute of training — the average peak height (in metres) the cube reached during each episode. Higher is better; the lift is considered solved when episodes consistently reach the configured `target_height` (currently 0.10 m).

**What you CAN modify:**
- `base_env.py` — shared base env (observation space, MuJoCo setup, contact detection)
- `lift_env.py` — reward shaping, observation space, reset logic, success criterion
- `train.py` — algorithm choice, model construction, callbacks
- `callbacks.py` — logging, custom callbacks
- `networks.py` — policy/value network architecture
- `conf/config.yaml` — hyperparameters (LR, batch size, gamma, net arch, etc.)
- `conf/env/lift.yaml` — lift-specific config (max steps, substeps, cube spawn range, penalties, target_height)

**What you CANNOT modify:**
- `fetch_wandb.py` — metrics fetching (read-only)
- `so101/` — robot model and scene XMLs (read-only).

**Domain randomization is non-negotiable.** The whole point of training in sim is to transfer to the real arm. Do NOT weaken the sim-to-real defenses to chase a higher metric:

- `obs_noise` (per-step Gaussian noise on qpos/qvel/ee/cube): the configured sigmas in `conf/config.yaml` are calibrated against the real sensors. Do not reduce, zero out, or remove.
- `obs_latency` (frames the agent's obs lags behind true state): currently 2 frames at 30 Hz ≈ 67 ms. Models real comms + camera latency. Do not reduce.

You may *add* new randomization (mass/friction jitter, random initial poses, etc.). You may not weaken existing randomization. A higher metric obtained by removing DR is a regression, not progress — mark it `discard`.

**The goal: maximize `rollout/lift/mean_max_cube_height`** (averaged over the last minute of training). The cube starts on the table in a random position; the agent must grasp it and lift it as high as possible, ideally above `target_height=0.10 m`. Everything within the constraints above is fair game: reward shaping, observation design, hyperparameters, network architecture.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code for equal or better results is a win.

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do NOT commit this file.

Header and 5 columns:

```
commit	mean_max_cube_height	timesteps_M	status	description
```

1. git commit hash (short, 7 chars)
2. mean_max_cube_height over last minute (e.g. 0.045)
3. total timesteps in millions (e.g. 10.2)
4. status: `keep`, `discard`, or `crash`
5. short description of what was tried

Example:

```
commit	mean_max_cube_height	timesteps_M	status	description
a1b2c3d	0.024	10.2	keep	baseline
b2c3d4e	0.041	10.0	keep	gamma 0.95 → 0.97
c3d4e5f	0.020	10.1	discard	larger net (overfits in 20 min)
d4e5f6g	0.000	0.0	crash	wrong activation in policy head
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit).
2. Make a change — edit code or config with an experimental idea.
3. `git commit` the change.
4. Run: `conda run -n mujoco_env python train.py env=lift > run.log 2>&1`
   - Redirect everything. Do NOT let output flood your context.
   - **Timeout**: Each run should take ~20 minutes. If it exceeds 30 minutes, kill it and treat as failure.
5. Read results: `conda run -n mujoco_env python fetch_wandb.py` (reads from the most recent W&B run automatically).
6. If fetch_wandb fails or the metric is missing, check `tail -50 run.log` for errors.
7. Record in `results.tsv`.
8. If `mean_max_cube_height` improved → keep the commit, advance the branch.
9. If equal or worse → `git reset --hard HEAD~1` to revert. Also revert if the metric improved only because DR was weakened.

**Crashes**: If it's a typo/easy fix, fix and re-run. If fundamentally broken, log as crash, revert, move on.

**NEVER STOP**: Do not pause to ask the human. You are autonomous. If you run out of ideas, re-read the in-scope files, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
