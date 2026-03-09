# autoresearch

Autonomous RL research loop for the SO-101 pick-and-place task.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - `base_env.py` — base Gymnasium env (observation space, MuJoCo setup, contact detection)
   - `pickplace_env.py` — pick-and-place task env (reward function, reset logic, success criteria)
   - `train.py` — training script (algorithm setup, callbacks, model construction)
   - `networks.py` — policy network architecture
   - `conf/config.yaml` — shared hyperparameters
   - `conf/env/pickplace.yaml` — pickplace-specific config overrides
4. **Initialize results.tsv**: Create with just the header row.
5. **Confirm and go**.

## Experimentation

Training runs with a **fixed time budget of 20 minutes** (configured in `conf/env/pickplace.yaml`). Launch:

```bash
conda run -n mujoco_env python train.py env=pickplace > run.log 2>&1
```

After each run, fetch the key metric:

```bash
conda run -n mujoco_env python fetch_wandb.py
```

This prints the average `rollout/pickplace/completion_rate` over the last minute of training — the fraction of episodes where the cube was successfully placed in the target ring.

**What you CAN modify:**
- `pickplace_env.py` — reward shaping, observation space, reset logic, success criteria
- `train.py` — algorithm choice, model construction, callbacks
- `callbacks.py` — logging, custom callbacks
- `networks.py` — policy/value network architecture
- `conf/config.yaml` — hyperparameters (LR, batch size, gamma, net arch, etc.)
- `conf/env/pickplace.yaml` — pickplace-specific config (action scale, max steps, substeps, cube spawn range, penalties)

**What you CANNOT modify:**
- `base_env.py` — shared base env (read-only)
- `fetch_wandb.py` — metrics fetching (read-only)
- `so101/` — robot model and scene XMLs (read-only)

**The goal: maximize `completion_rate`.** The cube starts on the table in a random position. The agent must pick it up and place it in the target ring. A completion_rate of 1.0 means every episode succeeds. Higher is better. Everything is fair game: reward shaping, observation design, hyperparameters, network architecture, action scale, physics parameters.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code for equal or better results is a win.

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do NOT commit this file.

Header and 5 columns:

```
commit	completion_rate	timesteps_M	status	description
```

1. git commit hash (short, 7 chars)
2. completion_rate over last minute (e.g. 0.15)
3. total timesteps in millions (e.g. 2.5)
4. status: `keep`, `discard`, or `crash`
5. short description of what was tried

Example:

```
commit	completion_rate	timesteps_M	status	description
a1b2c3d	0.00	5.0	keep	baseline
b2c3d4e	0.05	4.8	keep	increase action_scale to 0.1
c3d4e5f	0.02	5.2	discard	switch to SAC
d4e5f6g	0.00	0.0	crash	double net width (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit).
2. Make a change — edit code or config with an experimental idea.
3. `git commit` the change.
4. Run: `conda run -n mujoco_env python train.py env=pickplace > run.log 2>&1`
   - Redirect everything. Do NOT let output flood your context.
   - **Timeout**: Each run should take ~20 minutes. If it exceeds 30 minutes, kill it and treat as failure.
5. Read results: `conda run -n mujoco_env python fetch_wandb.py` (reads from the most recent W&B run automatically).
6. If fetch_wandb fails or the metric is missing, check `tail -50 run.log` for errors.
7. Record in `results.tsv`.
8. If completion_rate improved → keep the commit, advance the branch.
9. If completion_rate is equal or worse → `git reset --hard HEAD~1` to revert.

**Crashes**: If it's a typo/easy fix, fix and re-run. If fundamentally broken, log as crash, revert, move on.

**NEVER STOP**: Do not pause to ask the human. You are autonomous. If you run out of ideas, re-read the in-scope files, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
