---
name: run-progress
description: Show the latest training metrics from a run log in a compact, comparable form. Use when the user asks "how's the run", "check on training", "what are the metrics", or whenever a training process is in flight and a status snapshot is needed.
---

# Run progress

Quick status of a SB3 training run from its stdout log.

## Where to look

Logs live wherever the training command writes them. The default during this project is `/tmp/train_*.log`, but the user may pass a different path. If the user does not name a log, default to the **most recent `/tmp/train_*.log`** (`ls -t /tmp/train_*.log | head -1`).

## What to extract

Each rollout block in the log is a `---...---`-bracketed table containing keys like:

- `rollout/completion_rate`, `rollout/ring_contact_ratio`, `rollout/floor_contact_ratio`
- `rollout/mean_max_cube_height`, `rollout/mean_xy_progress`, `rollout/mean_xy_regress`
- `rollout/ep_rew_mean`, `rollout/episodes`
- `time/fps`, `time/time_elapsed`, `time/total_timesteps`
- `train/explained_variance`, `train/std`, `train/learning_rate`

## How to report

Run this script to get a single compact line for the **latest** rollout block:

```bash
python /home/mikhail/robot_arm/mujoco_training/.claude/skills/run-progress/run_progress.py [LOG_PATH]
```

If `LOG_PATH` is omitted the script picks the newest `/tmp/train_*.log`.

The script prints something like:

```
[ /tmp/train_dr1.log @ 2026-05-09 21:14:33 ]
elapsed=  8.5m  steps= 4.85M  fps= 8730  ep=  227
completion=0.303  ring=0.008  floor=0.015  cube_h=0.033
return=-22.9  xy_prog=0.83  xy_regr=-0.57
ev=0.77  std=0.815  lr=9.4e-04
status: RUNNING  (Time-limit cutoff fires at 20min)
```

Use those numbers in your reply to the user. Do not re-run heavy pipelines (grep loops, etc.) when the script already covers it.

## Status detection

The script also checks for terminal markers in the log:

- `Time limit reached` → status: `DONE (time-limit)`
- `Model saved to` → status: `DONE`
- `Traceback` / `Error` / `Killed` → status: `FAILED` (and prints the last error excerpt)
- otherwise → status: `RUNNING`

## When to call it

- The user asks any "how's training", "what are the numbers", "check the run" question
- You launched a training process and want a sanity check after a wakeup
- You want to compare across runs (call once per log file and lay them side-by-side)

Do not paste full rollout tables into the conversation — the skill exists precisely to keep the report tight.
