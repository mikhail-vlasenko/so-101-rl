# MuJoCo RL Training — SO-101

## Goal

Train RL policies in MuJoCo to make the SO-101 arm perform basic manipulation tasks.

## Model

- Scene: `so101/scene.xml` (includes `so101.xml` + floor/lighting)
- Assets: `so101/assets/` (STL meshes)
- Source: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
- Requires MuJoCo >= 3.1.3 (installed: 3.5.0)

## Reach Task

First training task: move end-effector to a random target position.

- `so101/scene_reach.xml` — scene with mocap target sphere
- `reach_env.py` — Gymnasium env (18-dim obs, 6-dim action, dense reward)
- `train.py` — SAC training with Hydra config + W&B logging
- `conf/config.yaml` — all hyperparameters (env, train, wandb)

### Usage

```bash
python train.py                              # train with defaults
python train.py wandb.enabled=false          # train without W&B
python train.py train.total_timesteps=200000 # override params
python train.py eval=true                    # visualize trained policy
```

### Stack

- **Config:** Hydra (`conf/config.yaml`)
- **Logging:** W&B (entity: `mvlasenko`, project: `robot-arm`)
- **Algorithm:** SAC (Stable-Baselines3)
- **Deps:** gymnasium, stable-baselines3, wandb, hydra-core

## Coding Principles

- **Fail fast, fail loud.** No blanket `try/except`, no `except Exception`, no swallowing errors. If something breaks, let it crash with a clear traceback. Only catch specific exceptions when there's a real recovery path.
- **No magic.** No `getattr`/`setattr` with string keys, no `**kwargs` passthrough when explicit args work, no dynamic dispatch when a simple `if`/`dict` suffices. Code should be readable without running it.
- **Single source of truth.** Don't duplicate constants, config values, or defaults across files. One place defines it, everywhere else reads from there. `conf/config.yaml` owns all hyperparameters.
- **No broken intermediate states.** Don't leave code half-working. If a change touches multiple files, all files must be consistent before moving on. Tests/imports should pass at every step.
- **Explicit over defensive.** Require values instead of falling back to defaults silently (`cfg["key"]` not `cfg.get("key", default)`). If a required value is missing, that's a bug — surface it immediately.
- **No dead code.** Delete unused variables, imports, and functions. Don't comment things out "for later." Version control exists.
- **No redundant comments.** Don't restate what the code already says. Comments explain *why*, not *what*.
