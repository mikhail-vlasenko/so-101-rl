# MuJoCo RL Training — SO-101

## Goal

Train RL policies in MuJoCo to make the SO-101 arm perform basic manipulation tasks.

## TODO

Long-term tasks and ideas live in `TODO.md` at the repo root. Check it for known sim-to-real gaps and other backlog items; add to it when surfacing follow-ups that shouldn't be inlined into the current change.

## Model

- Scene: `so101/scene.xml` (includes `so101.xml` + floor/lighting)
- Assets: `so101/assets/` (STL meshes)
- Source: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
- Requires MuJoCo >= 3.1.3 (installed: 3.5.0)

## Tasks

### Shared

- `src/base_env.py` — Base Gymnasium env with MuJoCo setup, contact detection, rendering, reset/step skeleton (20-dim obs, 6-dim action)

### Lift

Grasp a cube and lift it to 10cm. Simpler grasping prerequisite for pick-and-place.

- `so101/scene_lift.xml` — scene with cube only
- `src/lift_env.py` — Gymnasium env (height-progress reward)
- `conf/env/lift.yaml` — env config

### Pick and Place

Pick up a cube and place it at a target location. 3-phase task: REACH → PLACE → RETURN.

- `so101/scene_pickplace.xml` — scene with free-body cube, place target, and ring
- `src/pickplace_env.py` — Gymnasium env (phase-based reward)
- `conf/env/pickplace.yaml` — env config

## Real arm

- `real/twin/digital_twin.py` — MuJoCo passive viewer + tkinter side panel that mirrors the real SO-101 arm's encoders into MuJoCo and lets the user verify the rad↔raw mapping (live direction toggles per joint, raw/norm/rad readouts, optional slider control with torque-on). Built from scratch on raw `scservo_sdk`; does not import from other `real/*.py` files. Run with `python -m real.twin.digital_twin` (default port `/dev/ttyACM0`). Optional DualSense control (`real/twin/gamepad.py`): if a controller is connected, sticks/triggers drive `targets_rad` in CONTROL mode (LX=pan, LY=lift, RY=elbow, R1/R2=wrist_flex, RX=wrist_roll, L1/L2=gripper); D-pad up/down cycles SLOW/MED/FAST speed presets (Tk Up/Down keys do the same); Create toggles MIRROR↔CONTROL, PS is e-stop. Requires `evdev` (in `requirements.txt`); the user must be in the `input` group.

## Training

- `src/train.py` — Training with Hydra config + W&B logging
- `conf/config.yaml` — shared hyperparameters (train, algorithm, wandb), default env: pickplace
- `conf/env/` — per-env config group (selected via `env=lift`, `env=pickplace`, or `env=multitask`)

### Usage

All commands must run in the `mujoco_env` conda environment:

```bash
conda activate mujoco_env

python -m src.train env=lift                     # train lift
python -m src.train env=pickplace                # train pick-and-place (default)
python -m src.train env=multitask                # train on both lift + pickplace (50/50)
python -m src.train env=lift wandb.enabled=false # without W&B
python -m src.train train.total_timesteps=200000 # override params
python -m src.eval env=lift                      # eval lift checkpoint
python -m src.eval env=pickplace                 # eval pickplace checkpoint
python -m src.eval model=best                    # eval best model
python -m src.eval model=path/to/model.zip       # eval specific model
python -m src.show_starts                        # visualize spawn positions
```

When launching training from an agent (no interactive shell), wrap with
`conda run` and redirect to a log file, e.g.:

```bash
conda run -n mujoco_env python -m src.train env=pickplace > run.log 2>&1
```

`env=pickplace` and any other Hydra overrides are optional / swappable. Run it
via the agent's background-execution mechanism, not a shell `&` — `&` plus
`conda run`'s stdout buffering has been flaky in practice.

### Stack

- **Conda env:** `mujoco_env`
- **Config:** Hydra (`conf/config.yaml` + per-env overrides)
- **Logging:** W&B (entity: `mvlasenko`, project: `robot-arm`)
- **Algorithm:** SAC or PPO (Stable-Baselines3)
- **Deps:** gymnasium, stable-baselines3, wandb, hydra-core
- **Real-arm extras (pip-installed into the conda env):** `pip install -r requirements.txt` → `pyserial`. The twin GUI uses stdlib `tkinter`, but requires the Xft-enabled tk build from conda-forge (default conda tk is `noxft` and only sees the bitmap `fixed` font, making the UI unreadably small). Fix once per env: `conda install -n mujoco_env -c conda-forge 'tk=8.6.13=xft_*'`.

## Coding Principles

- **Fail fast, fail loud.** No blanket `try/except`, no `except Exception`, no swallowing errors. If something breaks, let it crash with a clear traceback. Only catch specific exceptions when there's a real recovery path.
- **No magic.** No `getattr`/`setattr` with string keys, no `**kwargs` passthrough when explicit args work, no dynamic dispatch when a simple `if`/`dict` suffices. Code should be readable without running it.
- **Single source of truth.** Don't duplicate constants, config values, or defaults across files. One place defines it, everywhere else reads from there. `conf/config.yaml` owns all hyperparameters.
- **No broken intermediate states.** Don't leave code half-working. If a change touches multiple files, all files must be consistent before moving on. Tests/imports should pass at every step.
- **Explicit over defensive.** Require values instead of falling back to defaults silently (`cfg["key"]` not `cfg.get("key", default)`). If a required value is missing, that's a bug — surface it immediately.
- **No dead code.** Delete unused variables, imports, and functions. Don't comment things out "for later." Version control exists.
- **No redundant comments.** Don't restate what the code already says. Comments explain *why*, not *what*.
- **Verify env behavior with tests, not inline scripts.** When checking that an environment, reward, or observation pipeline behaves as intended, write a pytest under `tests/` (e.g. `tests/test_obs_noise.py`) and run it with `pytest tests/<file>.py -v`. Tests are cheap, reusable, and document the contract; ad-hoc `python -c` snippets disappear after one use.
