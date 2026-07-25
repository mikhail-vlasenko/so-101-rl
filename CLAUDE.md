# MuJoCo RL Training — SO-101

Train RL policies in MuJoCo for the SO-101 arm and run them on the real arm.
Scene: `so101/scene.xml`, meshes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101).
Long-term tasks and known sim-to-real gaps live in `TODO.md` — check it, and add
follow-ups there instead of inlining them.

Module docstrings and `conf/*.yaml` comments are the primary documentation. This
file holds only cross-file contracts, gotchas, and commands — keep it that way.

## Map

- `src/` — envs (`base_env.py` two-layer base; reach/lift/pickplace on top),
  training (`train.py`, `eval.py`, `distill.py`), obs pipeline (`obs_history.py`,
  `obs_norm.py`, `camera_sim.py`, `marker_noise.py`), control chain (`units.py`,
  `servo_profile.py`)
- `conf/` — Hydra config. `config.yaml` owns all hyperparameters; groups: `env`
  (reach/lift/pickplace/multitask), `dr` + `shaping` (curriculum stages)
- `real/` — `vision/` (camera + marker detection/pose primitives), `calib/`
  (calibration stack), `tracking/` (stereo/SAM tracking, the shape dataset +
  estimator eval + visual hull), `rollout/` (rollout scripts,
  `rollout_common.py` core, `frame_bus.py` camera fan-out, `object_obs.py` SAM
  object channels), `diagnostics/` (one-off reads), `twin/` (digital twin,
  `python -m real.twin.digital_twin`); `marker_spec.py` stays at the top — the
  tag/marker source of truth every subpackage reads
- `sysid/` — real-vs-sim dynamics fitting (record → replay → analyze → fit) and
  probes; the fit is baked into `so101.xml`
- `panel/` — web control panel (`python -m panel`, port 8800)
- `tests/{env,training,control,sysid,panel,real}` — pytest
- `scripts/` — diagnostics + curriculum wrapper

## Contracts and gotchas

- **Observation**: `[actor block per history tap | privileged tail]`; the tail is
  critic-only and zero-padded on the real arm (layout doc:
  `src/base_env.obs_dim_for` / `priv_dim_for`). The policy sees AprilTag poses for
  the arm markers and the tag-free dual-channel cube obs (live triangulated
  centroid + static-refreshed center/√M, `src/shape_obs.py`; real source = SAM
  stereo, `real/rollout/object_obs.py`), never GT state; a hidden tag or channel
  holds its last value while its age channel grows. The sponge's tag id 1 is
  EVAL-ONLY (dataset GT / legacy distillation), never on the obs path.
- **Checkpoint compatibility**: anything that changes obs dim (`history_taps`,
  `prev_actions_n`, `marker_include_rot`) or the `src/obs_norm.py` constants
  invalidates checkpoints. Migrate with `src/distill.py`, not checkpoint surgery,
  and always follow distillation with a short PPO fine-tune (`resume=`).
- **Camera poses**: the real pipeline never stores one — every camera re-anchors
  per frame from the table tag, coasting on the last EMA'd anchor while the tag
  is occluded (`real/rollout/{marker_obs,object_obs}.py`). The sim mounts
  (`so101.xml` `tag_cam_mount` / `tag_cam_aux_mount`) are snapshots of that
  anchoring, so **re-run `real.diagnostics.snapshot_cam_mount --camera <main|aux>`
  after any remount** — otherwise sim visibility, spawn rejection and the fk twin
  model a camera that isn't there.
- **Sim/real twins**: some behavior is implemented twice and must stay identical —
  marker hold-last-pose/age (`src/base_env.py` ↔ `real/rollout/marker_obs.py`),
  `ObsHistory` feeding, `action_to_target`. Change one side, change the other;
  contract tests in `tests/` pin them. The cube channels are single-sourced
  instead: both sides drive `src/shape_obs.ObjectChannelDriver` — never fork it
  (pinned by `tests/real/test_object_twin.py`).
- `src/units.py` is the single source of the action → rad → servo-raw chain; never
  hand-derive the raw clamp.
- `use_servo_profile` stays **on**: the `so101.xml` sysid fit was refit with the
  profile, so the sim is under-damped without it. Keep any new sim driver on this
  flag.
- New real rollout scripts must build on `ArmLoop` (`real/rollout/rollout_common.py`) —
  never reimplement the per-tick command shaping.
- **Curriculum** = the `dr`/`shaping` config groups + the crutch flags
  (`marker_always_visible`, `cube_smallest_face_only`, `cube_no_flat_spawns`).
  None change obs dim, so
  they can flip across `resume` (which re-applies the composed config's
  hyperparameters — see `conf/config.yaml`). Shaping goes `none → light → full`;
  skipping `light` makes the policy stop touching the cube.
- The manipulated object is a 6×4×2.5 cm sponge, still named `cube_*` everywhere.
- The panel has **no auth** — `--host 0.0.0.0` only on trusted networks. Keep
  `panel/registry.py` updated when adding scripts or args. `--reload` SIGINTs every
  live run on each edit — don't use it mid-rollout.

## Usage

All commands run in the `mujoco_env` conda env:

```bash
python -m src.train env=lift                     # or pickplace (default), reach, multitask
python -m src.train env=lift wandb.enabled=false train.total_timesteps=200000
python -m src.train resume=path/to/ckpt.zip
python -m src.eval model=best                    # or latest / path/to/model.zip
python -m src.show_starts                        # visualize spawn positions
python -m src.distill env=lift distill.teacher=old.zip   # regimes: distill block in config.yaml
python -m panel
```

When launching training from an agent (no interactive shell), use
`conda run -n mujoco_env python -m src.train ... > run.log 2>&1` via the agent's
background-execution mechanism, not shell `&` — `&` plus `conda run`'s stdout
buffering has been flaky.

## Stack

Hydra + Stable-Baselines3 (PPO/SAC) + W&B (`mvlasenko/robot-arm`), conda env
`mujoco_env`. Real-arm extras: `pip install -r requirements.txt`. The twin's
tkinter needs the conda-forge Xft tk build (default `noxft` tk renders unreadably
small): `conda install -n mujoco_env -c conda-forge 'tk=8.6.13=xft_*'`.

## Coding Principles

- **Fail fast, fail loud.** No blanket `try/except`, no `except Exception`, no
  swallowing errors. If something breaks, let it crash with a clear traceback. Only
  catch specific exceptions when there's a real recovery path.
- **No magic.** No `getattr`/`setattr` with string keys, no `**kwargs` passthrough
  when explicit args work, no dynamic dispatch when a simple `if`/`dict` suffices.
  Code should be readable without running it.
- **Single source of truth.** Don't duplicate constants, config values, or defaults
  across files. One place defines it, everywhere else reads from there.
  `conf/config.yaml` owns all hyperparameters.
- **No broken intermediate states.** Don't leave code half-working. If a change
  touches multiple files, all files must be consistent before moving on.
  Tests/imports should pass at every step.
- **Explicit over defensive.** Require values instead of falling back to defaults
  silently (`cfg["key"]` not `cfg.get("key", default)`). If a required value is
  missing, that's a bug — surface it immediately.
- **No dead code.** Delete unused variables, imports, and functions. Don't comment
  things out "for later." Version control exists.
- **No redundant comments.** Don't restate what the code already says. Comments
  explain *why*, not *what*.
- **Verify env behavior with tests, not inline scripts.** When checking that an
  environment, reward, or observation pipeline behaves as intended, write a pytest
  under `tests/` (e.g. `tests/env/test_obs_noise.py`) and run it with
  `pytest tests/<file>.py -v`. Tests are cheap, reusable, and document the contract;
  ad-hoc `python -c` snippets disappear after one use.
