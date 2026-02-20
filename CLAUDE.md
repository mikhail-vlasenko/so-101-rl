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
