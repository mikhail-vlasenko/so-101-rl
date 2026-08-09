"""Tripwire against per-step RGB/depth rendering in vectorized BPS envs."""

import time

import numpy as np
from hydra import compose, initialize
from stable_baselines3.common.vec_env import DummyVecEnv

from src.lift_env import SO101LiftEnv
from src.train import runtime_cfg_from_hydra


def test_vectorized_bps_geometry_path_stays_above_rendering_floor():
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift", "dr=none"])
    runtime = runtime_cfg_from_hydra(cfg)
    n_envs = 4
    venv = DummyVecEnv([
        lambda: SO101LiftEnv(
            env_cfg=cfg.lift_env,
            xml_path="so101/scene_lift.xml",
            cfg=runtime,
        )
        for _ in range(n_envs)
    ])
    try:
        venv.reset()
        actions = np.zeros((n_envs, 6), dtype=np.float32)
        n_steps = 20
        start = time.perf_counter()
        for _ in range(n_steps):
            venv.step(actions)
        transitions_per_s = n_envs * n_steps / (time.perf_counter() - start)
        # Geometry sampling/raycasting is comfortably above this. Rendering a
        # stereo RGB/depth pair per env and tick falls below it on the training
        # host, so the deliberately loose floor catches that architectural bug
        # without acting as a microbenchmark.
        assert transitions_per_s > 15.0, transitions_per_s
    finally:
        venv.close()
