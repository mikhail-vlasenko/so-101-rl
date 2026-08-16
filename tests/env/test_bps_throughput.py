"""Tripwire against per-step RGB/depth rendering in vectorized BPS envs."""

import time

import numpy as np
import pytest
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


def test_light_dr_constructs_dense_clouds_at_worker_rate(monkeypatch):
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift", "dr=light"])
    env = SO101LiftEnv(
        env_cfg=cfg.lift_env,
        xml_path="so101/scene_lift.xml",
        cfg=runtime_cfg_from_hydra(cfg),
    )
    original_capture = env._bps_generator.capture
    capture_times = []

    def count_capture(model, data, cameras, cube_geom_id, cube_body_id,
                      half_extents, rng):
        capture_times.append(float(data.time))
        return original_capture(
            model, data, cameras, cube_geom_id, cube_body_id, half_extents, rng)

    monkeypatch.setattr(env._bps_generator, "capture", count_capture)
    try:
        env.reset(seed=123)
        action = np.zeros(6, dtype=np.float32)
        n_steps = 60
        for _ in range(n_steps):
            env.step(action)

        duration_s = n_steps * env._step_dt
        # One boot capture plus the measured 18 Hz worker schedule, allowing
        # one capture of phase slack at either end.
        expected = 1.0 + duration_s / env._camera.bps_frame_s
        assert len(capture_times) == pytest.approx(expected, abs=2.0)
        assert len(capture_times) < duration_s / env._camera.frame_s * 0.7
    finally:
        env.close()
