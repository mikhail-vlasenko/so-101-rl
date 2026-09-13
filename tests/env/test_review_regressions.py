"""Executable counterexamples from the September 2026 project review.

Strict expected failures describe desired contracts, not observed policy
exploits. Remove each xfail when its corresponding TODO is implemented.
"""

import numpy as np
import pytest
from hydra import compose, initialize

from src.base_env import RuntimeEnvConfig
from src.lift_env import SO101LiftEnv
from src.pickplace_env import SO101PickPlaceEnv


@pytest.fixture
def lift():
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "dr=none", "shaping=none"])
    env = SO101LiftEnv(env_cfg=cfg.lift_env, cfg=RuntimeEnvConfig())
    env.reset(seed=7)
    yield env
    env.close()


@pytest.mark.xfail(strict=True, reason="Lift terminates on the first grasped height crossing")
def test_lift_requires_hold_after_first_height_crossing(lift):
    cube = np.array([0.2, 0.0, lift.target_height + 0.001])
    lift._prev_cube_pos = cube - np.array([0.0, 0.0, 0.002])
    _, terminated, _ = lift._compute_step(cube, cube, 0.0, True, False)
    assert not terminated


@pytest.mark.xfail(strict=True, reason="Releasing on descent avoids negative height progress")
def test_lift_drop_and_regrasp_cycle_cannot_farm_height_progress(lift, monkeypatch):
    monkeypatch.setattr(lift, "_n_jaw_contacts", lambda: 0)
    low = np.array([0.2, 0.0, 0.03])
    high = np.array([0.2, 0.0, 0.08])
    lift._prev_cube_pos = low.copy()
    up, _, _ = lift._compute_step(high, high, 0.0, True, False)
    down, _, _ = lift._compute_step(low, low, 0.0, False, False)
    lift._prev_cube_pos = low.copy()
    hold, _, _ = lift._compute_step(low, low, 0.0, True, False)
    release, _, _ = lift._compute_step(low, low, 0.0, False, False)
    assert up + down <= hold + release


@pytest.mark.xfail(strict=True, reason="Placement ignores release and settling")
def test_placement_requires_releasing_the_object():
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=pickplace"])
    env = SO101PickPlaceEnv(env_cfg=cfg.pickplace_env, cfg=RuntimeEnvConfig())
    try:
        env.reset(seed=7)
        cube = env.place_target.copy()
        cube[2] = env.place_target_height - 0.001
        _, terminated, _ = env._compute_step(cube, cube, 0.0, True, False)
        assert not terminated
    finally:
        env.close()
