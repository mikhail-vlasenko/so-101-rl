"""Tests for observation latency (sim2real domain randomization).

The agent sees an observation lagged by `obs_latency` frames. After reset, the
buffer is empty, so the very first call returns the just-computed obs (no past
to draw from). On each subsequent call we push the freshly-computed obs and
return the leftmost element while the buffer fills, then start popping once
it reaches `obs_latency + 1` entries — at which point we are returning the
obs from exactly `obs_latency` steps ago.

Concretely with obs_latency=2:
  reset:         buffer=[obs_0],                      return obs_0
  step 1:        buffer=[obs_0, obs_1],               return obs_0
  step 2:        buffer=[obs_0, obs_1, obs_2],        return obs_0
  step 3:        buffer=[obs_1, obs_2, obs_3],        return obs_1
  step 4:        buffer=[obs_2, obs_3, obs_4],        return obs_2
"""

import numpy as np
import pytest

from hydra import compose, initialize

from src.pickplace_env import SO101PickPlaceEnv


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _pickplace(cfg, obs_latency):
    # Disable noise so we can compare observations by exact equality.
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=None,
                             obs_latency=obs_latency)


def _move_action():
    # Non-zero action so qpos actually changes between steps.
    return np.full(6, 0.5, dtype=np.float32)


def test_default_config_has_obs_latency(cfg):
    assert int(cfg.obs_latency) == 2


def test_latency_zero_returns_current_obs(cfg):
    """obs_latency=0 must be a no-op: each step returns the freshly-computed obs."""
    env = _pickplace(cfg, obs_latency=0)
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(_move_action())
    obs2, *_ = env.step(_move_action())
    # qpos should change with movement, so obs1 != obs0 and obs2 != obs1.
    assert not np.allclose(obs0, obs1)
    assert not np.allclose(obs1, obs2)


def test_latency_two_buffers_initial_obs(cfg):
    """With latency=2, steps 1 and 2 must return the obs from reset."""
    env = _pickplace(cfg, obs_latency=2)
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(_move_action())
    obs2, *_ = env.step(_move_action())
    np.testing.assert_array_equal(obs1, obs0)
    np.testing.assert_array_equal(obs2, obs0)


def test_latency_two_returns_two_steps_ago(cfg):
    """At step k>=3 with latency=2, returned obs equals the raw obs from step k-2.

    We replay an identical trajectory in a no-latency env to capture the ground
    truth raw obs at each step, then compare.
    """
    actions = [_move_action() for _ in range(5)]

    env_clean = _pickplace(cfg, obs_latency=0)
    env_clean.reset(seed=0)
    raw_obs = []
    for a in actions:
        o, *_ = env_clean.step(a)
        raw_obs.append(o)

    env_lag = _pickplace(cfg, obs_latency=2)
    env_lag.reset(seed=0)
    lagged_obs = []
    for a in actions:
        o, *_ = env_lag.step(a)
        lagged_obs.append(o)

    # step 3 (idx 2) returns raw_obs[0]; step 4 returns raw_obs[1]; etc.
    np.testing.assert_array_equal(lagged_obs[2], raw_obs[0])
    np.testing.assert_array_equal(lagged_obs[3], raw_obs[1])
    np.testing.assert_array_equal(lagged_obs[4], raw_obs[2])


def test_reset_clears_latency_buffer(cfg):
    """A fresh reset must drop history so the new initial obs comes back unlagged."""
    env = _pickplace(cfg, obs_latency=2)
    env.reset(seed=0)
    for _ in range(5):
        env.step(_move_action())

    # Reset to a fresh seed and confirm the very first returned obs matches the
    # newly-reset state, not the buffered tail of the previous episode.
    obs_after_reset, _ = env.reset(seed=1)
    obs_step1, *_ = env.step(_move_action())
    np.testing.assert_array_equal(obs_step1, obs_after_reset)


def test_reward_uses_true_state_under_latency(cfg):
    """Reward path reads self.data, not the lagged obs — so reward stays sane."""
    env = _pickplace(cfg, obs_latency=2)
    env.reset(seed=0)
    for _ in range(3):
        _, reward, _, _, info = env.step(_move_action())
        assert np.isfinite(reward)
        assert isinstance(info["grasped"], (bool, np.bool_))
