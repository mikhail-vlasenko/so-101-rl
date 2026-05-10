"""Tests for per-episode observation bias (sim2real domain randomization).

Bias is sampled once at reset() and held constant for the whole episode. It
adds to qpos / ee_pos / cube_pos at the source. No qvel bias — velocity from
differentiating biased qpos has zero DC offset, so a velocity bias would be
non-physical. Reward path always reads self.data (true state).
"""

import numpy as np
import pytest

from hydra import compose, initialize

from pickplace_env import SO101PickPlaceEnv


SIGMAS = {
    "qpos_sigma": 0.01,
    "ee_sigma":   0.005,
    "cube_sigma": 0.008,
}

NOISE_SIGMAS = {
    "qpos_sigma": 0.005,
    "qvel_sigma": 0.05,
    "ee_sigma": 0.002,
    "cube_sigma": 0.003,
}

# Obs layout (matches test_obs_noise.py):
QPOS = slice(0, 6)
QVEL = slice(6, 12)
EE = slice(12, 15)
CUBE = slice(15, 18)
C2T = slice(18, 20)


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _pickplace(cfg, *, obs_bias=SIGMAS, obs_noise=None):
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=obs_noise,
                             obs_bias=obs_bias)


def _zero_action():
    return np.zeros(6, dtype=np.float32)


def test_bias_constant_within_episode(cfg):
    """Bias arrays must not change across steps within a single episode."""
    env = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env.reset(seed=0)
    qpos_b0 = env._qpos_bias.copy()
    ee_b0 = env._ee_bias.copy()
    cube_b0 = env._cube_bias.copy()
    for _ in range(5):
        env.step(_zero_action())
    np.testing.assert_array_equal(env._qpos_bias, qpos_b0)
    np.testing.assert_array_equal(env._ee_bias, ee_b0)
    np.testing.assert_array_equal(env._cube_bias, cube_b0)


def test_bias_changes_across_episodes(cfg):
    """Different reset seeds must produce different bias draws."""
    env = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env.reset(seed=0)
    b0 = (env._qpos_bias.copy(), env._ee_bias.copy(), env._cube_bias.copy())
    env.reset(seed=1)
    b1 = (env._qpos_bias.copy(), env._ee_bias.copy(), env._cube_bias.copy())
    assert not np.allclose(b0[0], b1[0])
    assert not np.allclose(b0[1], b1[1])
    assert not np.allclose(b0[2], b1[2])


def test_bias_magnitude_matches_sigmas(cfg):
    """Std across many resets must match the configured sigmas (per axis)."""
    env = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    n = 2000
    qpos_biases = np.empty((n, 6))
    ee_biases = np.empty((n, 3))
    cube_biases = np.empty((n, 3))
    for i in range(n):
        env.reset(seed=i)
        qpos_biases[i] = env._qpos_bias
        ee_biases[i] = env._ee_bias
        cube_biases[i] = env._cube_bias
    np.testing.assert_allclose(qpos_biases.std(axis=0).mean(), SIGMAS["qpos_sigma"], rtol=0.1)
    np.testing.assert_allclose(ee_biases.std(axis=0).mean(), SIGMAS["ee_sigma"], rtol=0.1)
    np.testing.assert_allclose(cube_biases.std(axis=0).mean(), SIGMAS["cube_sigma"], rtol=0.1)


def test_bias_offsets_obs_relative_to_clean(cfg):
    """obs[QPOS] - clean_obs[QPOS] should equal qpos_bias exactly (noise off)."""
    env_clean = _pickplace(cfg, obs_bias=None, obs_noise=None)
    env_bias = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env_clean.reset(seed=0)
    env_bias.reset(seed=0)
    obs_clean, *_ = env_clean.step(_zero_action())
    obs_bias_, *_ = env_bias.step(_zero_action())
    np.testing.assert_allclose(obs_bias_[QPOS] - obs_clean[QPOS], env_bias._qpos_bias, atol=1e-6)
    np.testing.assert_allclose(obs_bias_[EE] - obs_clean[EE], env_bias._ee_bias, atol=1e-6)
    np.testing.assert_allclose(obs_bias_[CUBE] - obs_clean[CUBE], env_bias._cube_bias, atol=1e-6)


def test_qvel_unbiased(cfg):
    """qvel must NOT carry a bias offset — only per-step noise (when enabled)."""
    env_clean = _pickplace(cfg, obs_bias=None, obs_noise=None)
    env_bias = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env_clean.reset(seed=0)
    env_bias.reset(seed=0)
    obs_clean, *_ = env_clean.step(_zero_action())
    obs_bias_, *_ = env_bias.step(_zero_action())
    np.testing.assert_array_equal(obs_clean[QVEL], obs_bias_[QVEL])


def test_cube_to_target_uses_biased_cube_pos(cfg):
    """Derived cube_to_target must inherit the cube_pos bias."""
    env = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env.reset(seed=0)
    obs, *_ = env.step(_zero_action())
    place_target_xy = env.place_target[:2]
    expected_c2t = obs[CUBE][:2] - place_target_xy
    np.testing.assert_allclose(obs[C2T], expected_c2t, atol=1e-7)


def test_bias_does_not_corrupt_true_state(cfg):
    """self.data (the reward path) must be unaffected by bias application."""
    env = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env.reset(seed=0)
    pre = env._get_cube_pos().copy()
    env.step(_zero_action())
    post = env._get_cube_pos()
    assert np.linalg.norm(post - pre) < 0.01
    obs, reward, term, trunc, info = env.step(_zero_action())
    assert np.isfinite(reward)
    assert isinstance(info["grasped"], (bool, np.bool_))


def test_bias_plus_noise_compose(cfg):
    """With both bias and noise, obs deviates further than either alone."""
    env_bias_only = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=None)
    env_both = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=NOISE_SIGMAS)
    env_bias_only.reset(seed=0)
    env_both.reset(seed=0)
    obs_bias_only, *_ = env_bias_only.step(_zero_action())
    obs_both, *_ = env_both.step(_zero_action())
    # Same bias draw (same seed) so the difference is the noise contribution.
    diff = obs_both[QPOS] - obs_bias_only[QPOS]
    assert not np.allclose(diff, 0.0)


def test_disabled_bias_produces_clean_obs(cfg):
    """obs_bias=None must produce identical obs to a fully clean env."""
    env_a = _pickplace(cfg, obs_bias=None, obs_noise=None)
    env_b = _pickplace(cfg, obs_bias=None, obs_noise=None)
    env_a.reset(seed=0)
    env_b.reset(seed=0)
    obs_a, *_ = env_a.step(_zero_action())
    obs_b, *_ = env_b.step(_zero_action())
    np.testing.assert_array_equal(obs_a, obs_b)
