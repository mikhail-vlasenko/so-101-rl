"""Tests for per-episode observation bias (sim2real domain randomization).

Bias is sampled once at reset() and held constant for the whole episode. It
adds to qpos / marker poses / cube_pos in the observation. Marker biases are
drawn independently per marker (uncorrelated glue/extrinsics error per tag).
No qvel bias — velocity from differentiating biased qpos has zero DC offset.
Reward path always reads self.data (true state).
"""

import numpy as np
import pytest

from hydra import compose, initialize

from src.pickplace_env import SO101PickPlaceEnv


SIGMAS = {
    "qpos_sigma": 0.01,
    "marker_pos_sigma": 0.005,
    "marker_rot_sigma": 0.05,
    "cube_sigma": 0.008,
}

NOISE_SIGMAS = {
    "qpos_sigma": 0.005,
    "qvel_sigma": 0.05,
    "marker_pos_sigma": 0.002,
    "marker_rot_sigma": 0.02,
    "cube_sigma": 0.003,
}

QPOS = slice(0, 6)
QVEL = slice(6, 12)
MARKER_FINGER_POS = slice(12, 15)
MARKER_FINGER_ROT = slice(15, 18)
MARKER_WRIST_POS = slice(18, 21)
MARKER_WRIST_ROT = slice(21, 24)
CUBE = slice(24, 27)
C2T = slice(27, 29)


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _pickplace(cfg, *, obs_bias=SIGMAS, obs_noise=None, marker_always_visible=False):
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=obs_noise,
                             obs_bias=obs_bias,
                             marker_include_rot=True,
                             marker_always_visible=marker_always_visible)


def _zero_action():
    return np.zeros(6, dtype=np.float32)


def test_bias_constant_within_episode(cfg):
    """Obs offset (biased - clean) stays constant across steps; non-zero."""
    env_clean = _pickplace(cfg, obs_bias=None)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS)
    env_clean.reset(seed=4)
    env_biased.reset(seed=4)
    diffs = []
    for _ in range(5):
        obs_c, *_ = env_clean.step(_zero_action())
        obs_b, *_ = env_biased.step(_zero_action())
        diffs.append(obs_b - obs_c)
    first = diffs[0]
    assert not np.allclose(first[QPOS], 0)
    assert not np.allclose(first[CUBE], 0)
    for d in diffs[1:]:
        np.testing.assert_allclose(d[QPOS], first[QPOS], atol=1e-6)
        np.testing.assert_allclose(d[MARKER_FINGER_POS], first[MARKER_FINGER_POS], atol=1e-6)
        np.testing.assert_allclose(d[MARKER_WRIST_POS], first[MARKER_WRIST_POS], atol=1e-6)
        np.testing.assert_allclose(d[CUBE], first[CUBE], atol=1e-6)


def test_bias_changes_across_episodes(cfg):
    """Different seeds produce different bias offsets in the observation."""
    env_clean = _pickplace(cfg, obs_bias=None, marker_always_visible=True)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS, marker_always_visible=True)

    env_clean.reset(seed=0)
    env_biased.reset(seed=0)
    obs_c0, *_ = env_clean.step(_zero_action())
    obs_b0, *_ = env_biased.step(_zero_action())
    diff0 = obs_b0 - obs_c0

    env_clean.reset(seed=1)
    env_biased.reset(seed=1)
    obs_c1, *_ = env_clean.step(_zero_action())
    obs_b1, *_ = env_biased.step(_zero_action())
    diff1 = obs_b1 - obs_c1

    assert not np.allclose(diff0[QPOS], diff1[QPOS])
    assert not np.allclose(diff0[MARKER_FINGER_POS], diff1[MARKER_FINGER_POS])
    assert not np.allclose(diff0[CUBE], diff1[CUBE])


def test_marker_biases_uncorrelated(cfg):
    """Finger and wrist marker biases are drawn independently (per-tag glue error)."""
    env_clean = _pickplace(cfg, obs_bias=None, marker_always_visible=True)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS, marker_always_visible=True)
    n = 500
    finger = np.empty((n, 3))
    wrist = np.empty((n, 3))
    for i in range(n):
        env_clean.reset(seed=i)
        env_biased.reset(seed=i)
        obs_c, *_ = env_clean.step(_zero_action())
        obs_b, *_ = env_biased.step(_zero_action())
        finger[i] = obs_b[MARKER_FINGER_POS] - obs_c[MARKER_FINGER_POS]
        wrist[i] = obs_b[MARKER_WRIST_POS] - obs_c[MARKER_WRIST_POS]
    for axis in range(3):
        corr = np.corrcoef(finger[:, axis], wrist[:, axis])[0, 1]
        assert abs(corr) < 0.15


def test_bias_magnitude_matches_sigmas(cfg):
    """Std of obs offset across resets matches the configured sigmas."""
    env_clean = _pickplace(cfg, obs_bias=None, marker_always_visible=True)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS, marker_always_visible=True)
    n = 500
    qpos_diffs = np.empty((n, 6))
    marker_pos_diffs = np.empty((n, 6))
    marker_rot_diffs = np.empty((n, 6))
    cube_diffs = np.empty((n, 3))
    for i in range(n):
        env_clean.reset(seed=i)
        env_biased.reset(seed=i)
        obs_c, *_ = env_clean.step(_zero_action())
        obs_b, *_ = env_biased.step(_zero_action())
        qpos_diffs[i] = obs_b[QPOS] - obs_c[QPOS]
        marker_pos_diffs[i, :3] = obs_b[MARKER_FINGER_POS] - obs_c[MARKER_FINGER_POS]
        marker_pos_diffs[i, 3:] = obs_b[MARKER_WRIST_POS] - obs_c[MARKER_WRIST_POS]
        marker_rot_diffs[i, :3] = obs_b[MARKER_FINGER_ROT] - obs_c[MARKER_FINGER_ROT]
        marker_rot_diffs[i, 3:] = obs_b[MARKER_WRIST_ROT] - obs_c[MARKER_WRIST_ROT]
        cube_diffs[i] = obs_b[CUBE] - obs_c[CUBE]
    np.testing.assert_allclose(qpos_diffs.std(axis=0).mean(), SIGMAS["qpos_sigma"], rtol=0.15)
    np.testing.assert_allclose(marker_pos_diffs.std(axis=0).mean(),
                               SIGMAS["marker_pos_sigma"], rtol=0.15)
    np.testing.assert_allclose(marker_rot_diffs.std(axis=0).mean(),
                               SIGMAS["marker_rot_sigma"], rtol=0.15)
    np.testing.assert_allclose(cube_diffs.std(axis=0).mean(), SIGMAS["cube_sigma"], rtol=0.15)


def test_qvel_unbiased(cfg):
    """qvel must NOT carry a bias offset — only per-step noise (when enabled)."""
    env_clean = _pickplace(cfg, obs_bias=None)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS)
    env_clean.reset(seed=0)
    env_biased.reset(seed=0)
    obs_c, *_ = env_clean.step(_zero_action())
    obs_b, *_ = env_biased.step(_zero_action())
    np.testing.assert_array_equal(obs_c[QVEL], obs_b[QVEL])


def test_cube_to_target_uses_biased_cube_pos(cfg):
    """Derived cube_to_target must inherit the cube_pos bias."""
    env = _pickplace(cfg, obs_bias=SIGMAS)
    env.reset(seed=0)
    obs, *_ = env.step(_zero_action())
    expected_c2t = obs[CUBE][:2] - env.place_target[:2]
    np.testing.assert_allclose(obs[C2T], expected_c2t, atol=1e-7)


def test_bias_does_not_corrupt_physics(cfg):
    """Physics state and reward path must be unaffected by obs bias."""
    env = _pickplace(cfg, obs_bias=SIGMAS)
    env.reset(seed=0)
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(_zero_action())
        assert np.isfinite(reward)
        assert isinstance(info["grasped"], (bool, np.bool_))


def test_bias_plus_noise_compose(cfg):
    """With both bias and noise, obs deviates further than either alone."""
    env_bias_only = _pickplace(cfg, obs_bias=SIGMAS)
    env_both = _pickplace(cfg, obs_bias=SIGMAS, obs_noise=NOISE_SIGMAS)
    env_bias_only.reset(seed=0)
    env_both.reset(seed=0)
    obs_bias_only, *_ = env_bias_only.step(_zero_action())
    obs_both, *_ = env_both.step(_zero_action())
    diff = obs_both[QPOS] - obs_bias_only[QPOS]
    assert not np.allclose(diff, 0.0)


def test_disabled_bias_produces_clean_obs(cfg):
    """obs_bias=None must produce identical obs to a fully clean env."""
    env_a = _pickplace(cfg, obs_bias=None)
    env_b = _pickplace(cfg, obs_bias=None)
    env_a.reset(seed=0)
    env_b.reset(seed=0)
    obs_a, *_ = env_a.step(_zero_action())
    obs_b, *_ = env_b.step(_zero_action())
    np.testing.assert_array_equal(obs_a, obs_b)
