"""Tests for per-episode observation bias (sim2real domain randomization).

Bias is sampled once at reset() and held constant for the whole episode. It
adds to qpos / marker poses / the cube channels in the observation. The
per-tag marker biases are independent (each tag's own glue/pose-estimate
error), the cube channels carry their own live/precise offsets plus a constant
rotation of the estimated box axes, and a marker_common_sigma shift is shared
by ALL camera-derived positions (the camera re-anchor / table calibration
offset the real pipeline propagates everywhere). No qvel bias — velocity from
differentiating biased qpos has zero DC offset. Reward path always reads
self.data (true state).
"""

import numpy as np
import pytest

from hydra import compose, initialize

from src.base_env import RuntimeEnvConfig
from src.pickplace_env import SO101PickPlaceEnv
from src.shape_obs import (
    camera_depth_axis,
    depth_spread_excess,
    sqrtm_from_upper,
)


SIGMAS = {
    "qpos_sigma": 0.01,
    "marker_pos_sigma": 0.005,     # independent per-tag
    "marker_rot_sigma": 0.05,
    "live_sigma": 0.008,           # independent, live centroid
    "precise_sigma": 0.006,        # independent, precise center
    "sqrtm_depth_sigma": 0.008,    # hull spread along the shared camera axis
    "marker_common_sigma": 0.004,  # shared shift on all camera positions
}

NOISE_SIGMAS = {
    "qpos_sigma": 0.005,
    "marker_rot_sigma": 0.02,
    "tag_px_noise": 0.4,
    "tag_depth_factor": 2.0,
    "live_sigma": 0.003,
    "precise_sigma": 0.003,
}

# Actor-block layout (marker_include_rot=True, prev_actions_n=2): see
# tests/env/test_obs_noise.py.
QPOS = slice(0, 6)
QVEL = slice(6, 12)
MARKER_FINGER_POS = slice(12, 15)
MARKER_FINGER_ROT = slice(15, 18)
MARKER_WRIST_POS = slice(18, 21)
MARKER_WRIST_ROT = slice(21, 24)
LIVE = slice(26, 29)
CENTER = slice(30, 33)
SQRTM = slice(33, 39)
C2T = slice(40, 42)


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _pickplace(cfg, *, obs_bias=SIGMAS, obs_noise=None, marker_always_visible=False):
    runtime = RuntimeEnvConfig(obs_noise=obs_noise, obs_bias=obs_bias,
                               marker_include_rot=True,
                               marker_always_visible=marker_always_visible)
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             cfg=runtime)


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
    assert not np.allclose(first[LIVE], 0)
    assert not np.allclose(first[CENTER], 0)
    for d in diffs[1:]:
        np.testing.assert_allclose(d[QPOS], first[QPOS], atol=1e-6)
        np.testing.assert_allclose(d[MARKER_FINGER_POS], first[MARKER_FINGER_POS], atol=1e-6)
        np.testing.assert_allclose(d[MARKER_WRIST_POS], first[MARKER_WRIST_POS], atol=1e-6)
        np.testing.assert_allclose(d[LIVE], first[LIVE], atol=1e-6)
        np.testing.assert_allclose(d[CENTER], first[CENTER], atol=1e-6)


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
    assert not np.allclose(diff0[LIVE], diff1[LIVE])


def test_marker_bias_common_plus_independent(cfg):
    """Finger and wrist biases share a common-mode shift (camera re-anchor / table
    calib) plus an independent per-tag term. Their per-axis covariance recovers the
    common variance; each tag's total variance is common + independent."""
    env_clean = _pickplace(cfg, obs_bias=None, marker_always_visible=True)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS, marker_always_visible=True)
    n = 1500
    finger = np.empty((n, 3))
    wrist = np.empty((n, 3))
    for i in range(n):
        env_clean.reset(seed=i)
        env_biased.reset(seed=i)
        obs_c, *_ = env_clean.step(_zero_action())
        obs_b, *_ = env_biased.step(_zero_action())
        finger[i] = obs_b[MARKER_FINGER_POS] - obs_c[MARKER_FINGER_POS]
        wrist[i] = obs_b[MARKER_WRIST_POS] - obs_c[MARKER_WRIST_POS]
    common_var = SIGMAS["marker_common_sigma"] ** 2
    total_var = common_var + SIGMAS["marker_pos_sigma"] ** 2
    for axis in range(3):
        cov = np.cov(finger[:, axis], wrist[:, axis])
        # The shared component is the whole cross-covariance between the two tags.
        np.testing.assert_allclose(cov[0, 1], common_var, rtol=0.25)
        # Each tag's marginal variance is common + its own independent term.
        np.testing.assert_allclose(cov[0, 0], total_var, rtol=0.2)
        np.testing.assert_allclose(cov[1, 1], total_var, rtol=0.2)


def test_bias_magnitude_matches_sigmas(cfg):
    """Std of obs offset across resets matches the configured sigmas."""
    env_clean = _pickplace(cfg, obs_bias=None, marker_always_visible=True)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS, marker_always_visible=True)
    n = 500
    qpos_diffs = np.empty((n, 6))
    marker_pos_diffs = np.empty((n, 6))
    marker_rot_diffs = np.empty((n, 6))
    live_diffs = np.empty((n, 3))
    center_diffs = np.empty((n, 3))
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
        live_diffs[i] = obs_b[LIVE] - obs_c[LIVE]
        center_diffs[i] = obs_b[CENTER] - obs_c[CENTER]
    # Each channel's per-axis bias is common + independent, so its std combines both.
    marker_pos_std = np.sqrt(SIGMAS["marker_common_sigma"] ** 2 + SIGMAS["marker_pos_sigma"] ** 2)
    live_std = np.sqrt(SIGMAS["marker_common_sigma"] ** 2 + SIGMAS["live_sigma"] ** 2)
    center_std = np.sqrt(SIGMAS["marker_common_sigma"] ** 2 + SIGMAS["precise_sigma"] ** 2)
    np.testing.assert_allclose(qpos_diffs.std(axis=0).mean(), SIGMAS["qpos_sigma"], rtol=0.15)
    np.testing.assert_allclose(marker_pos_diffs.std(axis=0).mean(), marker_pos_std, rtol=0.15)
    np.testing.assert_allclose(marker_rot_diffs.std(axis=0).mean(),
                               SIGMAS["marker_rot_sigma"], rtol=0.15)
    np.testing.assert_allclose(live_diffs.std(axis=0).mean(), live_std, rtol=0.15)
    np.testing.assert_allclose(center_diffs.std(axis=0).mean(), center_std, rtol=0.15)


def test_hull_depth_bias_inflates_only_along_the_shared_camera_axis(cfg):
    """sqrtm_depth_sigma models the visual hull's error as spread added along
    the axis the two views share — world-fixed, NOT a rotation of the sponge's
    own axes. So the served √M must grow in exactly that direction and nowhere
    else, which is what makes one scalar reproduce the size inflation and the
    pose-dependent principal-axis error together."""
    env_clean = _pickplace(cfg, obs_bias=None, marker_always_visible=True)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS, marker_always_visible=True)
    inflated = 0
    for i in range(10):
        env_clean.reset(seed=i)
        env_biased.reset(seed=i)
        obs_c, *_ = env_clean.step(_zero_action())
        obs_b, *_ = env_biased.step(_zero_action())
        S_c = sqrtm_from_upper(obs_c[SQRTM])
        S_b = sqrtm_from_upper(obs_b[SQRTM])
        axis = camera_depth_axis([cam.pos for cam in env_biased.cube_cams],
                                 env_biased._get_cube_pos())
        M_c, M_b = S_c @ S_c, S_b @ S_b
        # Tolerances are float32-scale: the obs round-trips through float32.
        assert depth_spread_excess(M_b, M_c, axis) == pytest.approx(
            env_biased._sqrtm_depth_spread, abs=1e-6)
        # Nothing added perpendicular to it, in either perpendicular direction.
        for perp in np.linalg.svd(axis.reshape(1, 3))[2][1:]:
            assert depth_spread_excess(M_b, M_c, perp) \
                < 0.01 * env_biased._sqrtm_depth_spread
        inflated += env_biased._sqrtm_depth_spread > 0.0
    assert inflated == 10, "sqrtm_depth_sigma > 0 must actually inflate"


def test_qvel_unbiased(cfg):
    """qvel must NOT carry a bias offset — only per-step noise (when enabled)."""
    env_clean = _pickplace(cfg, obs_bias=None)
    env_biased = _pickplace(cfg, obs_bias=SIGMAS)
    env_clean.reset(seed=0)
    env_biased.reset(seed=0)
    obs_c, *_ = env_clean.step(_zero_action())
    obs_b, *_ = env_biased.step(_zero_action())
    np.testing.assert_array_equal(obs_c[QVEL], obs_b[QVEL])


def test_cube_to_target_uses_biased_live_centroid(cfg):
    """Derived cube_to_target must inherit the live-centroid bias."""
    env = _pickplace(cfg, obs_bias=SIGMAS)
    env.reset(seed=0)
    obs, *_ = env.step(_zero_action())
    expected_c2t = obs[LIVE][:2] - env.place_target[:2]
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
