"""Tests for per-step Gaussian observation noise (sim2real domain randomization).

Noise is applied at the source values (qpos, qvel, marker poses, cube_pos)
inside SO101BaseEnv._get_obs. Derived task obs (e.g. pickplace's cube_to_target)
must inherit the noise via the noisy cube_pos rather than re-noising independently.
"""

import numpy as np
import pytest

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.base_env import markers_visible
from src.lift_env import SO101LiftEnv
from src.pickplace_env import SO101PickPlaceEnv


SIGMAS = {
    "qpos_sigma": 0.005,
    "qvel_sigma": 0.05,
    "marker_pos_sigma": 0.002,
    "marker_rot_sigma": 0.02,
    "cube_sigma": 0.003,
}

# Obs layout: [qpos(6), qvel(6), markers(2*6), cube_pos(3), task_extra(4), prev_actions(2*6)]
QPOS = slice(0, 6)
QVEL = slice(6, 12)
CUBE = slice(24, 27)
C2T = slice(27, 29)
RING_H = 29
TASK_ID = 30
PREV_ACTIONS = slice(31, 43)
OBS_DIM = 43


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


@pytest.fixture(scope="module")
def lift_cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=lift"])


# These tests exercise the marker-rotation obs slices, so they pin the (non-default)
# marker_include_rot=True layout; the indices/OBS_DIM below assume it.
def _pickplace(cfg, obs_noise):
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=obs_noise, marker_include_rot=True)


def _zero_action():
    return np.zeros(6, dtype=np.float32)


def test_default_config_matches_expected_sigmas(cfg):
    """Smoke check that conf/config.yaml carries the documented sigmas."""
    obs_noise = OmegaConf.to_container(cfg.obs_noise, resolve=True)
    assert obs_noise == SIGMAS


def test_no_noise_produces_deterministic_obs(cfg):
    """obs_noise=None → identical obs from identical seeds."""
    env = _pickplace(cfg, obs_noise=None)
    env.reset(seed=0)
    obs_a, *_ = env.step(_zero_action())
    env.reset(seed=0)
    obs_b, *_ = env.step(_zero_action())
    assert np.array_equal(obs_a, obs_b)


def test_noise_perturbs_obs_relative_to_clean(cfg):
    """With noise enabled, obs differs from the clean obs at the same true state."""
    env_clean = _pickplace(cfg, obs_noise=None)
    env_noisy = _pickplace(cfg, obs_noise=SIGMAS)

    env_clean.reset(seed=0)
    env_noisy.reset(seed=0)
    obs_clean, *_ = env_clean.step(_zero_action())
    obs_noisy, *_ = env_noisy.step(_zero_action())

    assert not np.allclose(obs_clean, obs_noisy)


def test_constant_obs_dims_are_not_noised(cfg):
    """ring_height and task_id are passed straight through and must stay clean."""
    env_clean = _pickplace(cfg, obs_noise=None)
    env_noisy = _pickplace(cfg, obs_noise=SIGMAS)

    env_clean.reset(seed=0)
    env_noisy.reset(seed=0)
    obs_clean, *_ = env_clean.step(_zero_action())
    obs_noisy, *_ = env_noisy.step(_zero_action())

    assert obs_clean[RING_H] == obs_noisy[RING_H]
    assert obs_clean[TASK_ID] == obs_noisy[TASK_ID]
    # Prev actions are commanded values, not measurements — must not be noised.
    assert np.array_equal(obs_clean[PREV_ACTIONS], obs_noisy[PREV_ACTIONS])


def test_cube_to_target_uses_noisy_cube_pos(cfg):
    """cube_to_target must be derived from the same noisy cube_pos the agent sees,
    not re-noised or computed from ground truth."""
    env = _pickplace(cfg, obs_noise=SIGMAS)
    env.reset(seed=0)
    obs, *_ = env.step(_zero_action())

    place_target_xy = env.place_target[:2]
    expected_c2t = obs[CUBE][:2] - place_target_xy
    np.testing.assert_allclose(obs[C2T], expected_c2t, atol=1e-7)


def test_per_step_noise_magnitude_matches_sigmas(cfg):
    """Aggregate noise std across many steps should match configured sigmas."""
    env_clean = _pickplace(cfg, obs_noise=None)
    env_noisy = _pickplace(cfg, obs_noise=SIGMAS)

    n_samples = 1000
    diffs = np.empty((n_samples, OBS_DIM), dtype=np.float64)
    visible = np.empty((n_samples, 2), dtype=bool)
    for i in range(n_samples):
        env_clean.reset(seed=i)
        env_noisy.reset(seed=i)
        oc, *_ = env_clean.step(_zero_action())
        on, *_ = env_noisy.step(_zero_action())
        diffs[i] = on - oc
        # Hidden tags are zeroed in both envs (identical dynamics, so identical
        # visibility) — exclude them from the marker noise statistics.
        visible[i] = markers_visible(env_noisy.data, env_noisy.marker_site_ids,
                                     env_noisy.tag_cam_pos)
    assert visible.all(axis=1).mean() > 0.1, "too few both-visible samples"
    marker_pos_diffs = np.concatenate([diffs[visible[:, 0], 12:15].ravel(),
                                       diffs[visible[:, 1], 18:21].ravel()])
    marker_rot_diffs = np.concatenate([diffs[visible[:, 0], 15:18].ravel(),
                                       diffs[visible[:, 1], 21:24].ravel()])

    # Noise std should be within ~10% of configured sigma.
    np.testing.assert_allclose(diffs[:, QPOS].std(axis=0).mean(), SIGMAS["qpos_sigma"], rtol=0.1)
    np.testing.assert_allclose(diffs[:, QVEL].std(axis=0).mean(), SIGMAS["qvel_sigma"], rtol=0.1)
    np.testing.assert_allclose(marker_pos_diffs.std(), SIGMAS["marker_pos_sigma"], rtol=0.1)
    np.testing.assert_allclose(marker_rot_diffs.std(), SIGMAS["marker_rot_sigma"], rtol=0.1)
    np.testing.assert_allclose(diffs[:, CUBE].std(axis=0).mean(), SIGMAS["cube_sigma"], rtol=0.1)
    # cube_to_target inherits cube_sigma (same draw, derived).
    np.testing.assert_allclose(diffs[:, C2T].std(axis=0).mean(), SIGMAS["cube_sigma"], rtol=0.1)


def test_noise_does_not_corrupt_true_state(cfg):
    """The reward path uses true cube/ee state; noise must not leak into self.data."""
    env = _pickplace(cfg, obs_noise=SIGMAS)
    env.reset(seed=0)
    pre_cube = env._get_cube_pos().copy()
    env.step(_zero_action())  # this calls _get_obs which adds noise
    post_cube = env._get_cube_pos()
    # _get_cube_pos reads from self.data — must remain physically meaningful (unchanged
    # apart from physics; with zero action and 1 step, cube hardly moves).
    assert np.linalg.norm(post_cube - pre_cube) < 0.01
    # Reward info uses true grasped/dist — both sane (no NaN, no extreme values).
    obs, reward, term, trunc, info = env.step(_zero_action())
    assert np.isfinite(reward)
    assert isinstance(info["grasped"], (bool, np.bool_))


def test_lift_env_compatible_with_noise(lift_cfg):
    """Lift env must accept obs_noise and produce correctly-shaped obs."""
    env = SO101LiftEnv(env_cfg=lift_cfg.lift_env, xml_path="so101/scene_lift.xml",
                      obs_noise=SIGMAS, marker_include_rot=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    # Lift's _obs_extra returns zeros + task_id, so [27:30] should be zero, [30]=lift TASK_ID=0.0
    assert np.array_equal(obs[27:30], np.zeros(3, dtype=np.float32))
    assert obs[TASK_ID] == 0.0
    # Prev actions are zero immediately after reset (no action has been taken yet).
    assert np.array_equal(obs[PREV_ACTIONS], np.zeros(12, dtype=np.float32))


def test_prev_actions_track_last_two_actions(cfg):
    """After stepping, obs[PREV_ACTIONS] = [a(t-1), a(t)] (12 dims = 2 actions * 6 joints).
    Disable obs_latency so we see the most-recent buffer state without delay."""
    env = SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                            xml_path="so101/scene_pickplace.xml",
                            obs_noise=None, obs_latency=0, marker_include_rot=True)
    env.reset(seed=0)
    a0 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)
    a1 = np.array([-0.7, 0.8, -0.9, 0.4, -0.3, 0.2], dtype=np.float32)
    obs0, *_ = env.step(a0)
    # After first step: slot 0 still zero, slot 1 = a0.
    np.testing.assert_array_equal(obs0[PREV_ACTIONS][:6], np.zeros(6, dtype=np.float32))
    np.testing.assert_allclose(obs0[PREV_ACTIONS][6:], a0)
    obs1, *_ = env.step(a1)
    # After second step: slot 0 = a0, slot 1 = a1.
    np.testing.assert_allclose(obs1[PREV_ACTIONS][:6], a0)
    np.testing.assert_allclose(obs1[PREV_ACTIONS][6:], a1)
