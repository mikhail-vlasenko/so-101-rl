"""Tests for per-step Gaussian observation noise (sim2real domain randomization).

Noise is applied at the source values (qpos, qvel, ee_pos, cube_pos) inside
SO101BaseEnv._get_obs. Derived task obs (e.g. pickplace's cube_to_target) must
inherit the noise via the noisy cube_pos rather than re-noising independently.
"""

import numpy as np
import pytest

from hydra import compose, initialize
from omegaconf import OmegaConf

from lift_env import SO101LiftEnv
from pickplace_env import SO101PickPlaceEnv


SIGMAS = {
    "qpos_sigma": 0.005,
    "qvel_sigma": 0.05,
    "ee_sigma": 0.002,
    "cube_sigma": 0.003,
}

# Obs layout: [qpos(6), qvel(6), ee_pos(3), cube_pos(3), task_extra(4)]
QPOS = slice(0, 6)
QVEL = slice(6, 12)
EE = slice(12, 15)
CUBE = slice(15, 18)
C2T = slice(18, 20)
RING_H = 20
TASK_ID = 21


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


@pytest.fixture(scope="module")
def lift_cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=lift"])


def _pickplace(cfg, obs_noise):
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=obs_noise)


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
    diffs = np.empty((n_samples, 22), dtype=np.float64)
    for i in range(n_samples):
        env_clean.reset(seed=i)
        env_noisy.reset(seed=i)
        oc, *_ = env_clean.step(_zero_action())
        on, *_ = env_noisy.step(_zero_action())
        diffs[i] = on - oc

    # Noise std should be within ~10% of configured sigma.
    np.testing.assert_allclose(diffs[:, QPOS].std(axis=0).mean(), SIGMAS["qpos_sigma"], rtol=0.1)
    np.testing.assert_allclose(diffs[:, QVEL].std(axis=0).mean(), SIGMAS["qvel_sigma"], rtol=0.1)
    np.testing.assert_allclose(diffs[:, EE].std(axis=0).mean(), SIGMAS["ee_sigma"], rtol=0.1)
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
                      obs_noise=SIGMAS)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (22,)
    # Lift's _obs_extra returns zeros + task_id, so [18:21] should be zero, [21]=lift TASK_ID=0.0
    assert np.array_equal(obs[18:21], np.zeros(3, dtype=np.float32))
    assert obs[TASK_ID] == 0.0
