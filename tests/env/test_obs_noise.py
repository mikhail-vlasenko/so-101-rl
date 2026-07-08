"""Tests for per-sample Gaussian observation noise (sim2real domain randomization).

Noise is applied at the source values: qpos per step (SO101BaseEnv._compute_obs),
marker poses and cube_pos per camera frame (_process_frame, frozen at capture).
Derived task obs (e.g. pickplace's cube_to_target) must inherit the noise via the
noisy cube_pos rather than re-noising independently. There is no qvel sigma: the
qvel obs is the backward difference of consecutive noisy qpos over the control
tick (the real pipeline), so its noise is sqrt(2)*qpos_sigma/control_dt.
"""

import numpy as np
import pytest

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.base_env import RuntimeEnvConfig, cube_tag_visible, markers_visible
from src.lift_env import SO101LiftEnv
from src.marker_noise import pos_noise_sigmas
from src.pickplace_env import SO101PickPlaceEnv


# The obs_noise block conf/dr/full.yaml carries — the deployment DR default the
# config-match smoke test pins.
DEFAULT_OBS_NOISE = {
    "qpos_sigma": 0.005,
    "marker_rot_sigma": 0.02,
    "tag_px_noise": 0.2,
    "cube_px_noise": 0.2,
    "tag_depth_factor": 2.0,
}

# Behavioral tests use distinct marker/cube px values so the per-tag oracle in
# test_per_step_noise_magnitude_matches_derived_sigmas catches a key mix-up
# (dr=full happens to set both to 0.2, which would mask it).
SIGMAS = {
    "qpos_sigma": 0.005,
    "marker_rot_sigma": 0.02,
    "tag_px_noise": 0.4,
    "cube_px_noise": 0.15,
    "tag_depth_factor": 2.0,
}

# Obs layout: [qpos(6), qvel(6), markers(2*6), marker_age(2), cube_tag_pos(3),
# cube_tag_rot(3), cube_age(1), task_extra(4), prev_actions(2*6)]
QPOS = slice(0, 6)
QVEL = slice(6, 12)
MARKER_AGE = slice(24, 26)
CUBE = slice(26, 29)
CUBE_ROT = slice(29, 32)
CUBE_AGE = 32
C2T = slice(33, 35)
RING_H = 35
TASK_ID = 36
PREV_ACTIONS = slice(37, 49)
OBS_DIM = 49


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


@pytest.fixture(scope="module")
def lift_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=lift"])


# These tests exercise the marker-rotation obs slices, so they pin the (non-default)
# marker_include_rot=True layout; the indices/OBS_DIM below assume it.
def _pickplace(cfg, obs_noise):
    runtime = RuntimeEnvConfig(obs_noise=obs_noise, marker_include_rot=True)
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             cfg=runtime)


def _zero_action():
    return np.zeros(6, dtype=np.float32)


def test_default_config_matches_expected_sigmas(cfg):
    """Smoke check that conf/config.yaml (dr=full) carries the documented knobs."""
    obs_noise = OmegaConf.to_container(cfg.obs_noise, resolve=True)
    assert obs_noise == DEFAULT_OBS_NOISE


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
    # Marker/cube ages are frame-schedule timing, not measurements — identical
    # too (detection here is geometric-only, so it matches across the two envs).
    assert np.array_equal(obs_clean[MARKER_AGE], obs_noisy[MARKER_AGE])
    assert obs_clean[CUBE_AGE] == obs_noisy[CUBE_AGE]


def test_cube_to_target_uses_noisy_cube_pos(cfg):
    """cube_to_target must be derived from the same noisy cube_pos the agent sees,
    not re-noised or computed from ground truth."""
    env = _pickplace(cfg, obs_noise=SIGMAS)
    env.reset(seed=0)
    obs, *_ = env.step(_zero_action())

    place_target_xy = env.place_target[:2]
    expected_c2t = obs[CUBE][:2] - place_target_xy
    np.testing.assert_allclose(obs[C2T], expected_c2t, atol=1e-7)


def test_per_step_noise_magnitude_matches_derived_sigmas(cfg):
    """Per-tag position noise is anisotropic in the camera frame: projected onto
    each tag's own ray, the depth component matches the depth sigma and the
    lateral component the lateral sigma that src.marker_noise derives from that
    tag's live distance and size. qpos/qvel and the rotation channels stay
    isotropic."""
    env_clean = _pickplace(cfg, obs_noise=None)
    env_noisy = _pickplace(cfg, obs_noise=SIGMAS)

    cam = env_noisy.tag_cam_pos
    focal = env_noisy._focal_px
    kdepth = SIGMAS["tag_depth_factor"]
    # (obs pos slice, obs rot slice, tag size, px knob) per tag, in obs order —
    # the arm markers draw from tag_px_noise, the cube from cube_px_noise.
    tags = [(slice(12, 15), slice(15, 18), env_noisy._marker_tag_sizes[0], SIGMAS["tag_px_noise"]),
            (slice(18, 21), slice(21, 24), env_noisy._marker_tag_sizes[1], SIGMAS["tag_px_noise"]),
            (CUBE, CUBE_ROT, env_noisy._cube_tag_size, SIGMAS["cube_px_noise"])]

    n_samples = 1500
    norm_depth = []      # depth component / depth_sigma  -> N(0,1)
    norm_lat_sq = []     # |lateral|^2 / lateral_sigma^2  -> chi^2 with 2 dof (mean 2)
    rot_diffs = []       # rotation channels stay isotropic
    qpos_diffs = np.empty((n_samples, 6))
    qvel_diffs = np.empty((n_samples, 6))
    for i in range(n_samples):
        env_clean.reset(seed=i)
        env_noisy.reset(seed=i)
        oc, *_ = env_clean.step(_zero_action())
        on, *_ = env_noisy.step(_zero_action())
        qpos_diffs[i] = on[QPOS] - oc[QPOS]
        qvel_diffs[i] = on[QVEL] - oc[QVEL]
        marker_vis = markers_visible(env_noisy.data, env_noisy.marker_site_ids, cam)
        cube_vis = cube_tag_visible(env_noisy.model, env_noisy.data,
                                    env_noisy.cube_tag_site_id, cam, env_noisy.cube_body_id)
        for slot, (pos_sl, rot_sl, size, px) in enumerate(tags):
            if not (cube_vis if slot == 2 else marker_vis[slot]):
                continue  # a hidden tag holds a stale pose in both envs -> zero diff
            true_pos = oc[pos_sl]
            noise = on[pos_sl] - oc[pos_sl]
            lat_s, dep_s, depth_dir = pos_noise_sigmas(true_pos, cam, size, focal, px, kdepth)
            depth_c = noise @ depth_dir
            lat_c = noise - depth_c * depth_dir
            norm_depth.append(depth_c / dep_s)
            norm_lat_sq.append((lat_c @ lat_c) / (lat_s * lat_s))
            rot_diffs.append(on[rot_sl] - oc[rot_sl])

    norm_depth = np.array(norm_depth)
    norm_lat_sq = np.array(norm_lat_sq)
    rot_diffs = np.array(rot_diffs)
    assert len(norm_depth) > 300, "too few tag-visible samples"

    # After dividing by its own derived sigma, the depth component is unit-normal.
    np.testing.assert_allclose(norm_depth.std(), 1.0, rtol=0.1)
    np.testing.assert_allclose(norm_depth.mean(), 0.0, atol=0.1)
    # The lateral part spans the 2D image plane: |lateral|^2/lateral_sigma^2 is
    # chi-squared with 2 dof, mean 2.
    np.testing.assert_allclose(norm_lat_sq.mean(), 2.0, rtol=0.1)
    # Rotation noise is unchanged (isotropic marker_rot_sigma on every axis).
    np.testing.assert_allclose(rot_diffs.std(), SIGMAS["marker_rot_sigma"], rtol=0.1)

    # qpos noise std matches; qvel = (qpos_t - qpos_{t-1})/dt inherits sqrt(2)*qpos_sigma/dt.
    np.testing.assert_allclose(qpos_diffs.std(axis=0).mean(), SIGMAS["qpos_sigma"], rtol=0.1)
    qvel_sigma = np.sqrt(2.0) * SIGMAS["qpos_sigma"] / env_noisy._step_dt
    np.testing.assert_allclose(qvel_diffs.std(axis=0).mean(), qvel_sigma, rtol=0.1)


def test_depth_noise_dominates_lateral(cfg):
    """The whole point of the anisotropy: along the camera ray the position noise
    is many times larger than in the image plane."""
    env_clean = _pickplace(cfg, obs_noise=None)
    env_noisy = _pickplace(cfg, obs_noise=SIGMAS)
    cam = env_noisy.tag_cam_pos
    depth_abs, lat_abs = [], []
    for i in range(400):
        env_clean.reset(seed=i)
        env_noisy.reset(seed=i)
        oc, *_ = env_clean.step(_zero_action())
        on, *_ = env_noisy.step(_zero_action())
        vis = markers_visible(env_noisy.data, env_noisy.marker_site_ids, cam)
        for slot, pos_sl in enumerate((slice(12, 15), slice(18, 21))):
            if not vis[slot]:
                continue
            depth_dir = (oc[pos_sl] - cam) / np.linalg.norm(oc[pos_sl] - cam)
            noise = on[pos_sl] - oc[pos_sl]
            depth_c = noise @ depth_dir
            depth_abs.append(abs(depth_c))
            lat_abs.append(np.linalg.norm(noise - depth_c * depth_dir))
    assert np.mean(depth_abs) > 5.0 * np.mean(lat_abs)


def test_noise_does_not_corrupt_true_state(cfg):
    """The reward path uses true cube/ee state; noise must not leak into self.data."""
    env = _pickplace(cfg, obs_noise=SIGMAS)
    env.reset(seed=0)
    pre_cube = env._get_cube_pos().copy()
    env.step(_zero_action())  # this calls _compute_obs which adds noise
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
                       cfg=RuntimeEnvConfig(obs_noise=SIGMAS, marker_include_rot=True))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    # Lift's _obs_extra returns zeros + task_id, so [33:36] should be zero, [36]=lift TASK_ID=0.0
    assert np.array_equal(obs[33:36], np.zeros(3, dtype=np.float32))
    assert obs[TASK_ID] == 0.0
    # Prev actions are zero immediately after reset (no action has been taken yet).
    assert np.array_equal(obs[PREV_ACTIONS], np.zeros(12, dtype=np.float32))


def test_prev_actions_track_last_two_actions(cfg):
    """After stepping, obs[PREV_ACTIONS] = [a(t-1), a(t)] (12 dims = 2 actions * 6 joints)."""
    env = SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                            xml_path="so101/scene_pickplace.xml",
                            cfg=RuntimeEnvConfig(obs_noise=None, marker_include_rot=True))
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
