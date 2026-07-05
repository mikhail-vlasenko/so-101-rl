"""Tests for the DR sponge-resize (conf/dr/*.yaml:cube_size_jitter).

Each reset jitters the sponge box's three full side lengths by a uniform
+/- cube_size_jitter (half-extents by +/- jitter/2), independently per axis,
reject-sampling any draw that breaks the strict hx > hy > hz face ordering that
sample_cube_orientation and the tag-on-largest-face convention require. The
cube_tag site must ride onto the resized top face (z offset = new hz), and the
model geom_size must stay in lockstep with self.cube_half_extents.
"""

import numpy as np

from hydra import compose, initialize

from src.lift_env import SO101LiftEnv

NOMINAL = np.array([0.03, 0.02, 0.0125])
JITTER = 0.01  # matches conf/dr/full.yaml
HALF_JITTER = JITTER / 2.0


def _env(cube_size_jitter):
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    return SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                        obs_noise=None, obs_bias=None,
                        cube_size_jitter=cube_size_jitter)


def test_zero_jitter_keeps_nominal():
    """cube_size_jitter=0 (dr=none) must leave the nominal box every reset."""
    env = _env(0.0)
    for seed in range(10):
        env.reset(seed=seed)
        np.testing.assert_array_equal(env.cube_half_extents, NOMINAL)
        np.testing.assert_array_equal(env.model.geom_size[env.cube_geom_id], NOMINAL)


def test_jitter_within_bounds_and_ordered():
    """Every resized box stays within +/- HALF_JITTER of nominal per axis and
    preserves the strict long > mid > short ordering."""
    env = _env(JITTER)
    for seed in range(100):
        env.reset(seed=seed)
        half = env.cube_half_extents
        assert np.all(np.abs(half - NOMINAL) <= HALF_JITTER + 1e-12), (seed, half)
        assert half[0] > half[1] > half[2], (seed, half)


def test_geom_and_tag_track_size():
    """The model geom_size and the cube_tag site z-offset must follow the
    per-episode extents (tag glued to the center of the largest face)."""
    env = _env(JITTER)
    for seed in range(20):
        env.reset(seed=seed)
        half = env.cube_half_extents
        np.testing.assert_array_equal(env.model.geom_size[env.cube_geom_id], half)
        assert env.model.site_pos[env.cube_tag_site_id][2] == half[2]


def test_size_actually_varies():
    """Across resets the box must genuinely change size on every axis, not stay
    pinned to one draw."""
    env = _env(JITTER)
    seen = []
    for seed in range(30):
        env.reset(seed=seed)
        seen.append(env.cube_half_extents.copy())
    seen = np.array(seen)
    assert np.all(seen.std(axis=0) > 0.0), seen.std(axis=0)


def test_pinned_cube_uses_nominal():
    """A pinned cube pose (sysid replay) must use the modeled nominal sponge,
    skipping the resize even when cube_size_jitter > 0."""
    env = _env(JITTER)
    env.reset(seed=0, options={"cube_pos": np.array([0.2, 0.0, 0.03]),
                               "cube_quat": np.array([1.0, 0.0, 0.0, 0.0])})
    np.testing.assert_array_equal(env.cube_half_extents, NOMINAL)
    np.testing.assert_array_equal(env.model.geom_size[env.cube_geom_id], NOMINAL)
    assert env.model.site_pos[env.cube_tag_site_id][2] == NOMINAL[2]
