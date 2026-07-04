"""Tests for the ring-height gate on the XY progress reward.

xy_reward only fires when grasped AND cube_z > ring_height. With ring at 0,
even a resting cube clears the gate (1.5cm > 0); with ring at full height
(5cm) the cube must be lifted above 5cm for XY progress to count.
"""

import numpy as np
import pytest
from hydra import compose, initialize

from src.pickplace_env import SO101PickPlaceEnv, XY_PROGRESS_COEFF, GRASP_HOLD_REWARD


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _env(cfg):
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=None, obs_bias=None)


def _call_reward(env, cube_z, grasped, ring_height, prev_xy_dist=0.10, new_xy_dist=0.08):
    """Drive _compute_reward with a synthetic cube state. Returns the reward."""
    env.ring_height = ring_height
    env._prev_xy_dist = prev_xy_dist
    env._xy_progress_total = 0.0
    env._xy_regress_total = 0.0
    cube_pos = np.array([env.place_target[0] + new_xy_dist, env.place_target[1], cube_z])
    ee_pos = cube_pos + np.array([0.0, 0.0, 0.02])
    return env._compute_reward(
        ee_pos=ee_pos,
        cube_pos=cube_pos,
        ee_cube_dist=0.02,
        grasped=grasped,
        floor_contact=False,
        ring_contact=False,
    )


def test_gate_blocks_when_cube_below_ring(cfg):
    """Cube at floor with ring at 5cm — XY reward is gated off."""
    env = _env(cfg)
    env.reset(seed=0)
    r = _call_reward(env, cube_z=0.015, grasped=True, ring_height=0.05)
    # Only TIME_PENALTY + GRASP_HOLD_REWARD should remain — no XY contribution.
    expected = -0.05 + GRASP_HOLD_REWARD
    assert env._xy_progress_total == 0.0
    assert env._xy_regress_total == 0.0
    assert r == pytest.approx(expected, abs=1e-9)


def test_gate_open_when_cube_above_ring(cfg):
    """Cube at 8cm with ring at 5cm — XY reward fires."""
    env = _env(cfg)
    env.reset(seed=0)
    delta = 0.10 - 0.08  # closer by 2cm
    r = _call_reward(env, cube_z=0.08, grasped=True, ring_height=0.05)
    expected_xy = XY_PROGRESS_COEFF * delta
    assert env._xy_progress_total == pytest.approx(expected_xy)
    assert r == pytest.approx(-0.05 + GRASP_HOLD_REWARD + expected_xy, abs=1e-9)


def test_low_ring_lets_resting_cube_through(cfg):
    """Ring sunk to 0 — even cube_z=1.5cm clears the gate."""
    env = _env(cfg)
    env.reset(seed=0)
    delta = 0.10 - 0.08
    r = _call_reward(env, cube_z=0.015, grasped=True, ring_height=0.0)
    expected_xy = XY_PROGRESS_COEFF * delta
    assert env._xy_progress_total == pytest.approx(expected_xy)
    assert r == pytest.approx(-0.05 + GRASP_HOLD_REWARD + expected_xy, abs=1e-9)


def test_not_grasped_still_uses_approach_reward(cfg):
    """When not grasped, gate is irrelevant — fallback approach reward fires."""
    env = _env(cfg)
    env.reset(seed=0)
    r = _call_reward(env, cube_z=0.20, grasped=False, ring_height=0.05)
    # No XY accounting either way.
    assert env._xy_progress_total == 0.0
    assert env._xy_regress_total == 0.0
    # Reward = TIME_PENALTY + EE_CUBE_COEFF * max(ee_cube_dist, EE_CUBE_MIN_DIST)
    # = -0.05 + (-0.5) * max(0.02, 0.025) = -0.05 - 0.0125
    assert r == pytest.approx(-0.05 + (-0.5) * 0.025, abs=1e-9)


def test_xy_regress_also_gated(cfg):
    """When cube moves away while below ring, no negative XY reward either."""
    env = _env(cfg)
    env.reset(seed=0)
    r = _call_reward(env, cube_z=0.015, grasped=True, ring_height=0.05,
                     prev_xy_dist=0.08, new_xy_dist=0.10)
    assert env._xy_regress_total == 0.0
    assert r == pytest.approx(-0.05 + GRASP_HOLD_REWARD, abs=1e-9)
