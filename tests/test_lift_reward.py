"""Smoke + behavior tests for the lift reward changes."""

import numpy as np
import pytest

from src.lift_env import (
    SO101LiftEnv, CONTACT_FORCE_COEFF, CUBE_MOTION_COEFF,
    HEIGHT_PROGRESS_COEFF, GRASP_HOLD_REWARD,
)


def _cfg():
    return {
        "action_scale": 0.035,
        "max_steps": 150,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "floor_contact_penalty": -0.10,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": -0.05,
        "target_height": 0.10,
    }


def test_env_resets_and_steps():
    env = SO101LiftEnv(env_cfg=_cfg())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)


def test_height_progress_gated_on_grasp(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    # Force not-grasped: cube should not get height-progress reward even if it rises.
    monkeypatch.setattr(env, "_detect_grasp", lambda: False)
    monkeypatch.setattr(env, "_arm_cube_contact_force", lambda: 0.0)
    env._prev_cube_pos = np.array([0.2, 0.0, 0.05])
    cube_pos = np.array([0.2, 0.0, 0.10])  # rose 5 cm
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.15]),
        cube_pos=cube_pos,
        ee_cube_dist=0.05,
        grasped=False,
        floor_contact=False,
    )
    # Should NOT contain HEIGHT_PROGRESS_COEFF * 0.05; instead motion penalty applies.
    assert reward < 0  # dominated by negatives
    # If gating broke, this would be > +9 from the height-progress term.
    assert reward > -2.0  # sanity bound


def test_height_progress_credited_when_grasped():
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    env._prev_cube_pos = np.array([0.2, 0.0, 0.05])
    cube_pos = np.array([0.2, 0.0, 0.10])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.10]),
        cube_pos=cube_pos,
        ee_cube_dist=0.0,
        grasped=True,
        floor_contact=False,
    )
    # Expected: TIME_PENALTY + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF*0.05
    assert reward == pytest.approx(-0.05 + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.05)


def test_floor_proximity_penalty(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    monkeypatch.setattr(env, "_arm_cube_contact_force", lambda: 0.0)
    env._prev_cube_pos = np.array([0.2, 0.0, 0.05])

    # Case 1: arm far from floor → no proximity penalty
    monkeypatch.setattr(env, "_min_arm_floor_dist", lambda thresh: thresh)
    cube_pos = np.array([0.2, 0.0, 0.05])
    reward_far, _, _ = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.05]),
        cube_pos=cube_pos, ee_cube_dist=0.0,
        grasped=False, floor_contact=False,
    )

    # Case 2: arm 1mm from floor → proximity penalty fires (-0.05)
    env._prev_cube_pos = np.array([0.2, 0.0, 0.05])  # reset
    monkeypatch.setattr(env, "_min_arm_floor_dist", lambda thresh: 0.001)
    reward_near, _, _ = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.05]),
        cube_pos=cube_pos, ee_cube_dist=0.0,
        grasped=False, floor_contact=False,
    )

    assert reward_near == pytest.approx(reward_far - 0.05)


def test_min_arm_floor_dist_runs():
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    d = env._min_arm_floor_dist(0.02)
    assert -0.02 <= d <= 0.02


def test_cube_motion_penalty_pregrasp(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    monkeypatch.setattr(env, "_arm_cube_contact_force", lambda: 0.0)
    env._prev_cube_pos = np.array([0.20, 0.0, 0.05])
    # Move cube 1 cm laterally in one step (dt ~ 0.0667s) → speed ~0.15 m/s
    cube_pos = np.array([0.21, 0.0, 0.05])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.21, 0.0, 0.05]),
        cube_pos=cube_pos,
        ee_cube_dist=0.0,
        grasped=False,
        floor_contact=False,
    )
    speed = 0.01 / env._step_dt
    expected = -0.05 + 0.0 + CUBE_MOTION_COEFF * speed  # ee_cube_dist=0, no contact force
    assert reward == pytest.approx(expected)
