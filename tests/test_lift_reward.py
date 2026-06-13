"""Smoke + behavior tests for the lift reward changes."""

import numpy as np
import pytest

from src.lift_env import (
    SO101LiftEnv, CUBE_MOTION_COEFF, CUBE_MOTION_DEADZONE,
    HEIGHT_PROGRESS_COEFF, GRASP_HOLD_REWARD, TIME_PENALTY,
    EE_CUBE_COEFF, JAW_CONTACT_REWARD, GRIPPER_CLOSE_COEFF,
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
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "marker_hidden_penalty": 0.0,
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
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
    env._prev_cube_pos = np.array([0.2, 0.0, 0.05])
    cube_pos = np.array([0.2, 0.0, 0.10])  # rose 5 cm (vertical only)
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.15]),
        cube_pos=cube_pos,
        ee_cube_dist=0.05,
        grasped=False,
        floor_contact=False,
    )
    # Pre-grasp vertical rise must NOT be credited the height-progress term, and
    # horizontal-only motion penalty means a purely vertical move isn't penalized.
    assert reward == pytest.approx(TIME_PENALTY + EE_CUBE_COEFF * 0.05)
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
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
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
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
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
    # Only horizontal speed past the deadzone is penalized; no jaw contact.
    expected = TIME_PENALTY + CUBE_MOTION_COEFF * max(0.0, speed - CUBE_MOTION_DEADZONE)
    assert reward == pytest.approx(expected)


def test_grasp_contact_ladder_pregrasp(monkeypatch):
    """Pre-grasp reward rises with jaw contact: touch one jaw, then close on both."""
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    monkeypatch.setattr(env, "_gripper_closedness", lambda: 1.0)
    kwargs = dict(ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
                  ee_cube_dist=0.02, grasped=False, floor_contact=False)

    def step_with(n_jaw):
        env._prev_cube_pos = np.array([0.20, 0.0, 0.05])  # stationary: no motion penalty
        monkeypatch.setattr(env, "_n_jaw_contacts", lambda: n_jaw)
        r, _, _ = env._compute_step(**kwargs)
        return r

    r0, r1, r2 = step_with(0), step_with(1), step_with(2)
    # One jaw earns the contact reward; both jaws additionally earn the close reward.
    assert r1 == pytest.approx(r0 + JAW_CONTACT_REWARD)
    assert r2 == pytest.approx(r0 + JAW_CONTACT_REWARD + GRIPPER_CLOSE_COEFF)  # closedness=1


def test_poke_force_penalty_pregrasp(monkeypatch):
    """With shaping on, pre-grasp arm↔cube contact force is penalized (gentle approach)."""
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    env.poke_force_coeff = -0.002
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
    monkeypatch.setattr(env, "_arm_cube_contact_force", lambda: 20.0)
    env._prev_cube_pos = np.array([0.20, 0.0, 0.05])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
        ee_cube_dist=0.0, grasped=False, floor_contact=False,
    )
    assert reward == pytest.approx(TIME_PENALTY + env.poke_force_coeff * 20.0)


def test_cube_tip_penalty_pregrasp(monkeypatch):
    """With shaping on, pre-grasp cube angular speed (rolling it over) is penalized."""
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    env.cube_tip_coeff = -0.5
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
    monkeypatch.setattr(env, "_cube_angular_speed", lambda: 2.0)
    env._prev_cube_pos = np.array([0.20, 0.0, 0.05])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
        ee_cube_dist=0.0, grasped=False, floor_contact=False,
    )
    assert reward == pytest.approx(TIME_PENALTY + env.cube_tip_coeff * 2.0)


def test_shaping_terms_skipped_when_zero(monkeypatch):
    """shaping=none: poke/tip helpers and the proximity check aren't even called."""
    env = SO101LiftEnv(env_cfg=_cfg())  # poke/tip coeffs are 0 in the fixture
    env.reset(seed=0)
    env.floor_proximity_penalty = 0.0
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
    for name in ("_arm_cube_contact_force", "_cube_angular_speed", "_min_arm_floor_dist"):
        monkeypatch.setattr(env, name, lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(f"{name} should not be called when its shaping term is off")))
    env._prev_cube_pos = np.array([0.20, 0.0, 0.05])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
        ee_cube_dist=0.0, grasped=False, floor_contact=False,
    )
    assert reward == pytest.approx(TIME_PENALTY)
