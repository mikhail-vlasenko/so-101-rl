"""Smoke + behavior tests for the lift reward changes."""

import numpy as np
import pytest

from src.base_env import (
    RuntimeEnvConfig,
    _cube_faces_opposed,
    _dominant_loaded_cube_face,
)
from src.lift_env import (
    SO101LiftEnv, CUBE_MOTION_COEFF, CUBE_MOTION_DEADZONE,
    HEIGHT_PROGRESS_COEFF, GRASP_HOLD_REWARD, LIFT_BONUS, TIME_PENALTY,
    EE_CUBE_COEFF, JAW_CONTACT_REWARD, GRIPPER_CLOSE_COEFF,
)


def _cfg():
    return {
        "action_scale": 0.035,
        "use_servo_profile": True,
        "max_steps": 150,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "cube_smallest_face_only": False,
        "cube_no_flat_spawns": False,
        "floor_contact_penalty": -0.10,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": -0.05,
        "floor_force_coeff": 0.0,
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "target_height": 0.10,
    }


def test_env_resets_and_steps():
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim + env.priv_dim,)
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)


def test_episode_info_separates_two_jaw_contact_from_proper_grasp(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    env.step_count = env.max_steps - 1
    monkeypatch.setattr(env, "_detect_grasp", lambda: False)
    monkeypatch.setattr(env, "_has_gripper_contact", lambda: True)

    _, _, _, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert truncated
    assert info["grasp_ratio"] == 0.0
    assert info["two_jaw_contact_ratio"] == pytest.approx(1.0 / env.max_steps)


def test_height_progress_gated_on_grasp(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
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
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    # Stay below target_height=0.10 so this tests progress credit, not termination.
    env._prev_cube_pos = np.array([0.2, 0.0, 0.04])
    cube_pos = np.array([0.2, 0.0, 0.09])
    reward, terminated, _ = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.09]),
        cube_pos=cube_pos,
        ee_cube_dist=0.0,
        grasped=True,
        floor_contact=False,
    )
    assert not terminated
    # Expected: TIME_PENALTY + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF*0.05
    assert reward == pytest.approx(-0.05 + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.05)


def test_lift_success_bonus():
    """Crossing target_height while grasped terminates AND pays LIFT_BONUS —
    finishing must strictly beat holding just under the target (see lift_env.py)."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    env._prev_cube_pos = np.array([0.2, 0.0, 0.09])
    cube_pos = np.array([0.2, 0.0, 0.11])  # crosses target_height=0.10
    reward, terminated, info = env._compute_step(
        ee_pos=np.array([0.2, 0.0, 0.11]),
        cube_pos=cube_pos,
        ee_cube_dist=0.0,
        grasped=True,
        floor_contact=False,
    )
    assert terminated
    assert info["lift_success"]
    assert reward == pytest.approx(
        -0.05 + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.02 + LIFT_BONUS)


def test_floor_proximity_penalty(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
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
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    d = env._min_arm_floor_dist(0.02)
    assert -0.02 <= d <= 0.02


def test_cube_motion_penalty_pregrasp(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
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
    """Pre-grasp reward rises from one jaw to a closed opposing-face pinch."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    monkeypatch.setattr(env, "_gripper_closedness", lambda: 1.0)
    kwargs = dict(ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
                  ee_cube_dist=0.02, grasped=False, floor_contact=False)

    def step_with(n_jaw, opposed=False):
        env._prev_cube_pos = np.array([0.20, 0.0, 0.05])  # stationary: no motion penalty
        monkeypatch.setattr(env, "_n_jaw_contacts", lambda: n_jaw)
        monkeypatch.setattr(env, "_has_opposed_gripper_contact", lambda: opposed)
        r, _, _ = env._compute_step(**kwargs)
        return r

    r0, r1 = step_with(0), step_with(1)
    r_corner = step_with(2, opposed=False)
    r_opposed = step_with(2, opposed=True)
    # A corner pinch earns only the generic contact rung. The close reward is
    # reserved for two jaws loading opposite faces.
    assert r1 == pytest.approx(r0 + JAW_CONTACT_REWARD)
    assert r_corner == pytest.approx(r1)
    assert r_opposed == pytest.approx(
        r0 + JAW_CONTACT_REWARD + GRIPPER_CLOSE_COEFF)  # closedness=1


def test_loaded_contact_faces_are_classified_in_cube_frame():
    """Jaw contact quality follows the sponge faces, including under rotation."""
    angle = np.pi / 3.0
    cube_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    world_pos_x = cube_rotation @ np.array([1.0, 0.0, 0.0])
    world_neg_x = cube_rotation @ np.array([-1.0, 0.0, 0.0])
    world_pos_y = cube_rotation @ np.array([0.0, 1.0, 0.0])

    pos_x_face = _dominant_loaded_cube_face(
        cube_rotation, [world_pos_x, world_pos_y], [4.0, 1.0])
    neg_x_face = _dominant_loaded_cube_face(
        cube_rotation, [world_neg_x], [3.0])
    pos_y_face = _dominant_loaded_cube_face(
        cube_rotation, [world_pos_y], [3.0])

    assert pos_x_face == (0, 1)
    assert neg_x_face == (0, -1)
    assert _cube_faces_opposed(pos_x_face, neg_x_face)
    assert not _cube_faces_opposed(pos_x_face, pos_y_face)
    assert not _cube_faces_opposed(pos_x_face, pos_x_face)


def test_detect_grasp_rejects_adjacent_face_corner_pinch(monkeypatch):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    cube_pos = env._get_cube_pos().copy()
    monkeypatch.setattr(env, "_get_ee_pos", lambda: cube_pos.copy())
    env.data.qpos[env.joint_ids[env.gripper_idx]] = 0.0

    monkeypatch.setattr(env, "_has_opposed_gripper_contact", lambda: False)
    assert not env._detect_grasp()
    monkeypatch.setattr(env, "_has_opposed_gripper_contact", lambda: True)
    assert env._detect_grasp()


def test_poke_force_penalty_pregrasp(monkeypatch):
    """With shaping on, pre-grasp arm↔cube contact force is penalized (gentle approach)."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
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


def test_floor_force_penalty_applies_while_grasped(monkeypatch):
    """Arm↔floor contact force is penalized proportionally, including during the
    grasp — the binary contact/proximity terms alone let the policy lean on the
    floor at ~20 N for free once contact is already paid for."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    env.floor_force_coeff = -0.02
    env.floor_proximity_penalty = 0.0
    monkeypatch.setattr(env, "_arm_floor_contact_force", lambda: 15.0)
    env._prev_cube_pos = np.array([0.20, 0.0, 0.05])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
        ee_cube_dist=0.0, grasped=True, floor_contact=False,
    )
    assert reward == pytest.approx(
        TIME_PENALTY + GRASP_HOLD_REWARD + env.floor_force_coeff * 15.0)


def test_cube_tip_penalty_pregrasp(monkeypatch):
    """With shaping on, pre-grasp cube angular speed (rolling it over) is penalized."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
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
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())  # poke/tip coeffs are 0 in the fixture
    env.reset(seed=0)
    env.floor_proximity_penalty = 0.0
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
    for name in ("_arm_cube_contact_force", "_cube_angular_speed", "_min_arm_floor_dist",
                 "_arm_floor_contact_force"):
        monkeypatch.setattr(env, name, lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(f"{name} should not be called when its shaping term is off")))
    env._prev_cube_pos = np.array([0.20, 0.0, 0.05])
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.20, 0.0, 0.05]), cube_pos=np.array([0.20, 0.0, 0.05]),
        ee_cube_dist=0.0, grasped=False, floor_contact=False,
    )
    assert reward == pytest.approx(TIME_PENALTY)
