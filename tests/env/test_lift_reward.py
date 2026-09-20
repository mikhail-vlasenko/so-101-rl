"""Smoke + behavior tests for the lift reward changes."""

import numpy as np
import mujoco
import pytest
from hydra import compose, initialize

from src.base_env import (
    RuntimeEnvConfig,
    _cube_faces_opposed,
    _dominant_loaded_cube_face,
)
from src.lift_env import (
    SO101LiftEnv,
    HEIGHT_PROGRESS_COEFF, GRASP_HOLD_REWARD, LIFT_BONUS,
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
        "cube_motion_coeff": -1.0,
        "cube_motion_deadzone": 0.002,
        "time_penalty": -0.01,
        "ee_cube_coeff": -0.5,
        "jaw_contact_reward": 0.05,
        "gripper_close_coeff": 0.05,
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
    assert reward == pytest.approx(env.time_penalty + env.ee_cube_coeff * 0.05)
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
    assert reward == pytest.approx(env.time_penalty + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.05)


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
        env.time_penalty + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.02 + LIFT_BONUS)


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
    _record_motion(env, env._prev_cube_pos, cube_pos, True, monkeypatch)
    reward, _, _ = env._compute_step(
        ee_pos=np.array([0.21, 0.0, 0.05]),
        cube_pos=cube_pos,
        ee_cube_dist=0.0,
        grasped=False,
        floor_contact=False,
    )
    speed = 0.01 / env._step_dt
    # Only horizontal speed past the deadzone is penalized; no jaw contact.
    expected = env.time_penalty + env.cube_motion_coeff * max(0.0, speed - env.cube_motion_deadzone)
    assert reward == pytest.approx(expected)


@pytest.fixture(params=["none", "light", "full"])
def motion_env(request, monkeypatch):
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "dr=none", f"shaping={request.param}"])
    env = SO101LiftEnv(env_cfg=cfg.lift_env, cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    monkeypatch.setattr(env, "_n_jaw_contacts", lambda: 0)
    monkeypatch.setattr(env, "_min_arm_floor_dist", lambda threshold: threshold)
    monkeypatch.setattr(env, "_arm_floor_contact_force", lambda: 0.0)
    monkeypatch.setattr(env, "_arm_cube_contact_force", lambda: 0.0)
    monkeypatch.setattr(env, "_cube_angular_speed", lambda: 0.0)
    assert env.cube_motion_coeff == {"none": 0.0, "light": -2.5, "full": -5.0}[request.param]
    assert env.cube_motion_deadzone == cfg.cube_motion_deadzone == 0.002
    assert env.time_penalty == cfg.lift_time_penalty == -0.01
    assert env.ee_cube_coeff == {"none": -0.5, "light": 0.0, "full": 0.0}[request.param]
    assert env.jaw_contact_reward == {"none": 0.05, "light": 0.0, "full": 0.0}[request.param]
    assert env.gripper_close_coeff == {"none": 0.05, "light": 0.0, "full": 0.0}[request.param]
    assert LIFT_BONUS > (GRASP_HOLD_REWARD + env.time_penalty) / (1.0 - cfg.train.gamma)
    yield env
    env.close()


@pytest.mark.parametrize("grasped", [False, True])
def test_distance_shaping_only_in_bootstrap_stage(motion_env, grasped):
    env = motion_env
    cube = env._get_cube_pos().copy()
    near, _, near_info = env._compute_step(cube, cube, 0.02, grasped, False)
    far, _, far_info = env._compute_step(cube, cube, 0.20, grasped, False)
    assert far - near == pytest.approx(env.ee_cube_coeff * 0.18)
    assert near_info["ee_cube_dist"] == 0.02
    assert far_info["ee_cube_dist"] == 0.20


@pytest.mark.parametrize("grasped", [False, True])
@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("speed", [0.0, 0.001, 0.002, 0.01, 0.05])
def test_horizontal_motion_penalty_across_stages_and_grasps(
        motion_env, grasped, supported, speed, monkeypatch):
    env = motion_env
    # Support is mocked independently of height and grasp; geometry is tested below.
    start = np.array([0.2, 0.0, 0.08])
    cube = start + np.array([-0.6, 0.8, 0.0]) * speed * env._step_dt
    env._prev_cube_pos = start.copy()
    _record_motion(env, start, cube, supported, monkeypatch)
    reward, terminated, info = env._compute_step(cube, cube, 0.0, grasped, False)
    expected_motion = (env.cube_motion_coeff * max(0.0, speed - env.cube_motion_deadzone)
                       if supported else 0.0)
    expected = env.time_penalty + (GRASP_HOLD_REWARD if grasped else 0.0) + expected_motion
    assert reward == pytest.approx(expected)
    assert not terminated
    assert info["motion_penalty"] == pytest.approx(expected_motion)
    assert info["table_slide_distance_m"] == pytest.approx(speed * env._step_dt if supported else 0.0)
    np.testing.assert_array_equal(env._prev_cube_pos, cube)
    # Once movement stops, the preceding tick's displacement must not be charged again.
    stopped, _, _ = env._compute_step(cube, cube, 0.0, grasped, False)
    assert stopped == pytest.approx(env.time_penalty + (GRASP_HOLD_REWARD if grasped else 0.0))


@pytest.mark.parametrize("grasped", [False, True])
def test_vertical_motion_stays_free_of_sliding_penalty(motion_env, grasped, monkeypatch):
    env = motion_env
    env._prev_cube_pos = np.array([0.2, 0.0, 0.05])
    cube = np.array([0.2, 0.0, 0.09])
    _record_motion(env, env._prev_cube_pos, cube, True, monkeypatch)
    reward, _, _ = env._compute_step(cube, cube, 0.0, grasped, False)
    expected = env.time_penalty
    if grasped:
        expected += GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.04
    assert reward == pytest.approx(expected)


def test_careful_approach_time_and_motion_tradeoff(motion_env, monkeypatch):
    env = motion_env
    start = np.array([0.2, 0.0, 0.05])
    env._prev_cube_pos = start.copy()
    wait_return = sum(env._compute_step(start, start, 0.0, False, False)[0]
                      for _ in range(10))
    assert 10 * env._step_dt == pytest.approx(2 / 3)
    assert wait_return == pytest.approx(-0.1)

    moved = start + np.array([0.05, 0.0, 0.0])
    _record_motion(env, start, moved, True, monkeypatch)
    slide_return, _, _ = env._compute_step(moved, moved, 0.0, False, False)
    if env.cube_motion_coeff:
        assert slide_return < wait_return


def _record_motion(env, start, end, supported, monkeypatch):
    """Replay a constant-velocity control interval without the observation pipeline."""
    monkeypatch.setattr(env, "_cube_has_table_contact", lambda: supported)
    env._motion_prev_cube_xy = start[:2].copy()
    for i in range(env.n_substeps):
        env.data.qpos[env.cube_qpos_idx:env.cube_qpos_idx + 3] = (
            start + (end - start) * (i + 1) / env.n_substeps)
        env._record_table_motion()


@pytest.mark.parametrize("landing", [False, True])
def test_support_transition_only_charges_supported_substeps(motion_env, monkeypatch, landing):
    env = motion_env
    start = np.array([0.2, 0.0, 0.05])
    env._motion_prev_cube_xy = start[:2].copy()
    end = start.copy()
    # Lift-off / landing halfway through one control tick. Airborne travel is
    # deliberately faster, and must not leak into the next supported substep.
    for i in range(env.n_substeps):
        supported = (i >= env.n_substeps // 2) == landing
        monkeypatch.setattr(env, "_cube_has_table_contact", lambda value=supported: value)
        end[0] += (0.03 if supported else 0.3) * env.model.opt.timestep
        env.data.qpos[env.cube_qpos_idx:env.cube_qpos_idx + 3] = end
        env._record_table_motion()
    _, _, info = env._compute_step(end, end, 0.0, True, False)
    distance = 0.03 * env._step_dt / 2
    assert info["table_slide_distance_m"] == pytest.approx(distance)
    assert info["motion_penalty"] == pytest.approx(
        env.cube_motion_coeff * (0.03 - env.cube_motion_deadzone) / 2)
    _, _, stopped = env._compute_step(end, end, 0.0, True, False)
    assert stopped["motion_penalty"] == 0.0
    assert stopped["table_slide_distance_m"] == 0.0
    end_info = {}
    env._on_episode_end(end_info)
    assert end_info["episode_table_slide_distance_m"] == pytest.approx(distance)
    assert end_info["episode_motion_penalty"] == pytest.approx(info["motion_penalty"])
    env.reset(seed=1)
    reset_info = {}
    env._on_episode_end(reset_info)
    assert reset_info["episode_table_slide_distance_m"] == 0.0
    assert reset_info["episode_motion_penalty"] == 0.0
    assert env._table_slide_distance == env._table_slide_excess_distance == 0.0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_table_contact_detection_uses_actual_rotated_geometry(axis):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    try:
        env.reset(seed=0)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if axis == 0:
            quat = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
        elif axis == 1:
            quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
        idx = env.cube_qpos_idx
        env.data.qpos[idx:idx + 3] = [0.25, 0.0, env.cube_half_extents[axis] - 0.0001]
        env.data.qpos[idx + 3:idx + 7] = quat
        mujoco.mj_forward(env.model, env.data)
        assert env._cube_has_table_contact()
        # Even a 1mm gap is free; the old near-table height proxy would include it.
        env.data.qpos[idx + 2] += 0.0011
        mujoco.mj_forward(env.model, env.data)
        assert not env._cube_has_table_contact()
    finally:
        env.close()


@pytest.mark.parametrize("airborne", [False, True])
def test_physics_substeps_report_table_motion_and_episode_totals(airborne):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    try:
        env.reset(seed=0, options={
            "cube_pos": np.array([0.25, 0.0, 0.15 if airborne else
                                  env.model.geom_size[env.cube_geom_id, 2] - 0.0001]),
            "cube_quat": np.array([1.0, 0.0, 0.0, 0.0])})
        env.data.qvel[env.cube_dofadr] = 0.2
        mujoco.mj_forward(env.model, env.data)
        env.step_count = env.max_steps - 1
        _, _, _, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert truncated
        if airborne:
            assert info["motion_penalty"] == 0.0
            assert info["table_slide_distance_m"] == 0.0
        else:
            assert info["motion_penalty"] < 0.0
            assert info["table_slide_distance_m"] > 0.0
        assert info["episode_motion_penalty"] == info["motion_penalty"]
        assert info["episode_table_slide_distance_m"] == info["table_slide_distance_m"]
    finally:
        env.close()


@pytest.mark.parametrize("time_penalty", [0.0, -0.01, -0.05])
def test_time_penalty_config_override(time_penalty):
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "shaping=none", f"lift_time_penalty={time_penalty}"])
    env = SO101LiftEnv(env_cfg=cfg.lift_env, cfg=RuntimeEnvConfig())
    try:
        env.reset(seed=0)
        cube = env._get_cube_pos().copy()
        reward, _, _ = env._compute_step(cube, cube, 0.0, False, False)
        assert env.time_penalty == time_penalty
        assert reward == pytest.approx(time_penalty)
    finally:
        env.close()


@pytest.mark.parametrize("key,value", [
    ("cube_motion_coeff", 0.1), ("cube_motion_deadzone", -0.001),
    ("time_penalty", 0.01), ("ee_cube_coeff", 0.1),
    ("jaw_contact_reward", -0.05), ("gripper_close_coeff", -0.05)])
def test_invalid_motion_penalty_config_fails_loudly(key, value):
    cfg = _cfg()
    cfg[key] = value
    with pytest.raises(AssertionError, match=key):
        SO101LiftEnv(env_cfg=cfg, cfg=RuntimeEnvConfig())


@pytest.mark.parametrize("closedness", [0.0, 0.5, 1.0])
def test_grasp_contact_ladder_pregrasp(motion_env, monkeypatch, closedness):
    """Only bootstrap shaping pays for touching/closing before a proper grasp."""
    env = motion_env
    monkeypatch.setattr(env, "_gripper_closedness", lambda: closedness)
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
    assert r1 == pytest.approx(r0 + env.jaw_contact_reward)
    assert r_corner == pytest.approx(r1)
    assert r_opposed == pytest.approx(
        r0 + env.jaw_contact_reward + env.gripper_close_coeff * closedness)
    # Proper grasp keeps its hold/progress reward; pre-grasp bonuses don't stack.
    env._prev_cube_pos = kwargs["cube_pos"] - np.array([0.0, 0.0, 0.01])
    kwargs["grasped"] = True
    grasp_reward, terminated, _ = env._compute_step(**kwargs)
    assert not terminated
    assert grasp_reward == pytest.approx(r0 + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.01)


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


@pytest.mark.parametrize("gripper_angle", [0.0, 0.3, 0.6])
def test_detect_grasp_rejects_adjacent_face_corner_pinch(monkeypatch, gripper_angle):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    cube_pos = env._get_cube_pos().copy()
    monkeypatch.setattr(env, "_get_ee_pos", lambda: cube_pos.copy())
    env.data.qpos[env.joint_ids[env.gripper_idx]] = gripper_angle

    monkeypatch.setattr(env, "_has_opposed_gripper_contact", lambda: False)
    assert not env._detect_grasp()
    monkeypatch.setattr(env, "_has_opposed_gripper_contact", lambda: True)
    assert env._detect_grasp()


@pytest.mark.parametrize("height", [0.09, 0.10, 0.11])
@pytest.mark.parametrize("gripper_angle", [0.3, 0.6])
def test_wide_grasp_lift_progress_and_success(monkeypatch, height, gripper_angle):
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    cube_pos = np.array([0.2, 0.0, height])
    monkeypatch.setattr(env, "_get_cube_pos", lambda: cube_pos.copy())
    monkeypatch.setattr(env, "_get_ee_pos", lambda: cube_pos.copy())
    monkeypatch.setattr(env, "_has_opposed_gripper_contact", lambda: True)
    monkeypatch.setattr(env, "_min_arm_floor_dist", lambda thresh: thresh)
    env.data.qpos[env.joint_ids[env.gripper_idx]] = gripper_angle
    env._prev_cube_pos = cube_pos - np.array([0.0, 0.0, 0.02])

    reward, terminated, info = env._compute_step(
        ee_pos=env._get_ee_pos(), cube_pos=cube_pos, ee_cube_dist=0.0,
        grasped=env._detect_grasp(), floor_contact=False,
    )

    success = height >= env.target_height
    assert info["grasped"]
    assert bool(terminated) == success
    assert bool(info["lift_success"]) == success
    expected = env.time_penalty + GRASP_HOLD_REWARD + HEIGHT_PROGRESS_COEFF * 0.02
    if success:
        expected += LIFT_BONUS
    assert reward == pytest.approx(expected)
    env.close()


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
    assert reward == pytest.approx(env.time_penalty + env.poke_force_coeff * 20.0)


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
        env.time_penalty + GRASP_HOLD_REWARD + env.floor_force_coeff * 15.0)


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
    assert reward == pytest.approx(env.time_penalty + env.cube_tip_coeff * 2.0)


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
    assert reward == pytest.approx(env.time_penalty)
