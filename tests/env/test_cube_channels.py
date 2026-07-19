"""Dual-channel cube observation contract (src/base_env.py + src/shape_obs.py).

The sponge is tracked tag-free through two channels built from the two sim
cameras' visible-surface sampling: `live` (both-view visible-centroid average +
age, biased toward the visible surface) and `precise` (body center + √M + age,
refreshed only while static and well visible). Both hold their last measurement
while undetected, exactly like the arm markers.
"""

import mujoco
import numpy as np
import pytest

from src.base_env import (
    MARKER_AGE_CAP_S,
    RuntimeEnvConfig,
    cube_surface_points_world,
    cube_visible_surface,
)
from src.lift_env import SO101LiftEnv
from src.shape_obs import (
    STATIC_DWELL_S,
    VISIBLE_FRACTION_MIN,
    box_sqrtm,
    sqrtm_from_upper,
    sqrtm_upper,
)


def _cfg():
    return {
        "action_scale": 0.07,
        "use_servo_profile": True,
        "max_steps": 300,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "cube_smallest_face_only": False,
        "cube_no_flat_spawns": False,
        "floor_contact_penalty": 0.0,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": 0.0,
        "floor_force_coeff": 0.0,
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "target_height": 0.10,
    }


# Default layout (marker_include_rot=False, prev_actions_n=2):
# [qpos(6), qvel(6), markers(6), marker_age(2), live(3), live_age(1),
#  center(3), sqrtM(6), precise_age(1), extra(4), prev_actions(12)]
LIVE = slice(20, 23)
LIVE_AGE = 23
CENTER = slice(24, 27)
SQRTM = slice(27, 33)
PRECISE_AGE = 33


@pytest.fixture(scope="module")
def env():
    return SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())


def _zero(env):
    return np.zeros(env.n_joints, dtype=np.float32)


def _true_center(env):
    return env.data.geom_xpos[env.cube_geom_id].copy()


def _true_sqrtm6(env):
    R = env.data.geom_xmat[env.cube_geom_id].reshape(3, 3)
    return sqrtm_upper(box_sqrtm(R, env.cube_half_extents))


def test_reset_serves_fresh_channels(env):
    """The both-view-visible spawn + the static pre-episode world make both
    channels fresh at reset: live measured (surface-biased), precise exact
    (clean env), both ages zero (synchronous camera)."""
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert env._cam_frame.live_detected
        assert obs[LIVE_AGE] < 1e-4
        assert obs[PRECISE_AGE] < 1e-4
        np.testing.assert_allclose(obs[CENTER], _true_center(env), atol=1e-6)
        np.testing.assert_allclose(obs[SQRTM], _true_sqrtm6(env), atol=1e-6)
        # The live centroid is a visible-surface point: near the box, not at
        # its center (the surface is at least hz away from the center).
        offset = np.linalg.norm(obs[LIVE] - _true_center(env))
        assert 1e-4 < offset < np.linalg.norm(env.cube_half_extents)


def test_live_biased_toward_cameras(env):
    """The visible-surface centroid must sit on the camera side of the body
    center — the deliberate bias the live channel reproduces instead of
    pretending to see the true center."""
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        to_cams = np.array([cam.pos - _true_center(env) for cam in env.cube_cams])
        to_cams /= np.linalg.norm(to_cams, axis=1, keepdims=True)
        mean_dir = to_cams.mean(axis=0)
        assert (obs[LIVE] - _true_center(env)) @ mean_dir > 0.0, seed


def test_out_of_view_holds_and_ages(env):
    """Teleport the cube outside both camera frames: both channels must hold
    their last served values while their ages grow with sim time."""
    obs0, _ = env.reset(seed=0)
    held_live = obs0[LIVE].copy()
    held_center = obs0[CENTER].copy()
    held_sqrtm = obs0[SQRTM].copy()

    env.data.qpos[env.cube_qpos_idx:env.cube_qpos_idx + 3] = (-1.0, -1.0, 0.0125)
    mujoco.mj_forward(env.model, env.data)
    points, normals = cube_surface_points_world(env.data, env.cube_geom_id,
                                                env.cube_half_extents)
    for cam in env.cube_cams:
        frac, _ = cube_visible_surface(env.model, env.data, cam,
                                       env.cube_body_id, points, normals)
        assert frac == 0.0

    prev_live_age = 0.0
    prev_precise_age = 0.0
    for _ in range(5):
        obs, *_ = env.step(_zero(env))
        assert not env._cam_frame.live_detected
        np.testing.assert_array_equal(obs[LIVE], held_live)
        np.testing.assert_array_equal(obs[CENTER], held_center)
        np.testing.assert_array_equal(obs[SQRTM], held_sqrtm)
        assert obs[LIVE_AGE] > prev_live_age
        assert obs[PRECISE_AGE] > prev_precise_age
        prev_live_age = obs[LIVE_AGE]
        prev_precise_age = obs[PRECISE_AGE]


def test_single_view_loss_stales_live():
    """The live channel needs BOTH views: a frame where one camera sees
    nothing must not be a live detection (mono fallback is out of scope)."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    env.reset(seed=0)
    state = env._capture_camera_state()
    assert state.cube_seen.all()
    one_lost = state._replace(cube_seen=np.array([True, False]))
    assert not env._process_frame(one_lost).live_detected
    assert env._process_frame(state).live_detected


def test_precise_refreshes_only_when_static():
    """Shove the cube: live keeps tracking while the precise channel freezes
    (static gate); once the cube settles for a full dwell, precise refreshes
    at the new pose."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    obs0, _ = env.reset(seed=1)
    spawn_center = obs0[CENTER].copy()

    # Give the cube a sideways kick and let it slide.
    env.data.qvel[env.cube_dofadr:env.cube_dofadr + 3] = (0.4, 0.0, 0.0)
    moved_steps = 0
    obs = obs0
    for _ in range(4):
        obs, *_ = env.step(_zero(env))
        if obs[PRECISE_AGE] > 0.01:
            moved_steps += 1
    assert moved_steps > 0, "precise must go stale while the cube moves"
    # Live keeps measuring the moving cube (both views still see it).
    assert obs[LIVE_AGE] < 0.01
    np.testing.assert_array_equal(obs[CENTER], spawn_center)

    # Let it settle: after a full dwell of stillness the gate reopens.
    settle_steps = int(np.ceil(STATIC_DWELL_S / env._step_dt)) + 3
    for _ in range(settle_steps):
        obs, *_ = env.step(_zero(env))
    assert obs[PRECISE_AGE] < 0.01
    np.testing.assert_allclose(obs[CENTER], _true_center(env), atol=1e-6)
    np.testing.assert_allclose(obs[SQRTM], _true_sqrtm6(env), atol=1e-6)


def test_partial_occlusion_blocks_refresh_but_not_live(env):
    """An arm pose shadowing part of the cube in some view (visible fraction
    under the gate, but something still visible in both) keeps the live
    channel fresh while the precise channel stops refreshing."""
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(5000):
        qpos = rng.uniform(env.joint_low, env.joint_high)
        env.data.qpos[env.joint_qposadr] = qpos
        mujoco.mj_forward(env.model, env.data)
        points, normals = cube_surface_points_world(env.data, env.cube_geom_id,
                                                    env.cube_half_extents)
        stats = [cube_visible_surface(env.model, env.data, cam,
                                      env.cube_body_id, points, normals)
                 for cam in env.cube_cams]
        fracs = np.array([s[0] for s in stats])
        if np.all(fracs > 0.0) and fracs.min() < VISIBLE_FRACTION_MIN:
            break
    else:
        pytest.fail("no arm pose partially occluded the cube in 5000 samples")
    frame = env._process_frame(env._capture_camera_state())
    assert frame.live_detected
    assert (frame.vis_frac < VISIBLE_FRACTION_MIN).any()
    # Feed it through ingest at a later time: live refreshes, precise doesn't.
    t = env.data.time + 0.5
    env._ingest_frame(t, frame)
    live, live_age, _, _, precise_age = env._obj_state.serve(t)
    assert live_age == 0.0
    np.testing.assert_array_equal(live, frame.live)
    assert precise_age > 0.0


def test_full_dropout_never_seen_reads_zero_with_capped_age():
    """Total detector dropout starves the channels from reset on: zeros with
    ages pinned at MARKER_AGE_CAP_S (the visible spawn only guarantees
    geometry, not the dropout roll)."""
    env = SO101LiftEnv(env_cfg=_cfg(),
                       cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0}))
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(obs[LIVE], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(obs[CENTER], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(obs[SQRTM], np.zeros(6, dtype=np.float32))
        assert obs[LIVE_AGE] == np.float32(MARKER_AGE_CAP_S)
        assert obs[PRECISE_AGE] == np.float32(MARKER_AGE_CAP_S)


def test_always_visible_bypasses_dropout_and_gates():
    """marker_always_visible feeds both channels fresh regardless of dropout
    or the static gate, with live at the (clean) TRUE center — the easy-mode
    crutch covers the cube channels exactly like the arm markers."""
    env = SO101LiftEnv(
        env_cfg=_cfg(),
        cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0},
                             marker_always_visible=True),
    )
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(6, dtype=np.float32))
        assert env._cam_frame.live_detected
        np.testing.assert_allclose(obs[LIVE], _true_center(env), atol=1e-6)
        np.testing.assert_allclose(obs[CENTER], _true_center(env), atol=1e-6)
        assert obs[LIVE_AGE] < 1e-4
        assert obs[PRECISE_AGE] < 1e-4


def test_sqrtm_obs_tracks_resting_face():
    """After the cube settles on a different face, the refreshed √M's diagonal
    reflects the new vertical spread — the 'which way is it shorter' signal."""
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        S = sqrtm_from_upper(obs[SQRTM])
        vertical_spread = np.sqrt(3.0) * np.sqrt(np.array([0, 0, 1.0]) @ (S @ S)
                                                 @ np.array([0, 0, 1.0]))
        np.testing.assert_allclose(vertical_spread, env.cube_rest_half_z, atol=1e-6)


def test_channel_metrics_in_info(env):
    env.reset(seed=0)
    info = {}
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        if term or trunc:
            break
    assert 0.0 <= info["live_hidden_ratio"] <= 1.0
    assert 0.0 <= info["precise_age_mean"] <= MARKER_AGE_CAP_S
