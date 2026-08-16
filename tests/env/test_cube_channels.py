"""Live-centroid plus current/held BPS environment integration."""

import mujoco
import numpy as np
import pytest

from src.base_env import (
    EE_OBJECT_DELTA_DIM,
    MARKER_AGE_CAP_S,
    _OCCLUDER_GEOMGROUP,
    RuntimeEnvConfig,
    cube_surface_points_world,
    cube_visible_surface,
    state_dim_for,
)
from src.bps import BPS_DISTANCE_DIM
from src.lift_env import SO101LiftEnv
from src.shape_obs import STATIC_DWELL_S


def _cfg():
    return {
        "action_scale": 0.07, "use_servo_profile": True, "max_steps": 300,
        "n_substeps": 10, "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15], "cube_smallest_face_only": False,
        "cube_no_flat_spawns": False, "floor_contact_penalty": 0.0,
        "floor_proximity_thresh": 0.003, "floor_proximity_penalty": 0.0,
        "floor_force_coeff": 0.0, "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0, "target_height": 0.10,
    }


STATE_DIM = state_dim_for(2, False)
LIVE = slice(20, 23)
LIVE_AGE = 23
EE_OBJECT_DELTA = slice(STATE_DIM - EE_OBJECT_DELTA_DIM, STATE_DIM)
BPS = slice(STATE_DIM, STATE_DIM + BPS_DISTANCE_DIM)
CENTER = slice(BPS.stop, BPS.stop + 3)
PRECISE_AGE = CENTER.stop
VALID_FRACTION = PRECISE_AGE + 1


@pytest.fixture(scope="module")
def env():
    return SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())


def _zero(env):
    return np.zeros(env.n_joints, dtype=np.float32)


def _true_center(env):
    return env.data.geom_xpos[env.cube_geom_id].copy()


def test_reset_serves_fresh_live_and_bps(env):
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert env._cam_frame.live_detected
        assert obs[LIVE_AGE] < 1e-4
        assert obs[PRECISE_AGE] < 1e-4
        assert np.all((obs[BPS] >= 0.0) & (obs[BPS] <= 1.0))
        assert obs[VALID_FRACTION] > 0.0
        assert np.linalg.norm(obs[CENTER] - _true_center(env)) < 0.02
        offset = np.linalg.norm(obs[LIVE] - _true_center(env))
        assert 1e-4 < offset < np.linalg.norm(env.cube_half_extents)


def test_bps_block_matches_shared_state(env):
    obs, _ = env.reset(seed=7)
    served = env._bps_state.serve(env.data.time)
    np.testing.assert_array_equal(obs[BPS], served.distances)
    np.testing.assert_array_equal(obs[CENTER], served.center_base)
    assert obs[PRECISE_AGE] == served.age_s
    assert obs[VALID_FRACTION] == served.valid_fraction


def test_ee_object_delta_uses_held_live_centroid(env):
    """The derived feature must use the same deployable held live channel in
    the observation, never the simulator's ground-truth sponge position."""
    obs, _ = env.reset(seed=7)
    np.testing.assert_allclose(
        obs[EE_OBJECT_DELTA], env._get_ee_pos() - obs[LIVE], atol=1e-7)

    held_live = obs[LIVE].copy()
    env.data.qpos[env.cube_qpos_idx:env.cube_qpos_idx + 3] = (-1.0, -1.0, 0.0125)
    mujoco.mj_forward(env.model, env.data)
    obs, *_ = env.step(_zero(env))
    np.testing.assert_array_equal(obs[LIVE], held_live)
    np.testing.assert_allclose(
        obs[EE_OBJECT_DELTA], env._get_ee_pos() - held_live, atol=1e-7)


def test_out_of_view_holds_and_ages(env):
    obs0, _ = env.reset(seed=0)
    held_live = obs0[LIVE].copy()
    held_bps = obs0[BPS].copy()
    held_center = obs0[CENTER].copy()
    env.data.qpos[env.cube_qpos_idx:env.cube_qpos_idx + 3] = (-1.0, -1.0, 0.0125)
    mujoco.mj_forward(env.model, env.data)
    previous_live_age = previous_bps_age = 0.0
    for _ in range(5):
        obs, *_ = env.step(_zero(env))
        np.testing.assert_array_equal(obs[LIVE], held_live)
        np.testing.assert_array_equal(obs[BPS], held_bps)
        np.testing.assert_array_equal(obs[CENTER], held_center)
        assert obs[LIVE_AGE] > previous_live_age
        assert obs[PRECISE_AGE] > previous_bps_age
        previous_live_age = obs[LIVE_AGE]
        previous_bps_age = obs[PRECISE_AGE]


def test_single_view_loss_stales_live(env):
    env.reset(seed=0)
    state = env._capture_camera_state()
    assert state.cube_seen.all()
    assert not env._process_frame(
        state._replace(cube_seen=np.array([True, False]),
                       live_detected=False)).live_detected


def test_bps_refreshes_only_when_static():
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
    obs0, _ = env.reset(seed=1)
    held = obs0[BPS].copy()
    env.data.qvel[env.cube_dofadr:env.cube_dofadr + 3] = (0.4, 0.0, 0.0)
    stale_steps = 0
    for _ in range(4):
        obs, *_ = env.step(_zero(env))
        stale_steps += obs[PRECISE_AGE] > 0.01
    assert stale_steps > 0
    np.testing.assert_array_equal(obs[BPS], held)
    for _ in range(int(np.ceil(STATIC_DWELL_S / env._step_dt)) + 3):
        obs, *_ = env.step(_zero(env))
    assert obs[PRECISE_AGE] < 0.01
    assert not np.array_equal(obs[BPS], held)


def test_full_dropout_never_seen_reads_zero_with_capped_age():
    env = SO101LiftEnv(
        env_cfg=_cfg(),
        cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0}),
    )
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[LIVE], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(obs[BPS], np.zeros(BPS_DISTANCE_DIM, dtype=np.float32))
    np.testing.assert_array_equal(obs[CENTER], np.zeros(3, dtype=np.float32))
    assert obs[LIVE_AGE] == MARKER_AGE_CAP_S
    assert obs[PRECISE_AGE] == MARKER_AGE_CAP_S
    assert obs[VALID_FRACTION] == 0.0


def test_camera_stand_ins_never_occlude(env):
    group1 = [i for i in range(env.model.ngeom) if env.model.geom_group[i] == 1]
    names = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in group1}
    assert names == {"tag_cam_body", "tag_cam_lens",
                     "tag_cam_aux_body", "tag_cam_aux_lens"}
    assert _OCCLUDER_GEOMGROUP[1] == 0
    env.reset(seed=0)
    points, normals = cube_surface_points_world(env.data, env.cube_geom_id,
                                                env.cube_half_extents)
    for cam in env.cube_cams:
        frac, _ = cube_visible_surface(env.model, env.data, cam,
                                       env.cube_body_id, points, normals)
        assert frac == 1.0


def test_channel_metrics_in_info(env):
    env.reset(seed=0)
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(_zero(env))
        if term or trunc:
            break
    assert 0.0 <= info["live_hidden_ratio"] <= 1.0
    assert 0.0 <= info["precise_age_mean"] <= MARKER_AGE_CAP_S
