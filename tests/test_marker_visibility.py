"""Marker-visibility contract: tags turned >70° from the tag camera are zeroed
in the obs and penalized per step via marker_hidden_penalty."""

import mujoco
import numpy as np
import pytest

from src.base_env import (
    MARKER_VIS_MAX_ANGLE_DEG, N_MARKERS, TAG_CAM_NAME, markers_visible,
)
from src.lift_env import SO101LiftEnv


def _cfg(marker_hidden_penalty=0.0):
    return {
        "action_scale": 0.07,
        "max_steps": 300,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "floor_contact_penalty": 0.0,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": 0.0,
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "marker_hidden_penalty": marker_hidden_penalty,
        "target_height": 0.10,
    }


@pytest.fixture(scope="module")
def env():
    return SO101LiftEnv(env_cfg=_cfg())


# Obs layout: [qpos(6), qvel(6), pos_finger(3), rot_finger(3), pos_wrist(3),
# rot_wrist(3), cube(3), extra(4), prev_actions]
MARKER_OBS_START = 12


def test_angle_threshold(env):
    """Visibility flips exactly at the 70° plane angle, per synthetic cam_pos."""
    env.reset(seed=0)
    sid = env.marker_site_ids[0]
    site_pos = env.data.site_xpos[sid].copy()
    mat = env.data.site_xmat[sid].reshape(3, 3)
    normal, tangent = mat[:, 2], mat[:, 0]

    def cam_at(angle_deg, dist=0.4):
        a = np.radians(angle_deg)
        return site_pos + dist * (np.cos(a) * normal + np.sin(a) * tangent)

    assert markers_visible(env.data, [sid], cam_at(0.0))[0]        # head-on
    assert markers_visible(env.data, [sid], cam_at(69.0))[0]       # just inside
    assert not markers_visible(env.data, [sid], cam_at(71.0))[0]   # just past
    assert not markers_visible(env.data, [sid], cam_at(90.0))[0]   # edge-on
    assert not markers_visible(env.data, [sid], cam_at(180.0))[0]  # behind the tag
    assert MARKER_VIS_MAX_ANGLE_DEG == 70.0


def test_wrist_roll_sweep_flips_finger_visibility(env):
    """Rolling the wrist must produce both visible and hidden finger-tag poses,
    and the finger marker obs slice must be zero exactly when hidden."""
    env.reset(seed=0)
    roll_idx = 4  # wrist_roll in JOINT_NAMES
    lo, hi = env.joint_low[roll_idx], env.joint_high[roll_idx]
    seen = set()
    for roll in np.linspace(lo, hi, 25):
        env.data.qpos[env.joint_qposadr[roll_idx]] = roll
        mujoco.mj_forward(env.model, env.data)
        visible = markers_visible(env.data, env.marker_site_ids, env.tag_cam_pos)
        seen.add(bool(visible[0]))
        obs = env._compute_obs()
        finger_slice = obs[MARKER_OBS_START:MARKER_OBS_START + 6]
        if visible[0]:
            assert np.any(finger_slice != 0.0)
        else:
            assert np.all(finger_slice == 0.0)
    assert seen == {True, False}, "wrist_roll sweep should cover both visibility states"


def test_hidden_marker_zeroed_after_noise():
    """An undetected tag yields exactly zeros even with obs noise/bias active."""
    noise = {"qpos_sigma": 0.01, "qvel_sigma": 0.01, "marker_pos_sigma": 0.005,
             "marker_rot_sigma": 0.02, "cube_sigma": 0.005}
    bias = {"qpos_sigma": 0.01, "marker_pos_sigma": 0.005,
            "marker_rot_sigma": 0.02, "cube_sigma": 0.005}
    env = SO101LiftEnv(env_cfg=_cfg(), obs_noise=noise, obs_bias=bias)
    env.reset(seed=0)
    checked = 0
    for _ in range(40):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        visible = markers_visible(env.data, env.marker_site_ids, env.tag_cam_pos)
        for i in range(N_MARKERS):
            sl = obs[MARKER_OBS_START + 6 * i:MARKER_OBS_START + 6 * (i + 1)]
            if not visible[i]:
                assert np.all(sl == 0.0)
                checked += 1
        if term or trunc:
            env.reset()
    assert checked > 0, "random rollout never hid a marker; weaken the test setup"


def test_penalty_matches_hidden_count():
    """Reward with the penalty differs from the zero-penalty run by exactly
    penalty * n_hidden each step (dynamics are identical)."""
    penalty = -0.05
    env_p = SO101LiftEnv(env_cfg=_cfg(marker_hidden_penalty=penalty))
    env_0 = SO101LiftEnv(env_cfg=_cfg(marker_hidden_penalty=0.0))
    env_p.reset(seed=3)
    env_0.reset(seed=3)
    rng = np.random.default_rng(3)
    total_hidden = 0
    for _ in range(30):
        action = rng.uniform(-1, 1, size=6).astype(np.float32)
        _, r_p, term_p, trunc_p, _ = env_p.step(action)
        _, r_0, term_0, trunc_0, _ = env_0.step(action)
        n_hidden = N_MARKERS - int(
            markers_visible(env_p.data, env_p.marker_site_ids, env_p.tag_cam_pos).sum())
        total_hidden += n_hidden
        assert r_p - r_0 == pytest.approx(penalty * n_hidden)
        if term_p or trunc_p or term_0 or trunc_0:
            break
    assert total_hidden > 0, "rollout never hid a marker; penalty path untested"


def test_marker_hidden_ratio_in_info():
    env = SO101LiftEnv(env_cfg=_cfg())
    env.reset(seed=0)
    info = {}
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        if term or trunc:
            break
    assert "marker_hidden_ratio" in info
    assert 0.0 <= info["marker_hidden_ratio"] <= 1.0


def test_tag_cam_placement(env):
    """Camera sits to the right of the arm (-y), ~40 cm out, above the floor."""
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, TAG_CAM_NAME)
    pos = env.model.cam_pos[cam_id]
    assert pos[1] < -0.2, "camera should be on the arm's right (-y)"
    assert pos[2] > 0.1, "camera should be above the workspace floor"
    assert 0.3 < np.linalg.norm(pos) < 0.5, "camera should be ~40 cm from the arm base"
