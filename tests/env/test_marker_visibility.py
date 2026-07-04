"""Marker-visibility contract: a tag turned >70° from the tag camera, or dropped
by the DR detector (marker_dropout), is zeroed in the obs. There is no reward
term — losing the tag's pose is the policy's own penalty."""

import mujoco
import numpy as np
import pytest

from src.base_env import (
    MARKER_VIS_MAX_ANGLE_DEG, MARKER_VIS_MAX_HEIGHT_M, MARKER_VIS_NEAR_ANGLE_DEG,
    N_MARKERS, marker_dropout_prob, marker_world_normals, marker_world_poses,
    markers_visible, tag_cam_world_pos,
)
from src.lift_env import SO101LiftEnv


def _prob(data, site_ids, cam_pos, p_near, p_far):
    """marker_dropout_prob evaluated at the current MjData pose."""
    pos, _ = marker_world_poses(data, site_ids)
    normals = marker_world_normals(data, site_ids)
    return marker_dropout_prob(pos, normals, cam_pos, p_near, p_far)


def _refresh_detection(env):
    """Re-roll a camera detection of the env's current (manually posed) state —
    what the synchronous camera does at each obs instant — and return the frame."""
    env._cam_frame = env._process_frame(env._capture_camera_state())
    env._set_marker_render_colors(env._cam_frame.detected)
    return env._cam_frame


def _cfg():
    return {
        "action_scale": 0.07,
        "use_servo_profile": True,
        "max_steps": 300,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "cube_smallest_face_only": False,
        "floor_contact_penalty": 0.0,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": 0.0,
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "target_height": 0.10,
    }


# The obs-slice assertions below assume marker rotations are in the obs, so the
# envs here pin the (non-default) marker_include_rot=True layout.
@pytest.fixture(scope="module")
def env():
    return SO101LiftEnv(env_cfg=_cfg(), marker_include_rot=True)


# Obs layout: [qpos(6), qvel(6), pos_finger(3), rot_finger(3), pos_wrist(3),
# rot_wrist(3), cube(3), extra(4), prev_actions]
MARKER_OBS_START = 12

# Geometric-visibility tests must pose the arm under the 0.23 m height ceiling: the
# home pose puts marker_finger at ~0.275 m (above it), so the height cut-off would
# mask the angle/dropout logic. This forward-low pose faces both tags at the camera
# with the finger ~0.14 m up, and rolling wrist_roll from here still flips the
# finger across the angle bound (height stays ~0.14 m, so the flip is purely angle).
LOW_BOTH_VISIBLE = np.array([0.0, -0.2174, 0.3455, 1.5381, 0.6928, 0.5])


def _pose_low(env):
    env.data.qpos[env.joint_qposadr] = LOW_BOTH_VISIBLE
    mujoco.mj_forward(env.model, env.data)


def test_angle_threshold(env):
    """Visibility flips exactly at the 70° plane angle, per synthetic cam_pos."""
    env.reset(seed=0)
    _pose_low(env)
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


def test_height_ceiling_hides_high_tag(env):
    """A tag above MARKER_VIS_MAX_HEIGHT_M is undetected even when facing the camera
    head-on — it has left the top of the real camera frame. The home pose puts
    marker_finger above the ceiling, so reset(seed=0) gives a ready high tag."""
    env.reset(seed=0)
    sid = env.marker_site_ids[0]
    assert env.data.site_xpos[sid][2] > MARKER_VIS_MAX_HEIGHT_M
    normal = env.data.site_xmat[sid].reshape(3, 3)[:, 2]
    cam_head_on = env.data.site_xpos[sid] + 0.4 * normal   # angle check passes
    assert not markers_visible(env.data, [sid], cam_head_on)[0]
    assert _prob(env.data, [sid], cam_head_on, 0.2, 0.05)[0] == 1.0
    # The same tag, lowered under the ceiling and facing the camera, is visible.
    _pose_low(env)
    sid_low = env.marker_site_ids[0]
    assert env.data.site_xpos[sid_low][2] <= MARKER_VIS_MAX_HEIGHT_M
    normal = env.data.site_xmat[sid_low].reshape(3, 3)[:, 2]
    assert markers_visible(env.data, [sid_low], env.data.site_xpos[sid_low] + 0.4 * normal)[0]
    assert MARKER_VIS_MAX_HEIGHT_M == 0.23


def test_wrist_roll_sweep_flips_finger_visibility(env):
    """Rolling the wrist must produce both visible and hidden finger-tag poses,
    and the finger marker obs slice must be zero exactly when hidden."""
    env.reset(seed=0)
    _pose_low(env)
    roll_idx = 4  # wrist_roll in JOINT_NAMES
    lo, hi = env.joint_low[roll_idx], env.joint_high[roll_idx]
    seen = set()
    for roll in np.linspace(lo, hi, 25):
        env.data.qpos[env.joint_qposadr[roll_idx]] = roll
        mujoco.mj_forward(env.model, env.data)
        # Detection is rolled separately from _compute_obs; refresh it for this pose.
        _refresh_detection(env)
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
    noise = {"qpos_sigma": 0.01, "marker_pos_sigma": 0.005,
             "marker_rot_sigma": 0.02, "cube_sigma": 0.005}
    bias = {"qpos_sigma": 0.01, "marker_pos_sigma": 0.005,
            "marker_rot_sigma": 0.02, "cube_sigma": 0.005}
    env = SO101LiftEnv(env_cfg=_cfg(), obs_noise=noise, obs_bias=bias,
                       marker_include_rot=True)
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


def test_dropout_prob_bands(env):
    """marker_dropout_prob: 1.0 past MAX angle, p_near in the [NEAR, MAX] band,
    p_far for tags comfortably facing the camera."""
    env.reset(seed=0)
    _pose_low(env)
    sid = env.marker_site_ids[0]
    site_pos = env.data.site_xpos[sid].copy()
    mat = env.data.site_xmat[sid].reshape(3, 3)
    normal, tangent = mat[:, 2], mat[:, 0]

    def cam_at(angle_deg, dist=0.4):
        a = np.radians(angle_deg)
        return site_pos + dist * (np.cos(a) * normal + np.sin(a) * tangent)

    pn, pf = 0.2, 0.05
    assert _prob(env.data, [sid], cam_at(0.0), pn, pf)[0] == pf
    assert _prob(env.data, [sid], cam_at(60.0), pn, pf)[0] == pf
    assert _prob(env.data, [sid], cam_at(67.0), pn, pf)[0] == pn
    assert _prob(env.data, [sid], cam_at(71.0), pn, pf)[0] == 1.0
    assert MARKER_VIS_NEAR_ANGLE_DEG == 65.0


def test_dropout_one_zeroes_even_facing_tags():
    """marker_dropout near=far=1.0 zeroes every tag's obs, even ones facing the
    camera (the geometric-visibility path alone would leave them populated)."""
    env = SO101LiftEnv(env_cfg=_cfg(), marker_dropout={"near": 1.0, "far": 1.0},
                       marker_include_rot=True)
    env.reset(seed=0)
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(np.zeros(6, dtype=np.float32))
        markers = obs[MARKER_OBS_START:MARKER_OBS_START + 6 * N_MARKERS]
        assert np.all(markers == 0.0)
        if term or trunc:
            break


def test_dropout_rate_matches_prob():
    """Across many draws at a fixed pose, each tag's empirical drop rate matches
    marker_dropout_prob for that pose (whatever band it falls in)."""
    probs = {"near": 0.3, "far": 0.15}
    env = SO101LiftEnv(env_cfg=_cfg(), marker_dropout=probs)
    env.reset(seed=2)
    expected = _prob(env.data, env.marker_site_ids, env.tag_cam_pos,
                     probs["near"], probs["far"])
    n = 4000
    drops = np.zeros(N_MARKERS)
    for _ in range(n):
        frame = _refresh_detection(env)  # re-roll at the same (frozen) pose
        drops += ~frame.detected
    np.testing.assert_allclose(drops / n, expected, atol=0.03)


def test_marker_always_visible_disables_zeroing():
    """marker_always_visible feeds every tag (no zeroing) even when geometry hides
    it and dropout would drop it — the easy-mode crutch for training from scratch."""
    env = SO101LiftEnv(env_cfg=_cfg(), marker_dropout={"near": 1.0, "far": 1.0},
                       marker_always_visible=True, marker_include_rot=True)
    env.reset(seed=0)
    roll_idx = 4  # sweep wrist_roll through poses that would hide the finger tag
    lo, hi = env.joint_low[roll_idx], env.joint_high[roll_idx]
    for roll in np.linspace(lo, hi, 25):
        env.data.qpos[env.joint_qposadr[roll_idx]] = roll
        mujoco.mj_forward(env.model, env.data)
        assert _refresh_detection(env).detected.all()
        obs = env._compute_obs()
        markers = obs[MARKER_OBS_START:MARKER_OBS_START + 6 * N_MARKERS]
        assert np.any(markers != 0.0)


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
    """Camera sits to the right of the arm (-y), ~40 cm out, above the floor.

    The camera now hangs off a fixed mount body, so its world position comes
    from tag_cam_world_pos (cam_xpos), not the body-relative model.cam_pos."""
    pos = env.tag_cam_pos
    np.testing.assert_allclose(pos, tag_cam_world_pos(env.model, env.data), atol=1e-9)
    assert pos[1] < -0.2, "camera should be on the arm's right (-y)"
    assert pos[2] > 0.1, "camera should be above the workspace floor"
    assert 0.3 < np.linalg.norm(pos) < 0.5, "camera should be ~40 cm from the arm base"


def test_cam_body_geoms_inert(env):
    """The visible webcam stand-in must not collide or join the arm-geom set."""
    for name in ("tag_cam_body", "tag_cam_lens"):
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"geom '{name}' not found in model"
        assert env.model.geom_contype[gid] == 0
        assert env.model.geom_conaffinity[gid] == 0
        assert gid not in env.arm_geom_ids


def test_marker_render_colors_track_detection(env):
    """Each marker site's render rgba is green when detected this frame, red when
    not — retinted with each camera frame (visual only). Swept over wrist_roll so
    the finger tag deterministically passes through both states."""
    from src.base_env import MARKER_HIDDEN_RGBA, MARKER_VISIBLE_RGBA

    env.reset(seed=0)
    _pose_low(env)
    roll_idx = 4  # wrist_roll in JOINT_NAMES
    lo, hi = env.joint_low[roll_idx], env.joint_high[roll_idx]
    saw_visible = saw_hidden = False
    for roll in np.linspace(lo, hi, 25):
        env.data.qpos[env.joint_qposadr[roll_idx]] = roll
        mujoco.mj_forward(env.model, env.data)
        frame = _refresh_detection(env)
        for sid, det in zip(env.marker_site_ids, frame.detected):
            expected = MARKER_VISIBLE_RGBA if det else MARKER_HIDDEN_RGBA
            np.testing.assert_allclose(env.model.site_rgba[sid], expected, atol=1e-6)
            saw_visible |= bool(det)
            saw_hidden |= not bool(det)
    assert saw_visible and saw_hidden, "wrist sweep should exercise both color states"
