"""Marker-visibility contract: a tag turned >70° from the tag camera, or dropped
by the DR detector (marker_dropout), keeps its last detected pose in the obs
while its per-tag age channel grows (a tag never detected since reset reads
zero with age pinned at MARKER_AGE_CAP_S). There is no reward term — losing
fresh pose data is the policy's own penalty."""

import mujoco
import numpy as np
import pytest

from src.base_env import (
    MARKER_AGE_CAP_S, MARKER_VIS_MAX_ANGLE_DEG,
    MARKER_VIS_NEAR_ANGLE_DEG, N_MARKERS, CameraModel, RuntimeEnvConfig,
    marker_dropout_prob, marker_world_normals, marker_world_poses, markers_visible,
    tag_cam_model,
)
from src.lift_env import SO101LiftEnv
from src.marker_noise import load_camera_intrinsics


def _prob(data, site_ids, cam, p_near, p_far):
    """marker_dropout_prob evaluated at the current MjData pose."""
    pos, _ = marker_world_poses(data, site_ids)
    normals = marker_world_normals(data, site_ids)
    return marker_dropout_prob(pos, normals, cam, p_near, p_far)


def _cam_looking_at(pos, target, intr):
    """A CameraModel at `pos` aimed straight at world `target` (so the target
    projects to the image center, always inside the frame) with intrinsics `intr`.
    Lets the facing-angle bound be tested in isolation from the FOV bound."""
    pos = np.asarray(pos, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - pos
    forward /= np.linalg.norm(forward)
    z_axis = -forward                       # MuJoCo camera +z points opposite the view
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return CameraModel(pos=pos, mat=np.column_stack([x_axis, y_axis, z_axis]), intr=intr)


def _refresh_detection(env):
    """Re-roll a camera detection of the env's current (manually posed) state
    and ingest it — what the synchronous camera does at each obs instant —
    returning the frame."""
    frame = env._process_frame(env._capture_camera_state())
    env._ingest_frame(env.data.time, frame)
    return frame


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


# The obs-slice assertions below assume marker rotations are in the obs, so the
# envs here pin the (non-default) marker_include_rot=True layout.
@pytest.fixture(scope="module")
def env():
    return SO101LiftEnv(env_cfg=_cfg(),
                        cfg=RuntimeEnvConfig(marker_include_rot=True))


# Obs layout: [qpos(6), qvel(6), pos_finger(3), rot_finger(3), pos_wrist(3),
# rot_wrist(3), marker_age(2), cube_tag(6), cube_age(1), extra(4), prev_actions]
MARKER_OBS_START = 12

# Geometric-visibility tests pose the arm so both tags face the tag camera and sit
# well inside its frame: the home pose puts marker_finger at ~0.275 m out the top of
# the real frame, which would mask the angle/dropout logic. This forward-low pose
# faces both tags at the camera with the finger ~0.14 m up, and rolling wrist_roll
# from here flips the finger across the facing-angle bound while it stays in frame,
# so the flip is purely angle.
LOW_BOTH_VISIBLE = np.array([0.0, -0.2174, 0.3455, 1.5381, 0.6928, 0.5])


def _pose_low(env):
    env.data.qpos[env.joint_qposadr] = LOW_BOTH_VISIBLE
    mujoco.mj_forward(env.model, env.data)


def test_in_view_frustum():
    """CameraModel.in_view projects a world point through the calibrated camera
    matrix and tests the image bounds — the real FOV, not a world-height proxy:
    on-axis points are in view, points past the horizontal half-FOV are out, and
    points behind the camera are out."""
    intr = load_camera_intrinsics()
    # Camera at the origin in MuJoCo default orientation (identity): +x right,
    # +y up, view along local -z. A world point (x, y, -d) is then in front.
    cam = CameraModel(pos=np.zeros(3), mat=np.eye(3), intr=intr)
    d = 1.0
    half_h = np.arctan((intr.width / 2) / intr.fx)   # horizontal half-FOV, radians
    assert cam.in_view([[0.0, 0.0, -d]])[0]                          # optical axis
    assert cam.in_view([[d * np.tan(0.9 * half_h), 0.0, -d]])[0]     # inside horizontal FOV
    assert not cam.in_view([[d * np.tan(1.1 * half_h), 0.0, -d]])[0]  # past the frame edge
    assert not cam.in_view([[0.0, 0.0, d]])[0]                       # behind the camera
    mask = cam.in_view([[0.0, 0.0, -d], [d * np.tan(1.1 * half_h), 0.0, -d]])
    assert mask.tolist() == [True, False]                            # one bool per point


def test_angle_threshold(env):
    """Visibility flips exactly at the 70° facing angle. The synthetic camera is
    aimed at the tag (centered in frame) so only the angle bound is exercised."""
    env.reset(seed=0)
    _pose_low(env)
    sid = env.marker_site_ids[0]
    site_pos = env.data.site_xpos[sid].copy()
    mat = env.data.site_xmat[sid].reshape(3, 3)
    normal, tangent = mat[:, 2], mat[:, 0]
    intr = load_camera_intrinsics()

    def cam_at(angle_deg, dist=0.4):
        a = np.radians(angle_deg)
        pos = site_pos + dist * (np.cos(a) * normal + np.sin(a) * tangent)
        return _cam_looking_at(pos, site_pos, intr)

    assert markers_visible(env.data, [sid], cam_at(0.0))[0]        # head-on
    assert markers_visible(env.data, [sid], cam_at(69.0))[0]       # just inside
    assert not markers_visible(env.data, [sid], cam_at(71.0))[0]   # just past
    assert not markers_visible(env.data, [sid], cam_at(90.0))[0]   # edge-on
    assert not markers_visible(env.data, [sid], cam_at(180.0))[0]  # behind the tag
    assert MARKER_VIS_MAX_ANGLE_DEG == 70.0


def test_out_of_frame_hides_facing_tag(env):
    """A tag squarely facing the camera but projecting outside the image frame is
    undetected — the real camera's FOV, not a world-height cutoff, is what hides it.
    The same tag centered in frame is detected, isolating the FOV bound."""
    env.reset(seed=0)
    _pose_low(env)
    sid = env.marker_site_ids[0]
    site_pos = env.data.site_xpos[sid].copy()
    mat = env.data.site_xmat[sid].reshape(3, 3)
    normal, tangent = mat[:, 2], mat[:, 0]
    cam_pos = site_pos + 0.4 * normal          # head-on: the facing-angle check passes
    intr = load_camera_intrinsics()
    # Aimed at the tag -> centered -> visible.
    assert markers_visible(env.data, [sid], _cam_looking_at(cam_pos, site_pos, intr))[0]
    # Same pose, camera panned ~56° off the tag -> tag leaves the frame -> hidden,
    # despite still facing the camera head-on.
    off_target = site_pos + 0.6 * tangent
    assert not markers_visible(env.data, [sid], _cam_looking_at(cam_pos, off_target, intr))[0]


def test_wrist_roll_sweep_flips_finger_visibility(env):
    """Rolling the wrist must produce both visible and hidden finger-tag poses;
    the finger obs slice must track the live pose while visible and freeze at
    the last visible pose while hidden."""
    env.reset(seed=0)
    _pose_low(env)
    roll_idx = 4  # wrist_roll in JOINT_NAMES
    lo, hi = env.joint_low[roll_idx], env.joint_high[roll_idx]
    seen = set()
    held = None  # expected finger slice while hidden (last visible detection)
    for roll in np.linspace(lo, hi, 25):
        env.data.qpos[env.joint_qposadr[roll_idx]] = roll
        mujoco.mj_forward(env.model, env.data)
        # Detection is rolled separately from _compute_obs; refresh it for this pose.
        _refresh_detection(env)
        visible = markers_visible(env.data, env.marker_site_ids, env.tag_cam)
        seen.add(bool(visible[0]))
        obs = env._compute_obs()
        finger_slice = obs[MARKER_OBS_START:MARKER_OBS_START + 6]
        pos, rot = marker_world_poses(env.data, env.marker_site_ids)
        if visible[0]:
            np.testing.assert_allclose(finger_slice,
                                       np.concatenate([pos[0], rot[0]]), atol=1e-6)
            held = finger_slice.copy()
        elif held is not None:
            np.testing.assert_array_equal(finger_slice, held)
    assert seen == {True, False}, "wrist_roll sweep should cover both visibility states"


def test_hidden_marker_holds_last_measurement():
    """An undetected tag's obs slice stays frozen at its last measured value —
    noise and bias included, no re-noising of held poses — while its age
    channel grows; a freshly detected tag reads age 0 (synchronous camera)."""
    noise = {"qpos_sigma": 0.01, "marker_rot_sigma": 0.02,
             "tag_px_noise": 0.4, "tag_depth_factor": 2.0,
             "live_sigma": 0.003, "precise_sigma": 0.003}
    bias = {"qpos_sigma": 0.01, "marker_pos_sigma": 0.005,
            "marker_rot_sigma": 0.02, "live_sigma": 0.005, "precise_sigma": 0.005,
            "sqrtm_depth_sigma": 0.008, "marker_common_sigma": 0.003}
    env = SO101LiftEnv(env_cfg=_cfg(),
                       cfg=RuntimeEnvConfig(obs_noise=noise, obs_bias=bias,
                                            marker_include_rot=True))
    env.reset(seed=0)
    age_start = MARKER_OBS_START + 6 * N_MARKERS
    prev = None
    checked = 0
    for _ in range(40):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        visible = markers_visible(env.data, env.marker_site_ids, env.tag_cam)
        for i in range(N_MARKERS):
            sl = obs[MARKER_OBS_START + 6 * i:MARKER_OBS_START + 6 * (i + 1)]
            if visible[i]:
                assert obs[age_start + i] < 1e-4
            elif prev is not None:
                np.testing.assert_array_equal(
                    sl, prev[MARKER_OBS_START + 6 * i:MARKER_OBS_START + 6 * (i + 1)])
                assert obs[age_start + i] > 0.0
                checked += 1
        prev = obs
        if term or trunc:
            env.reset()
            prev = None
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
    intr = load_camera_intrinsics()

    def cam_at(angle_deg, dist=0.4):
        a = np.radians(angle_deg)
        pos = site_pos + dist * (np.cos(a) * normal + np.sin(a) * tangent)
        return _cam_looking_at(pos, site_pos, intr)

    pn, pf = 0.2, 0.05
    assert _prob(env.data, [sid], cam_at(0.0), pn, pf)[0] == pf
    assert _prob(env.data, [sid], cam_at(60.0), pn, pf)[0] == pf
    assert _prob(env.data, [sid], cam_at(67.0), pn, pf)[0] == pn
    assert _prob(env.data, [sid], cam_at(71.0), pn, pf)[0] == 1.0
    assert MARKER_VIS_NEAR_ANGLE_DEG == 65.0


def test_dropout_one_starves_even_facing_tags():
    """marker_dropout near=far=1.0 means no tag is ever detected, even ones
    facing the camera (the geometric-visibility path alone would detect them):
    never-seen tags read all-zero poses with ages pinned at MARKER_AGE_CAP_S."""
    env = SO101LiftEnv(env_cfg=_cfg(),
                       cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0},
                                            marker_include_rot=True))
    env.reset(seed=0)
    age_start = MARKER_OBS_START + 6 * N_MARKERS
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(np.zeros(6, dtype=np.float32))
        markers = obs[MARKER_OBS_START:MARKER_OBS_START + 6 * N_MARKERS]
        assert np.all(markers == 0.0)
        np.testing.assert_array_equal(obs[age_start:age_start + N_MARKERS],
                                      np.full(N_MARKERS, MARKER_AGE_CAP_S,
                                              dtype=np.float32))
        if term or trunc:
            break


def test_dropout_rate_matches_prob():
    """Across many draws at a fixed pose, each tag's empirical drop rate matches
    marker_dropout_prob for that pose (whatever band it falls in)."""
    probs = {"near": 0.3, "far": 0.15}
    env = SO101LiftEnv(env_cfg=_cfg(),
                       cfg=RuntimeEnvConfig(marker_dropout=probs))
    env.reset(seed=2)
    expected = _prob(env.data, env.marker_site_ids, env.tag_cam,
                     probs["near"], probs["far"])
    n = 4000
    drops = np.zeros(N_MARKERS)
    for _ in range(n):
        frame = _refresh_detection(env)  # re-roll at the same (frozen) pose
        drops += ~frame.detected
    np.testing.assert_allclose(drops / n, expected, atol=0.03)


def test_marker_always_visible_disables_staleness():
    """marker_always_visible feeds every tag fresh (never held) even when
    geometry hides it and dropout would drop it — the easy-mode crutch for
    training from scratch."""
    env = SO101LiftEnv(
        env_cfg=_cfg(),
        cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0},
                             marker_always_visible=True, marker_include_rot=True),
    )
    env.reset(seed=0)
    age_start = MARKER_OBS_START + 6 * N_MARKERS
    roll_idx = 4  # sweep wrist_roll through poses that would hide the finger tag
    lo, hi = env.joint_low[roll_idx], env.joint_high[roll_idx]
    for roll in np.linspace(lo, hi, 25):
        env.data.qpos[env.joint_qposadr[roll_idx]] = roll
        mujoco.mj_forward(env.model, env.data)
        assert _refresh_detection(env).detected.all()
        obs = env._compute_obs()
        pos, rot = marker_world_poses(env.data, env.marker_site_ids)
        expected = np.hstack([pos, rot]).flatten()
        np.testing.assert_allclose(obs[MARKER_OBS_START:MARKER_OBS_START + 6 * N_MARKERS],
                                   expected, atol=1e-6)
        assert np.all(obs[age_start:age_start + N_MARKERS] < 1e-4)


def test_marker_hidden_ratio_in_info():
    env = SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())
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
    from tag_cam_model (cam_xpos), not the body-relative model.cam_pos."""
    pos = env.tag_cam_pos
    np.testing.assert_allclose(pos, tag_cam_model(env.model, env.data).pos, atol=1e-9)
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
