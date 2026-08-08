"""Tests for the sponge-box spawn (orientation sampling) and marker observations.

The 6 x 4 x 2.5 cm sponge box may spawn resting on ANY face (tag-free tracking
imposes no facing constraint): side-standing 4 or 6 cm tall AND flat on the
largest face, all occurring, stable under settling; the curriculum crutch flags
restrict the set. Every spawn must be comfortably visible to both cameras. The
marker obs dims must carry the world poses of the marker_finger / marker_wrist
sites (xyz + axis-angle rotation vector each).
"""

import numpy as np
import pytest

import mujoco
from hydra import compose, initialize

from src.base_env import (
    RuntimeEnvConfig, cube_surface_points_world, cube_visible_surface,
    marker_world_poses, markers_visible, obs_dim_for, sample_cube_orientation,
)
from src.lift_env import SO101LiftEnv
from src.shape_obs import VISIBLE_FRACTION_MIN


MARKER_FINGER_POS = slice(12, 15)
MARKER_FINGER_ROT = slice(15, 18)
MARKER_WRIST_POS = slice(18, 21)
MARKER_WRIST_ROT = slice(21, 24)

HALF = np.array([0.03, 0.02, 0.0125])


@pytest.fixture(scope="module")
def env():
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    # marker_include_rot=True so the MARKER_*_ROT obs slices below are populated.
    return SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                        cfg=RuntimeEnvConfig(obs_noise=None, obs_bias=None,
                                             marker_include_rot=True))


def test_sample_cube_orientation_all_faces():
    """Default sampling rests the box on any face: all three rest heights
    occur, including the flat largest-face pose, and the vertical body axis is
    always exactly vertical."""
    rng = np.random.default_rng(0)
    rest_halves = set()
    for _ in range(300):
        quat, rest_half_z = sample_cube_orientation(rng, HALF)
        assert rest_half_z in (0.03, 0.02, 0.0125)
        rest_halves.add(rest_half_z)
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-9)
        # Exactly one body axis is vertical (its |world z| is 1), the others flat.
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        z_components = np.abs(mat.reshape(3, 3)[2, :])
        assert np.isclose(z_components.max(), 1.0, atol=1e-9)
        assert np.sum(z_components > 1e-9) == 1
    assert rest_halves == {0.03, 0.02, 0.0125}, "all three rest heights must occur"


def test_sample_cube_orientation_no_flat_spawns():
    """cube_no_flat_spawns excludes the largest-face poses: only the historic
    side-standing spawns (both occurring)."""
    rng = np.random.default_rng(0)
    rest_halves = set()
    for _ in range(200):
        quat, rest_half_z = sample_cube_orientation(rng, HALF, no_flat_spawns=True)
        assert rest_half_z in (0.03, 0.02)
        rest_halves.add(rest_half_z)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        # The z body axis (smallest half-extent) must end up horizontal.
        assert abs(mat.reshape(3, 3)[:, 2][2]) < 1e-9
    assert rest_halves == {0.03, 0.02}, "both standing heights must occur"


def test_sample_cube_orientation_smallest_face_only():
    """smallest_face_only always stands on the hy*hz face (x-axis vertical, tallest)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        quat, rest_half_z = sample_cube_orientation(rng, HALF, smallest_face_only=True)
        assert rest_half_z == 0.03  # hx: standing on the smallest face
        # The geom x-axis (largest half-extent) must end up vertical.
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        x_axis_world = mat.reshape(3, 3)[:, 0]
        assert abs(x_axis_world[2]) == pytest.approx(1.0, abs=1e-9)


def test_sample_cube_orientation_rejects_unordered_extents():
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        sample_cube_orientation(rng, np.array([0.03, 0.03, 0.03]))


def test_spawn_is_stable_and_at_rest_height(env):
    """After reset + settling with zero action, the cube must stay resting."""
    for seed in range(5):
        env.reset(seed=seed)
        spawn_z = env._get_cube_pos()[2]
        assert spawn_z == pytest.approx(env.cube_rest_half_z)
        for _ in range(10):
            env.step(np.zeros(6, dtype=np.float32))
        # Still resting on the same face (allow a little contact settling).
        assert env._get_cube_pos()[2] == pytest.approx(env.cube_rest_half_z, abs=0.002)


def test_configured_workspace_extends_baseward_and_narrows_lateral(env):
    np.testing.assert_allclose(env.cube_low, (0.10, -0.10))
    np.testing.assert_allclose(env.cube_high, (0.30, 0.10))
    for seed in range(10):
        env.reset(seed=seed)
        xy = env._get_cube_pos()[:2]
        assert np.all(xy >= env.cube_low)
        assert np.all(xy <= env.cube_high)


def test_spawn_flat_faces_occur(env):
    """The env-level spawn (rejection-sampled on visibility) must still
    produce flat largest-face poses — they can't all be rejected."""
    rest_halves = set()
    for seed in range(40):
        env.reset(seed=seed)
        rest_halves.add(round(env.cube_rest_half_z, 4))
    assert round(float(env.cube_half_extents[2]), 4) in rest_halves, \
        "no flat spawn in 40 resets"


def test_spawn_visible_to_both_cameras(env):
    """Every spawn must be comfortably visible in BOTH camera views (at least
    VISIBLE_FRACTION_MIN of the facing surface) with no arm contact — the sim
    twin of placing the real sponge in both cameras' clear view."""
    for seed in range(50):
        env.reset(seed=seed)
        points, normals = cube_surface_points_world(env.data, env.cube_geom_id,
                                                    env.cube_half_extents)
        for cam in env.cube_cams:
            frac, centroid = cube_visible_surface(env.model, env.data, cam,
                                                  env.cube_body_id, points, normals)
            assert frac >= VISIBLE_FRACTION_MIN, seed
            assert centroid is not None
        assert not env._cube_arm_contact(), seed


# Forward-low pose that faces both arm tags at the camera and keeps them inside
# its frame (shared with tests/env/test_marker_visibility.py). A raised reset pose
# leaves the tags out of frame, where the obs holds a stale pose instead.
LOW_BOTH_VISIBLE = np.array([0.0, -0.2174, 0.3455, 1.5381, 0.6928, 0.5])


def test_marker_obs_match_site_poses(env):
    """Clean obs marker dims must equal FK world poses of the marker sites when
    both tags are visible — hidden tags would hold their last detection instead
    (covered by test_marker_visibility.py)."""
    env.reset(seed=5)
    env.data.qpos[env.joint_qposadr] = LOW_BOTH_VISIBLE
    mujoco.mj_forward(env.model, env.data)
    assert markers_visible(env.data, env.marker_site_ids, env.tag_cam).all()
    # Re-roll a camera detection of this posed state and rebuild the obs from it.
    env._ingest_frame(env.data.time, env._process_frame(env._capture_camera_state()))
    obs = env._compute_obs()
    marker_pos, marker_rot = marker_world_poses(env.data, env.marker_site_ids)
    np.testing.assert_allclose(obs[MARKER_FINGER_POS], marker_pos[0], atol=1e-6)
    np.testing.assert_allclose(obs[MARKER_FINGER_ROT], marker_rot[0], atol=1e-6)
    np.testing.assert_allclose(obs[MARKER_WRIST_POS], marker_pos[1], atol=1e-6)
    np.testing.assert_allclose(obs[MARKER_WRIST_ROT], marker_rot[1], atol=1e-6)
    # Markers ride on different links — poses must differ.
    assert np.linalg.norm(marker_pos[0] - marker_pos[1]) > 0.01


def test_marker_rot_is_rotation_vector(env):
    """The rot obs must be axis-angle: reconstructing the quaternion from it
    must reproduce the site orientation."""
    env.reset(seed=0)
    _, marker_rot = marker_world_poses(env.data, env.marker_site_ids)
    for i, sid in enumerate(env.marker_site_ids):
        quat_expected = np.empty(4)
        mujoco.mju_mat2Quat(quat_expected, env.data.site_xmat[sid])
        angle = np.linalg.norm(marker_rot[i])
        axis = marker_rot[i] / angle
        quat_rebuilt = np.empty(4)
        mujoco.mju_axisAngle2Quat(quat_rebuilt, axis, angle)
        # Quaternion double cover: q and -q are the same rotation.
        assert (np.allclose(quat_rebuilt, quat_expected, atol=1e-6)
                or np.allclose(quat_rebuilt, -quat_expected, atol=1e-6))


def test_default_obs_drops_marker_rotations():
    """The default (marker_include_rot=False) obs carries marker positions only:
    6 dims shorter than the rot-included layout, and the marker section equals the
    two FK positions back-to-back (the live centroid follows the marker ages)."""
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    # marker_always_visible so neither pose goes stale regardless of the spawn
    # pose — and the live channel serves the clean true center.
    env = SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                       cfg=RuntimeEnvConfig(marker_always_visible=True))
    assert env.marker_include_rot is False
    assert env.obs_dim == obs_dim_for(env.prev_actions_n, marker_include_rot=False)
    assert env.obs_dim == obs_dim_for(env.prev_actions_n, marker_include_rot=True) - 6

    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim + env.priv_dim,)
    marker_pos, _ = marker_world_poses(env.data, env.marker_site_ids)
    # qpos(6)+qvel(6)=12, then pos_finger(3), pos_wrist(3), marker_age(2),
    # then live(3).
    np.testing.assert_allclose(obs[12:15], marker_pos[0], atol=1e-6)
    np.testing.assert_allclose(obs[15:18], marker_pos[1], atol=1e-6)
    np.testing.assert_allclose(obs[20:23], env.data.geom_xpos[env.cube_geom_id],
                               atol=1e-6)
