"""Tests for the sponge-box spawn (orientation sampling) and marker observations.

The 3 x 2 x 1.5 cm sponge box must spawn standing on one of its two non-largest
faces (2 or 3 cm tall, both occurring), stable under settling. The marker obs
dims must carry the world poses of the marker_finger / marker_wrist sites
(xyz + axis-angle rotation vector each).
"""

import numpy as np
import pytest

import mujoco
from hydra import compose, initialize

from src.base_env import marker_world_poses, markers_visible, sample_cube_orientation
from src.lift_env import SO101LiftEnv


MARKER_FINGER_POS = slice(12, 15)
MARKER_FINGER_ROT = slice(15, 18)
MARKER_WRIST_POS = slice(18, 21)
MARKER_WRIST_ROT = slice(21, 24)


@pytest.fixture(scope="module")
def env():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    return SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                        obs_noise=None, obs_bias=None)


def test_sample_cube_orientation_rest_heights():
    rng = np.random.default_rng(0)
    half = np.array([0.015, 0.01, 0.0075])
    rest_halves = set()
    for _ in range(200):
        quat, rest_half_z = sample_cube_orientation(rng, half)
        assert rest_half_z in (0.015, 0.01)
        rest_halves.add(rest_half_z)
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-9)
        # The face touching the floor must be a non-largest face: the geom
        # z-axis (smallest half-extent) must end up horizontal.
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        z_axis_world = mat.reshape(3, 3)[:, 2]
        assert abs(z_axis_world[2]) < 1e-9
    assert rest_halves == {0.015, 0.01}, "both standing heights must occur"


def test_sample_cube_orientation_smallest_face_only():
    """smallest_face_only always stands on the hy*hz face (x-axis vertical, tallest)."""
    rng = np.random.default_rng(0)
    half = np.array([0.015, 0.01, 0.0075])
    for _ in range(200):
        quat, rest_half_z = sample_cube_orientation(rng, half, smallest_face_only=True)
        assert rest_half_z == 0.015  # hx: standing on the smallest face
        # The geom x-axis (largest half-extent) must end up vertical.
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        x_axis_world = mat.reshape(3, 3)[:, 0]
        assert abs(x_axis_world[2]) == pytest.approx(1.0, abs=1e-9)


def test_sample_cube_orientation_rejects_unordered_extents():
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        sample_cube_orientation(rng, np.array([0.015, 0.015, 0.015]))


def test_spawn_is_stable_and_at_rest_height(env):
    """After reset + settling with zero action, the cube must stay standing."""
    for seed in range(5):
        env.reset(seed=seed)
        spawn_z = env._get_cube_pos()[2]
        assert spawn_z == pytest.approx(env.cube_rest_half_z)
        for _ in range(10):
            env.step(np.zeros(6, dtype=np.float32))
        # Still standing on the same face (allow a little contact settling).
        assert env._get_cube_pos()[2] == pytest.approx(env.cube_rest_half_z, abs=0.002)


def test_marker_obs_match_site_poses(env):
    """Clean obs marker dims must equal FK world poses of the marker sites.

    Seed 4 spawns the arm with both tags facing tag_cam — hidden tags are
    zeroed instead (covered by tests/test_marker_visibility.py)."""
    obs, _ = env.reset(seed=4)
    assert markers_visible(env.data, env.marker_site_ids, env.tag_cam_pos).all()
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
