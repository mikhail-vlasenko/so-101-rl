"""Pin the OpenCV -> MuJoCo camera-frame conversion used to write the sim
camera mounts (real/diagnostics/snapshot_cam_mount.py)."""

import numpy as np

from real.calib.extrinsics import mat_to_pos_quat, pos_quat_to_mat
from real.diagnostics.snapshot_cam_mount import mount_pose_from_T_base_cam


def test_conversion_preserves_position_and_view_direction():
    """The flip touches only the camera's y/z axes: position is unchanged and
    the OpenCV +z (view direction, into the scene) maps to MuJoCo -z."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    angle = 0.7
    quat_cv = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * v])
    T_cv = pos_quat_to_mat(rng.normal(size=3), quat_cv)
    pos, quat = mount_pose_from_T_base_cam(T_cv)
    np.testing.assert_allclose(pos, T_cv[:3, 3], atol=1e-12)
    R_mj = pos_quat_to_mat(pos, quat)[:3, :3]
    np.testing.assert_allclose(R_mj[:, 2], -T_cv[:3, :3][:, 2], atol=1e-9)
    np.testing.assert_allclose(R_mj[:, 0], T_cv[:3, :3][:, 0], atol=1e-9)


def test_mount_roundtrips_through_pos_quat():
    rng = np.random.default_rng(1)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 1.1
    quat = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    T_base_cam = pos_quat_to_mat(rng.normal(size=3), quat)
    pos, quat = mount_pose_from_T_base_cam(T_base_cam)
    T = pos_quat_to_mat(pos, quat)
    pos2, quat2 = mat_to_pos_quat(T)
    if quat2 @ quat < 0.0:
        quat2 = -quat2
    np.testing.assert_allclose(pos2, pos, atol=1e-12)
    np.testing.assert_allclose(quat2, quat, atol=1e-12)
