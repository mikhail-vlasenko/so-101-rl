"""Pin the OpenCV -> MuJoCo camera-frame conversion used to write the sim
camera mounts (real/diagnostics/snapshot_cam_mount.py)."""

import numpy as np
import pytest

from real.calib.extrinsics import mat_to_pos_quat, pos_quat_to_mat
from real.diagnostics.snapshot_cam_mount import (
    MountSnapshot,
    mount_pose_from_T_base_cam,
    update_scene_mounts,
)


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


def _snapshot(camera, pos, quat):
    return MountSnapshot(
        camera=camera,
        accepted_frames=100,
        requested_frames=100,
        translation_spread_mean_mm=0.1,
        rotation_spread_mean_deg=0.02,
        pos=np.asarray(pos, dtype=np.float64),
        quat=np.asarray(quat, dtype=np.float64),
    )


def test_scene_mount_update_replaces_both_poses_and_preserves_other_xml(tmp_path):
    path = tmp_path / "so101.xml"
    original = """<mujoco>
  <worldbody>
    <body name="tag_cam_mount" pos="1 2 3"
          quat="1 0 0 0">
      <camera name="tag_cam"/>
    </body>
    <body name="untouched" pos="4 5 6"/>
    <body name="tag_cam_aux_mount" pos="7 8 9"
          quat="1 0 0 0">
      <camera name="tag_cam_aux"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(original)
    snapshots = {
        "main": _snapshot("main", [0.1, -0.2, 0.3], [0.9, 0.1, 0.2, 0.3]),
        "aux": _snapshot("aux", [0.4, -0.5, 0.6], [0.8, 0.2, 0.3, 0.4]),
    }

    update_scene_mounts(snapshots, path)

    updated = path.read_text()
    assert 'name="tag_cam_mount" pos="0.100000 -0.200000 0.300000"' in updated
    assert 'quat="0.900000 0.100000 0.200000 0.300000"' in updated
    assert 'name="tag_cam_aux_mount" pos="0.400000 -0.500000 0.600000"' in updated
    assert 'quat="0.800000 0.200000 0.300000 0.400000"' in updated
    assert '<body name="untouched" pos="4 5 6"/>' in updated


def test_scene_mount_update_is_atomic_when_a_target_is_missing(tmp_path):
    path = tmp_path / "so101.xml"
    original = """<mujoco>
  <worldbody>
    <body name="tag_cam_mount" pos="1 2 3"
          quat="1 0 0 0">
      <camera name="tag_cam"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(original)
    snapshots = {
        "main": _snapshot("main", [0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]),
        "aux": _snapshot("aux", [0.4, 0.5, 0.6], [1.0, 0.0, 0.0, 0.0]),
    }

    with pytest.raises(RuntimeError, match="tag_cam_aux_mount.*found 0"):
        update_scene_mounts(snapshots, path)

    assert path.read_text() == original
