"""Pin the OpenCV -> MuJoCo camera-frame conversion used to write the sim
camera mounts (real/diagnostics/snapshot_cam_mount.py).

The main camera's pose exists in both conventions: extrinsics.yaml stores the
calibrated `t_base_cam_fixed` (OpenCV frame, +z into the scene) and
so101.xml's `tag_cam_mount` carries the physical pose in MuJoCo's convention
(looks down -z, +y up). The two solves differ by calibration drift
(centimeters at most — the XML comes from live table-tag anchoring, the yaml
from the calibrate_qpos-era solve), while a broken conversion would be off by
a large rotation — so a loose comparison still catches any silent flip."""

import numpy as np

from real.calib.extrinsics import load_extrinsics, mat_to_pos_quat, pos_quat_to_mat
from real.diagnostics.snapshot_cam_mount import (
    current_xml_mount,
    mount_pose_from_T_base_cam,
)


def test_main_mount_matches_extrinsics_conversion_up_to_drift():
    _, T_base_cam, _, _ = load_extrinsics()
    pos, quat = mount_pose_from_T_base_cam(T_base_cam)
    xml_pos, xml_quat = current_xml_mount("main")
    if xml_quat @ quat < 0.0:
        xml_quat = -xml_quat   # quaternion double cover
    # Same physical mount through two solves: centimeter/two-degree agreement
    # proves the axes convention; any flip would miss by ~90-180 degrees.
    np.testing.assert_allclose(pos, xml_pos, atol=0.05)
    np.testing.assert_allclose(quat, xml_quat, atol=0.05)


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
    _, T_base_cam, _, _ = load_extrinsics()
    pos, quat = mount_pose_from_T_base_cam(T_base_cam)
    T = pos_quat_to_mat(pos, quat)
    pos2, quat2 = mat_to_pos_quat(T)
    if quat2 @ quat < 0.0:
        quat2 = -quat2
    np.testing.assert_allclose(pos2, pos, atol=1e-12)
    np.testing.assert_allclose(quat2, quat, atol=1e-12)
