"""Contract tests for real/calib/extrinsics.py — the camera↔base transform algebra.

No hardware: we synthesise a camera pose and a known base-frame tag, push the tag
*through* the camera (the inverse of what the rig does), then assert the pipeline
recovers exactly what we put in. If the convention or any compose/invert is
wrong, the recovered pose diverges and the test fails.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import (
    average_transforms,
    base_cam_from_table,
    load_extrinsics,
    mat_inv,
    mat_to_pos_quat,
    mat_to_pos_rotvec,
    mat_to_rt,
    pos_quat_to_mat,
    quarter_turn_mat,
    rigid_register,
    rt_to_mat,
    save_extrinsics,
    snap_inplane_offset,
    transform_spread,
)


def tag_pose_in_base(T_base_cam, rvec, tvec):
    """The rollout's per-tag mapping (matrix form), recreated for the tests."""
    return mat_to_pos_rotvec(T_base_cam @ rt_to_mat(rvec, tvec))


def make_T(rotvec, pos):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def test_rt_mat_round_trip():
    rvec = np.array([0.3, -0.7, 1.1])
    tvec = np.array([0.05, -0.12, 0.8])
    rvec2, tvec2 = mat_to_rt(rt_to_mat(rvec, tvec))
    np.testing.assert_allclose(rvec2, rvec, atol=1e-9)
    np.testing.assert_allclose(tvec2, tvec, atol=1e-9)


def test_mat_inv_matches_numpy():
    T = make_T([0.4, 0.2, -0.9], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(mat_inv(T), np.linalg.inv(T), atol=1e-12)
    np.testing.assert_allclose(mat_inv(T) @ T, np.eye(4), atol=1e-12)


def test_pos_quat_mat_round_trip():
    T = make_T([0.2, -1.3, 0.5], [0.3, -0.1, 0.6])
    pos, quat = mat_to_pos_quat(T)
    np.testing.assert_allclose(pos_quat_to_mat(pos, quat), T, atol=1e-12)


def test_tag_pose_in_base_recovers_known_tag():
    # The camera sits somewhere in the base frame; a tag sits somewhere else.
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    T_base_tag = make_T([1.2, 0.3, -0.4], [0.05, 0.10, 0.08])
    # What the camera measures: the tag expressed in the camera frame.
    rvec, tvec = mat_to_rt(mat_inv(T_base_cam) @ T_base_tag)

    pos, rotvec = tag_pose_in_base(T_base_cam, rvec, tvec)
    exp_pos, exp_rotvec = mat_to_pos_rotvec(T_base_tag)
    np.testing.assert_allclose(pos, exp_pos, atol=1e-9)
    np.testing.assert_allclose(rotvec, exp_rotvec, atol=1e-9)


def test_base_cam_from_table_recovers_camera():
    # Calibrated table-tag pose in base, and the (unknown-at-runtime) camera pose.
    T_base_table = make_T([0.0, 0.0, 1.4], [0.25, 0.05, 0.0])
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    # The camera measures the table tag in its own frame.
    rvec, tvec = mat_to_rt(mat_inv(T_base_cam) @ T_base_table)

    np.testing.assert_allclose(
        base_cam_from_table(T_base_table, rvec, tvec), T_base_cam, atol=1e-9)


def test_full_chain_table_anchored_arm_tag():
    # End-to-end: recover the camera from the table tag, then map an arm tag —
    # exactly the per-frame rollout path. Must return the arm tag's true base pose.
    T_base_table = make_T([0.0, 0.0, 1.4], [0.25, 0.05, 0.0])
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    T_base_armtag = make_T([0.7, -0.2, 0.9], [-0.03, 0.15, 0.12])

    r_tab, t_tab = mat_to_rt(mat_inv(T_base_cam) @ T_base_table)
    r_arm, t_arm = mat_to_rt(mat_inv(T_base_cam) @ T_base_armtag)

    cam = base_cam_from_table(T_base_table, r_tab, t_tab)
    pos, rotvec = tag_pose_in_base(cam, r_arm, t_arm)
    exp_pos, exp_rotvec = mat_to_pos_rotvec(T_base_armtag)
    np.testing.assert_allclose(pos, exp_pos, atol=1e-9)
    np.testing.assert_allclose(rotvec, exp_rotvec, atol=1e-9)


def test_average_transforms_identity_of_repeats():
    T = make_T([0.2, 0.3, -0.5], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(average_transforms([T, T, T]), T, atol=1e-12)


def test_average_transforms_median_rejects_outlier():
    good = make_T([0.0, 0.0, 0.0], [0.10, 0.20, 0.30])
    outlier = make_T([0.0, 0.0, 0.0], [0.10, 0.20, 5.0])  # wild z on one sample
    mean = average_transforms([good, good, good, outlier])
    # Median translation ignores the lone outlier entirely.
    np.testing.assert_allclose(mean[:3, 3], good[:3, 3], atol=1e-9)


def test_transform_spread_zero_when_identical():
    T = make_T([0.2, 0.3, -0.5], [0.1, 0.2, 0.3])
    trans_mm, rot_deg = transform_spread([T, T], T)
    np.testing.assert_allclose(trans_mm, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(rot_deg, [0.0, 0.0], atol=1e-9)


def test_transform_spread_units():
    T_ref = make_T([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    T = make_T([np.radians(2.0), 0.0, 0.0], [0.003, 0.0, 0.0])  # 3 mm, 2 deg
    trans_mm, rot_deg = transform_spread([T], T_ref)
    np.testing.assert_allclose(trans_mm, [3.0], atol=1e-6)
    np.testing.assert_allclose(rot_deg, [2.0], atol=1e-6)


def test_rigid_register_recovers_transform():
    rng = np.random.default_rng(0)
    T = make_T([0.3, -0.7, 0.4], [0.1, -0.2, 0.5])
    src = rng.normal(size=(12, 3))
    dst = src @ T[:3, :3].T + T[:3, 3]
    T_fit, rms = rigid_register(src, dst)
    np.testing.assert_allclose(T_fit, T, atol=1e-9)
    assert rms < 1e-9


def test_rigid_register_rms_reports_noise():
    rng = np.random.default_rng(1)
    T = make_T([0.1, 0.2, -0.3], [0.0, 0.1, 0.2])
    src = rng.normal(size=(200, 3))
    noise = rng.normal(scale=0.003, size=(200, 3))   # 3 mm
    dst = src @ T[:3, :3].T + T[:3, 3] + noise
    _, rms = rigid_register(src, dst)
    np.testing.assert_allclose(rms, 0.003 * np.sqrt(3), rtol=0.15)


def test_quarter_turn_mat_periodicity():
    np.testing.assert_allclose(quarter_turn_mat(0), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(quarter_turn_mat(4), np.eye(4), atol=1e-12)
    # -k undoes +k.
    np.testing.assert_allclose(quarter_turn_mat(1) @ quarter_turn_mat(-1), np.eye(4), atol=1e-12)
    # 90° about z maps +x -> +y.
    np.testing.assert_allclose(quarter_turn_mat(1)[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_snap_inplane_offset_recovers_each_quarter_turn():
    R_intended = Rotation.from_rotvec([0.3, -0.8, 0.5]).as_matrix()
    for true_k in range(4):
        R_meas = R_intended @ quarter_turn_mat(true_k)[:3, :3]
        k, residual = snap_inplane_offset(R_intended, R_meas)
        assert k == true_k
        assert residual < 1e-6


def test_snap_inplane_offset_reports_residual_when_askew():
    R_intended = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
    # 90° in-plane plus a 12° tilt that no quarter turn can absorb.
    askew = quarter_turn_mat(1)[:3, :3] @ Rotation.from_rotvec([np.radians(12), 0, 0]).as_matrix()
    k, residual = snap_inplane_offset(R_intended, R_intended @ askew)
    assert k == 1
    np.testing.assert_allclose(residual, 12.0, atol=1e-6)


def test_save_load_round_trip(tmp_path):
    T_base_table = make_T([0.0, 0.0, 1.4], [0.25, 0.05, 0.0])
    T_base_table_11 = make_T([0.0, 0.0, -0.2], [0.40, 0.05, 0.0])
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    path = tmp_path / "extrinsics.yaml"
    save_extrinsics(path, {10: T_base_table, 11: T_base_table_11},
                    T_base_cam, focus_absolute=30,
                    n_samples=9, spread_mm=2.1, spread_deg=0.7,
                    quarter_turns={0: 1, 2: 3})
    anchors2, cam2, focus, quarter_turns = load_extrinsics(path)
    np.testing.assert_allclose(anchors2[10], T_base_table, atol=1e-9)
    np.testing.assert_allclose(anchors2[11], T_base_table_11, atol=1e-9)
    np.testing.assert_allclose(cam2, T_base_cam, atol=1e-9)
    assert focus == 30
    assert quarter_turns == {0: 1, 2: 3}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
