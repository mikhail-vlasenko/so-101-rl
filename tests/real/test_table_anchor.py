import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import load_table_anchor_poses, mat_inv, pos_quat_to_mat
from real.calib.table_anchor import TableAnchorLimits, TableAnchorTracker
from real.marker_spec import TABLE_TAG_IDS
from real.vision.detect import Detection
from real.vision.pose import tag_object_points


K = np.array([[800.0, 0.0, 640.0],
              [0.0, 800.0, 360.0],
              [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
LIMITS = TableAnchorLimits(
    ema_alpha=0.05,
    max_reprojection_rmse_px=0.5,
    max_camera_translation_disagreement_mm=1.0,
    max_camera_rotation_disagreement_deg=0.5,
)


def _T(pos, euler_deg):
    q_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return pos_quat_to_mat(pos, np.roll(q_xyzw, 1))


ANCHORS = {
    10: _T([0.08, -0.075, 0.0], [0.0, 0.0, 92.0]),
    11: _T([0.222, -0.073, 0.0], [0.0, 0.0, 180.0]),
}
CAMERA = pos_quat_to_mat(
    [0.1088, -0.4056, 0.1849],
    [-0.6180, 0.7761, -0.1148, 0.0499],
)


def _detections(T_base_cam=CAMERA):
    T_cam_base = mat_inv(T_base_cam)
    detections = {}
    for tag in TABLE_TAG_IDS:
        corners = tag_object_points(tag)
        corners_base = corners @ ANCHORS[tag][:3, :3].T + ANCHORS[tag][:3, 3]
        corners_cam = corners_base @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
        pixels, _ = cv2.projectPoints(
            corners_cam, np.zeros(3), np.zeros(3), K, DIST)
        detections[tag] = Detection(tag, pixels.reshape(4, 2).astype(np.float32))
    return detections


def test_two_tags_seed_and_recover_camera_pose():
    tracker = TableAnchorTracker(
        K, DIST, anchor_poses=ANCHORS, limits=LIMITS)
    assert tracker.observe(_detections())
    np.testing.assert_allclose(tracker.value(), CAMERA, atol=2e-6)
    assert tracker.quality.updated
    assert tracker.quality.reprojection_rmse_px < 1e-3


def test_one_visible_tag_holds_last_ema_pose_exactly():
    tracker = TableAnchorTracker(
        K, DIST, anchor_poses=ANCHORS, limits=LIMITS)
    both = _detections()
    assert tracker.observe(both)
    before = tracker.value().copy()
    assert not tracker.observe({10: both[10]})
    np.testing.assert_array_equal(tracker.value(), before)
    assert tracker.quality.rejection == "both table tags not visible"


def test_one_visible_tag_cannot_seed_a_session():
    tracker = TableAnchorTracker(
        K, DIST, anchor_poses=ANCHORS, limits=LIMITS)
    detections = _detections()
    assert not tracker.observe({11: detections[11]})
    assert tracker.value() is None


def test_inconsistent_tag_pair_is_rejected_and_held():
    tracker = TableAnchorTracker(
        K, DIST, anchor_poses=ANCHORS, limits=LIMITS)
    good = _detections()
    assert tracker.observe(good)
    before = tracker.value().copy()
    bad = dict(good)
    bad[11] = Detection(11, good[11].corners + np.array([20.0, 0.0]))
    assert not tracker.observe(bad)
    np.testing.assert_array_equal(tracker.value(), before)
    assert tracker.quality.rejection is not None


def test_valid_pair_updates_existing_pose_through_ema():
    tracker = TableAnchorTracker(
        K, DIST, anchor_poses=ANCHORS, limits=LIMITS)
    assert tracker.observe(_detections())
    moved = CAMERA.copy()
    moved[0, 3] += 0.010
    assert tracker.observe(_detections(moved))
    np.testing.assert_allclose(
        tracker.value()[0, 3], CAMERA[0, 3] + 0.0005, atol=2e-6)


def test_deployed_anchor_board_is_exactly_the_base_table_plane():
    anchors = load_table_anchor_poses()
    assert set(anchors) == set(TABLE_TAG_IDS)
    for transform in anchors.values():
        assert transform[2, 3] == 0.0
        np.testing.assert_allclose(transform[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)
