"""The table anchor is a bolted-down camera's static pose, not a per-frame
measurement: losing the tag to an occluding sponge or arm must not stale every
arm marker at once (real/rollout/marker_obs.CameraMarkerSource._poses_to_base).
"""

import time

import numpy as np
import pytest

from real.calib.extrinsics import PoseEMA, pos_quat_to_mat
from real.marker_spec import ARM_TAG_TO_SITE
from real.rollout.marker_obs import (
    CameraMarkerSource,
    StereoCameraMarkerSource,
    fuse_marker_views,
)

ARM_TAGS = sorted(ARM_TAG_TO_SITE)


def _source():
    """A CameraMarkerSource with no camera behind it — _poses_to_base is pure
    given the extrinsics loaded in __init__, so the feed is never touched."""
    return CameraMarkerSource(feed=None, family="apriltag")


def _rvec_tvec(tvec):
    return np.zeros(3), np.asarray(tvec, dtype=np.float64)


def test_arm_tags_map_through_a_held_camera_pose():
    src = _source()
    poses = {ARM_TAGS[0]: _rvec_tvec([0.02, 0.0, 0.5])}
    T_base_cam = np.eye(4)
    pos_seen, _, detected_seen = src._poses_to_base(poses, T_base_cam)
    assert detected_seen[0], "arm tag must resolve through the accepted camera pose"

    # The anchor tracker holds this transform when either table tag is hidden.
    pos_held, _, detected_held = src._poses_to_base(poses, T_base_cam)
    assert detected_held[0], "an incomplete anchor pair must not stale arm tags"
    np.testing.assert_allclose(pos_held[0], pos_seen[0], atol=1e-12)


def test_nothing_resolves_before_the_first_anchor():
    """With no anchor ever seen there is nothing to map through — the tags
    stay undetected rather than being mapped through a bogus identity."""
    src = _source()
    pos, rot, detected = src._poses_to_base(
        {ARM_TAGS[0]: _rvec_tvec([0.02, 0.0, 0.5])}, None)
    assert not detected.any()
    assert not pos.any() and not rot.any()


def test_held_anchor_tracks_the_smoothed_value_not_the_raw_last_frame():
    """Coasting must serve the EMA state, so a jittery final observation
    before the occlusion does not jump the held anchor."""
    ema = PoseEMA(0.05)
    first = pos_quat_to_mat(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
    ema.update(first)
    jittered = pos_quat_to_mat(np.array([0.2, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
    smoothed = ema.update(jittered)
    np.testing.assert_allclose(ema.value(), smoothed, atol=1e-12)
    # 5% of the 0.1 m jump, not the raw sample.
    assert ema.value()[0, 3] == pytest.approx(0.105)


def test_pose_ema_value_before_any_sample_fails_loud():
    with pytest.raises(AssertionError):
        PoseEMA(0.05).value()


def test_stereo_fusion_averages_positions_seen_by_both_cameras():
    positions = np.array([
        [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
        [[0.12, 0.18, 0.32], [0.70, 0.80, 0.90]],
    ])
    rotations = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
    ])
    detected = np.array([[True, True], [True, False]])

    pos, rot, seen, capture_t = fuse_marker_views(
        positions, rotations, detected, np.array([10.0, 10.01]))

    np.testing.assert_allclose(pos[0], [0.11, 0.19, 0.31])
    np.testing.assert_allclose(pos[1], positions[0, 1])
    np.testing.assert_array_equal(rot, rotations[0])
    np.testing.assert_array_equal(seen, [True, True])
    np.testing.assert_allclose(capture_t, [10.005, 10.0])


def test_stereo_fusion_uses_aux_when_main_does_not_see_tag():
    positions = np.array([
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    ])
    rotations = positions + 0.5
    detected = np.array([[False, False], [True, False]])

    pos, rot, seen, capture_t = fuse_marker_views(
        positions, rotations, detected, np.array([20.0, 20.02]))

    np.testing.assert_array_equal(pos[0], positions[1, 0])
    np.testing.assert_array_equal(rot[0], rotations[1, 0])
    np.testing.assert_array_equal(seen, [True, False])
    assert capture_t[0] == pytest.approx(20.02)
    assert capture_t[1] == -np.inf


def test_stereo_source_holds_last_fused_pose_when_both_views_lose_tag():
    now = time.monotonic()

    class FrameSource:
        def __init__(self, frame):
            self.frame = frame

        def latest_marker_frame(self):
            return self.frame

    main = FrameSource((
        np.array([[0.10, 0.20, 0.30], [0.0, 0.0, 0.0]]),
        np.zeros((2, 3)), np.array([True, False]), now - 0.02))
    aux = FrameSource((
        np.array([[0.12, 0.18, 0.32], [0.0, 0.0, 0.0]]),
        np.zeros((2, 3)), np.array([True, False]), now - 0.01))
    source = object.__new__(StereoCameraMarkerSource)
    source._sources = {"main": main, "aux": aux}
    source._pos = np.zeros((2, 3))
    source._rot = np.zeros((2, 3))
    source._last_capture_t = np.full(2, -np.inf)

    pos, _, age, visible = source.marker_observation()
    np.testing.assert_allclose(pos[0], [0.11, 0.19, 0.31])
    np.testing.assert_array_equal(visible, [True, False])
    assert 0.01 <= age[0] < 0.1

    main.frame = (np.zeros((2, 3)), np.zeros((2, 3)),
                  np.array([False, False]), now)
    aux.frame = (np.zeros((2, 3)), np.zeros((2, 3)),
                 np.array([False, False]), now)
    held, _, held_age, visible = source.marker_observation()
    np.testing.assert_allclose(held[0], pos[0])
    np.testing.assert_array_equal(visible, [False, False])
    assert held_age[0] >= age[0]
