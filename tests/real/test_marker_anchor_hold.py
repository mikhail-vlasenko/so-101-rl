"""The table anchor is a bolted-down camera's static pose, not a per-frame
measurement: losing the tag to an occluding sponge or arm must not stale every
arm marker at once (real/rollout/marker_obs.CameraMarkerSource._poses_to_base).
"""

import numpy as np
import pytest

from real.calib.extrinsics import PoseEMA, pos_quat_to_mat
from real.marker_spec import ARM_TAG_TO_SITE, TABLE_TAG_ID
from real.rollout.marker_obs import CameraMarkerSource

ARM_TAGS = sorted(ARM_TAG_TO_SITE)


def _source():
    """A CameraMarkerSource with no camera behind it — _poses_to_base is pure
    given the extrinsics loaded in __init__, so the feed is never touched."""
    return CameraMarkerSource(feed=None, family="apriltag")


def _rvec_tvec(tvec):
    return np.zeros(3), np.asarray(tvec, dtype=np.float64)


def test_arm_tags_survive_a_missing_table_tag():
    src = _source()
    anchored = {TABLE_TAG_ID: _rvec_tvec([0.0, 0.0, 0.6]),
                ARM_TAGS[0]: _rvec_tvec([0.02, 0.0, 0.5])}
    pos_seen, _, detected_seen = src._poses_to_base(anchored)
    assert detected_seen[0], "arm tag must resolve while the table tag is visible"

    # Same arm detection, table tag now occluded: still resolved, same place.
    pos_held, _, detected_held = src._poses_to_base({ARM_TAGS[0]: anchored[ARM_TAGS[0]]})
    assert detected_held[0], "an occluded table tag must not stale the arm tags"
    np.testing.assert_allclose(pos_held[0], pos_seen[0], atol=1e-12)


def test_nothing_resolves_before_the_first_anchor():
    """With no anchor ever seen there is nothing to map through — the tags
    stay undetected rather than being mapped through a bogus identity."""
    src = _source()
    pos, rot, detected = src._poses_to_base({ARM_TAGS[0]: _rvec_tvec([0.02, 0.0, 0.5])})
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
