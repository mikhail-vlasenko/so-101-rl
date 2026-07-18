"""Pose pipeline contract:
  1. PoseEstimator recovers a known camera-frame pose from synthetically projected
     tag corners (translation in m, rotation as a Rodrigues vector).
  2. Both detection backends emit corners in the canonical TL, TR, BR, BL order
     that the estimator's object points assume — the link that makes one solver
     work for both families.
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import pytest

from real.vision.detect import make_detector
from real.marker_spec import FAMILIES, APRILTAG_FAMILY
from real.vision.pose import PoseEstimator, _object_points


def _camera_matrix():
    return np.array([[969.0, 0.0, 648.0],
                     [0.0, 968.0, 335.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class _Det:
    def __init__(self, id, corners):
        self.id = id
        self.corners = corners


@pytest.mark.parametrize("rvec_true, tvec_true", [
    (np.array([0.0, 0.0, 0.0]), np.array([0.00, 0.00, 0.40])),       # frontal
    (np.array([0.3, -0.2, 0.1]), np.array([0.05, -0.03, 0.35])),     # tilted, off-axis
])
def test_pose_roundtrip(rvec_true, tvec_true):
    cam = _camera_matrix()
    dist = np.zeros(5)
    size_mm = 40.0
    obj = _object_points(size_mm / 1000.0)
    img_pts, _ = cv2.projectPoints(obj, rvec_true, tvec_true, cam, dist)
    det = _Det(10, img_pts.reshape(4, 2).astype(np.float32))

    est = PoseEstimator(cam, dist, tag_size_mm={10: size_mm})
    rvec, tvec = est.estimate(det)

    np.testing.assert_allclose(tvec, tvec_true, atol=1e-4)
    # geodesic angle between recovered and true rotation
    r_err = cv2.Rodrigues(rvec)[0].T @ cv2.Rodrigues(rvec_true)[0]
    angle = np.arccos(np.clip((np.trace(r_err) - 1) / 2, -1, 1))
    assert angle < 1e-3


def test_unregistered_id_fails_loud():
    est = PoseEstimator(_camera_matrix(), np.zeros(5), tag_size_mm={10: 40.0})
    with pytest.raises(KeyError):
        est.estimate(_Det(99, np.zeros((4, 2), np.float32)))


@pytest.mark.parametrize("family", ["aruco", "apriltag"])
def test_backend_corner_order_is_canonical(family):
    """Render an axis-aligned tag and confirm detected corners are TL, TR, BR, BL."""
    dictionary = aruco.getPredefinedDictionary(FAMILIES[family])
    img = aruco.generateImageMarker(dictionary, 0, 300)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

    dets = make_detector(family).detect(img)
    assert len(dets) == 1
    c = dets[0].corners
    mid = 250.0  # image center (300 + 2*100) / 2
    sides = [("T", "L"), ("T", "R"), ("B", "R"), ("B", "L")]
    for (x, y), (vert, horiz) in zip(c, sides):
        assert (("T" if y < mid else "B") == vert)
        assert (("L" if x < mid else "R") == horiz)
