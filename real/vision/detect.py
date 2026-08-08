"""Marker detection backends normalised to a common Detection result.

ArUco is detected with OpenCV. AprilTag (tag36h11) is detected with the native
pupil-apriltags detector, which is markedly more robust to motion blur and small
apparent size than OpenCV's ArUco pipeline decoding the same dictionary — worth
it for the moving wrist tag. Both backends take a grayscale frame and return a
list of `Detection`, so callers (viewer, pose) don't care which family is active.
"""
from dataclasses import dataclass

import cv2.aruco as aruco
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector

from real.marker_spec import FAMILIES, APRILTAG_FAMILY


@dataclass
class Detection:
    id: int
    corners: np.ndarray   # (4, 2) float32 image points, canonical order TL, TR, BR, BL


# pupil-apriltags returns corners as TR, TL, BL, BR (verified by detecting a
# rendered tag); reorder to the OpenCV-ArUco canonical TL, TR, BR, BL so both
# families feed real.vision.pose.PoseEstimator the same corner order.
_APRILTAG_TO_CANONICAL = [1, 0, 3, 2]


class ArucoBackend:
    def __init__(self):
        self._det = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(FAMILIES["aruco"]),
            aruco.DetectorParameters(),
        )

    def detect(self, gray):
        corners, ids, _ = self._det.detectMarkers(gray)
        if ids is None:
            return []
        return [Detection(int(i), c.reshape(4, 2).astype(np.float32))
                for c, i in zip(corners, ids.flatten())]

    def close(self):
        pass


class _SafeAprilTagDetector(AprilTagDetector):
    """pupil-apriltags detector with the native objects destroyed in dependency order.

    pupil-apriltags 1.0.4.post11 destroys the tag family before the detector.
    The detector's destructor then clears its family pointers, dereferencing the
    already-freed family; after a long capture this has produced a repeatable
    segfault in ``quick_decode_uninit``. The AprilTag C API requires the detector
    to go first, followed by the family.
    """

    def close(self):
        if self.tag_detector_ptr is None:
            return
        detector_destroy = self.libc.apriltag_detector_destroy
        detector_destroy.restype = None
        detector_destroy(self.tag_detector_ptr)
        family_destroy = self.libc.tag36h11_destroy
        family_destroy.restype = None
        family_destroy(self.tag_families[APRILTAG_FAMILY])
        self.tag_detector_ptr = None
        self.tag_families.clear()

    def __del__(self):
        self.close()


class AprilTagBackend:
    def __init__(self):
        self._det = _SafeAprilTagDetector(families=APRILTAG_FAMILY)

    def detect(self, gray):
        return [Detection(d.tag_id, d.corners[_APRILTAG_TO_CANONICAL].astype(np.float32))
                for d in self._det.detect(gray)]

    def close(self):
        self._det.close()


def make_detector(family):
    if family == "aruco":
        return ArucoBackend()
    if family == "apriltag":
        return AprilTagBackend()
    raise ValueError(f"unknown family: {family}")
