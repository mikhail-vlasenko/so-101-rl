"""Marker pose in the camera frame from calibrated intrinsics.

Turns a tag's four image corners into its 3-D pose relative to the camera:
translation in metres and rotation as a Rodrigues vector. One solver for both
families — `cv2.solvePnP` with `SOLVEPNP_IPPE_SQUARE`, the method built for square
planar fiducials — which works because `real.detect` hands us every tag's corners
in the *same* canonical order (TL, TR, BR, BL), so the recovered pose means the
same thing no matter which detector found the tag.

Scale comes from the tag's true printed edge length (`real.marker_spec.TAG_SIZE_MM`):
get the size wrong and the reported distance scales by the same factor. Distortion
is handled (the solver takes the calibrated `dist_coeffs`).
"""
import os

import cv2
import numpy as np
import yaml

from real.marker_spec import TAG_SIZE_MM

def intrinsics_path(camera):
    """Per-unit intrinsics file for a camera name from `real.camera.SERIALS`.

    Intrinsics are per-lens — the two C922 units differ by ~1.4% in focal
    length — so each unit gets its own file. "main" keeps the legacy unsuffixed
    name that predates the second camera.
    """
    suffix = "" if camera == "main" else f"_{camera}"
    return os.path.join(os.path.dirname(__file__), f"camera_intrinsics{suffix}.yaml")


INTRINSICS_PATH = intrinsics_path("main")


def load_intrinsics(path=INTRINSICS_PATH):
    """Load the calibrated camera matrix (3x3) and distortion coeffs from YAML.

    The intrinsics are only valid at the focus they were calibrated at, so open
    the camera with that same `focus_absolute` (see the YAML) — otherwise the
    focal length differs and every pose is wrong.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def _object_points(size_m):
    """Tag corners in the tag's own frame (metres), ordered TL, TR, BR, BL to match
    `Detection.corners` and what SOLVEPNP_IPPE_SQUARE expects. The tag lies in the
    z=0 plane with +x right, +y up, +z out of the printed face toward the viewer."""
    h = size_m / 2.0
    return np.array([[-h,  h, 0.0],
                     [ h,  h, 0.0],
                     [ h, -h, 0.0],
                     [-h, -h, 0.0]], dtype=np.float64)


class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, tag_size_mm=TAG_SIZE_MM):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size_mm = tag_size_mm

    def estimate(self, detection):
        """Return (rvec (3,), tvec (3,)) of the tag in the camera frame, metres.

        Raises KeyError if the id has no registered physical size — refuse to
        guess scale rather than report a confidently-wrong distance.
        """
        size_m = self.tag_size_mm[detection.id] / 1000.0
        ok, rvec, tvec = cv2.solvePnP(
            _object_points(size_m), detection.corners.astype(np.float64),
            self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            raise RuntimeError(f"solvePnP failed for tag {detection.id}")
        return rvec.reshape(3), tvec.reshape(3)
