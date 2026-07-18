"""Shared dual-camera access for the two-C922 stereo scripts.

One place opens a rig camera *by name* — its own per-unit intrinsics, at the
focus those intrinsics were calibrated at, with the marker exposure settings —
so every stereo consumer (stereo_check, sam_track, a future dual-camera obs
source) agrees on capture configuration.
"""
import yaml

from real.vision.camera import SERIALS, device_index_for_serial, open_camera
from real.marker_spec import MARKER_EXPOSURE, MARKER_GAIN
from real.vision.pose import intrinsics_path, load_intrinsics

CAMERA_NAMES = tuple(SERIALS)


def open_rig_camera(name):
    """Open one unit at its calibrated focus: (cap, camera_matrix, dist_coeffs)."""
    path = intrinsics_path(name)
    with open(path) as f:
        focus = int(yaml.safe_load(f)["focus_absolute"])
    camera_matrix, dist_coeffs = load_intrinsics(path)
    cap = open_camera(device=device_index_for_serial(SERIALS[name]), focus=focus,
                      exposure=MARKER_EXPOSURE, gain=MARKER_GAIN)
    return cap, camera_matrix, dist_coeffs
