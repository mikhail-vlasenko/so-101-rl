"""Single source of truth for the printed fiducial markers.

Shared by `scripts/make_markers.py` (generation) and the `real/` detection +
pose code, so the dictionary and per-id physical sizes can never drift apart —
a detector reading the wrong dictionary or wrong size silently fails or returns
garbage pose.

ArUco id N and AprilTag id N are *different* markers (different dictionaries)
even at the same N.
"""
import cv2.aruco as aruco

# Family name -> OpenCV predefined dictionary id. Used for *generation* (both
# families) and for ArUco *detection*. AprilTag detection uses the native
# pupil-apriltags detector instead (see APRILTAG_FAMILY) for blur robustness.
FAMILIES = {
    "aruco": aruco.DICT_5X5_50,
    "apriltag": aruco.DICT_APRILTAG_36h11,
}

# Same physical AprilTag family as FAMILIES["apriltag"], named the way
# pupil-apriltags expects it. Kept beside FAMILIES so the two can't drift.
APRILTAG_FAMILY = "tag36h11"

# Printed black-square edge length per tag id, millimetres (same id->size across
# families). Large tags for far-range accuracy, small tags for the end effector.
TAG_SIZE_MM = {
    0: 20.0, 1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0,
    10: 40.0, 11: 40.0, 12: 40.0,
}

# Intended physical role per id (from calibration_plan.md). Detection doesn't
# need this; it documents the planned layout in one place. Ids 3/4 are spares.
ROLES = {
    0: "wrist", 1: "finger", 2: "base",
    10: "table", 11: "table", 12: "table",
}
