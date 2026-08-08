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
    5: 20.0, 6: 20.0, 7: 20.0,
    10: 40.0, 11: 40.0, 12: 40.0,
}

# Physical role per id, as glued on the rig. Detection keys tags by id and the
# rollout maps id->obs slot, so this is the single source of truth for the
# layout — keep it matching the arm. The finger/wrist sites in so101.xml carry
# the same ids in their comments. IDs 10 and 11 form the active table-anchor
# board; id 12 is a spare and is deliberately not trusted by the anchor solver.
#
# The sponge's tags (id 1 on a largest face, plus the cube_eval set on its
# other faces) are EVAL-ONLY: they provide ground truth for the shape-tracking
# dataset (real/tracking/record_shapes.py -> eval_estimator.py) and for
# rollout evaluation — never the policy obs path, which is tag-free (SAM
# stereo, real/rollout/object_obs.py). The eval set is glued for a recording
# session and peeled afterwards; its in-plane placement is solved by
# real/tracking/tag_body_calib.py into sponge_tags.yaml.
ROLES = {
    0: "finger", 1: "cube", 2: "wrist",
    3: "cube_eval", 4: "cube_eval", 5: "cube_eval",
    6: "cube_eval", 7: "cube_eval",
    10: "table", 11: "table", 12: "table",
}

# Tag id -> the so101.xml site modelling the same physical tag, named exactly as
# in src.base_env.MARKER_SITE_NAMES. The calibrator uses each site's FK pose as
# the base-frame anchor for its tag; the rollout writes a measured tag into the
# observation slot of its site. Keep the ids in sync with ROLES.
ARM_TAG_TO_SITE = {0: "marker_finger", 2: "marker_wrist"}
TABLE_TAG_IDS = (10, 11)
CUBE_TAG_ID = 1
# Every tag glued to the sponge (id 1 + the eval-only faces), the set the
# dataset GT pipeline detects and tag_body_calib solves.
SPONGE_TAG_IDS = tuple(sorted(
    tag for tag, role in ROLES.items() if role in ("cube", "cube_eval")))

# Camera capture settings tuned for marker detection on this rig (shared by the
# viewer, the pose step, and deployment). Manual exposure kills motion blur.
# 100 = 10 ms, which also spans one full 50 Hz-mains flicker cycle so the lighting
# banding cancels; shorter exposures showed rolling horizontal bands. Gain is
# maxed to keep that short exposure bright enough.
MARKER_EXPOSURE = 100        # exposure_time_absolute, 100 us units (-> 10 ms)
MARKER_GAIN = 255            # sensor gain 0-255
