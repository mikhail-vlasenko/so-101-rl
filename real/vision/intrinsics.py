"""Camera-name -> per-unit intrinsics-file mapping (cv2-free).

Split out of real/vision/pose.py so the sim side (src/marker_noise.py, which
builds its CameraIntrinsics view of the same YAMLs) can share the single path
mapping without importing cv2.
"""
import os


def intrinsics_path(camera):
    """Per-unit intrinsics file for a camera name from `real.vision.camera.SERIALS`.

    Intrinsics are per-lens — the two C922 units differ by ~1.4% in focal
    length — so each unit gets its own file. "main" keeps the legacy unsuffixed
    name that predates the second camera.
    """
    suffix = "" if camera == "main" else f"_{camera}"
    return os.path.join(os.path.dirname(__file__), f"camera_intrinsics{suffix}.yaml")
