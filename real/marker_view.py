"""Live marker viewer: camera feed with every detected tag outlined and labelled
with its ID (and its planned role, if known). Detection only — no pose yet; this
is the sanity check that the camera actually sees the printed markers.

Works for either printed family, selected with --family (default aruco):
    conda run -n mujoco_env python -m real.marker_view                  # ArUco (OpenCV)
    conda run -n mujoco_env python -m real.marker_view --family apriltag # AprilTag (pupil-apriltags)

ArUco uses OpenCV; AprilTag uses the native pupil-apriltags detector (better
motion-blur robustness). Both go through real.detect so this file is backend-agnostic.

Uses the pinned calibration focus so the feed matches the intrinsics, and the
shared marker spec so the dictionary always matches what make_markers printed.
"""
import argparse

import cv2
import numpy as np

from real.camera import open_camera, HEIGHT
from real.marker_spec import FAMILIES, ROLES
from real.calibrate_camera import FOCUS_ABSOLUTE
from real.detect import make_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=list(FAMILIES), default="aruco")
    args = parser.parse_args()

    cap = open_camera(focus=FOCUS_ABSOLUTE)
    detector = make_detector(args.family)
    print(f"{args.family} viewer — q/ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets = detector.detect(gray)
        for d in dets:
            quad = d.corners.astype(np.int32)
            cv2.polylines(frame, [quad], True, (0, 255, 0), 2)
            cv2.circle(frame, tuple(quad[0]), 4, (255, 0, 0), -1)   # corner 0 marks orientation
            center = d.corners.mean(axis=0).astype(int)
            label = f"{d.id}" + (f" {ROLES[d.id]}" if d.id in ROLES else "")
            cv2.putText(frame, label, (center[0] - 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        seen = sorted(d.id for d in dets)
        cv2.putText(frame, f"{args.family}  detected {len(seen)}: {seen}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "q/ESC quit", (12, HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f"{args.family} view", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
