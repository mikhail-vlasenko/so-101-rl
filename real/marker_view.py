"""Live marker viewer: camera feed with every detected tag outlined, labelled with
its ID (and planned role), and — for tags whose physical size is registered in
marker_spec — its 3-D pose in the camera frame: position (cm) and rotation (roll/
pitch/yaw, deg), with the tag's axes drawn (x red, y green, z out of the face blue).

Works for either printed family, selected with --family (default aruco):
    conda run -n mujoco_env python -m real.marker_view                  # ArUco (OpenCV)
    conda run -n mujoco_env python -m real.marker_view --family apriltag # AprilTag (pupil-apriltags)

ArUco uses OpenCV; AprilTag uses the native pupil-apriltags detector (better
motion-blur robustness). Both go through real.detect so this file is backend-agnostic.

Tune exposure/gain live to fight motion blur: shorten exposure until tags survive
motion, then raise gain to recover brightness. Pass a known-good value with
--exposure to start there (and reuse it on the rig). Keys:
    [ / ]   exposure shorter / longer (switches to manual)
    - / =   gain down / up
    a       back to auto-exposure
    q/ESC   quit
"""
import argparse

import cv2
import numpy as np

from real.camera import open_camera, set_exposure, set_auto_exposure, v4l2_set, HEIGHT
from real.marker_spec import FAMILIES, MARKER_EXPOSURE, MARKER_GAIN
from real.calibrate_camera import FOCUS_ABSOLUTE
from real.detect import make_detector
from real.overlay import annotate_detections
from real.pose import load_intrinsics, PoseEstimator

EXPOSURE_MIN, EXPOSURE_MAX, EXPOSURE_STEP = 3, 2047, 10
MANUAL_START = 80              # exposure_time_absolute to drop to when leaving auto (~8 ms)
GAIN_MAX, GAIN_STEP = 255, 15


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=list(FAMILIES), default="aruco")
    parser.add_argument("--exposure", type=int, default=MARKER_EXPOSURE,
                        help="manual exposure_time_absolute (100 us units); 'a' in-window for auto")
    parser.add_argument("--gain", type=int, default=MARKER_GAIN, help="sensor gain 0-255")
    args = parser.parse_args()

    cap = open_camera(focus=FOCUS_ABSOLUTE, exposure=args.exposure, gain=args.gain)
    detector = make_detector(args.family)
    estimator = PoseEstimator(*load_intrinsics())
    exposure = args.exposure      # None == auto
    gain = args.gain
    print(f"{args.family} viewer — [ ] exposure, - = gain, a auto, q/ESC quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets = detector.detect(gray)
        annotate_detections(frame, dets, estimator)

        seen = sorted(d.id for d in dets)
        exp_str = "auto" if exposure is None else f"{exposure} (~{exposure / 10:.1f}ms)"
        cv2.putText(frame, f"{args.family}  detected {len(seen)}: {seen}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"exposure {exp_str}   gain {gain}", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "[ ] exposure   - = gain   a auto   q quit", (12, HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f"{args.family} view", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("["), ord("]")):
            base = MANUAL_START if exposure is None else exposure
            delta = -EXPOSURE_STEP if key == ord("[") else EXPOSURE_STEP
            exposure = int(np.clip(base + delta, EXPOSURE_MIN, EXPOSURE_MAX))
            set_exposure(exposure)
        elif key in (ord("-"), ord("="), ord("+")):
            gain = int(np.clip(gain + (-GAIN_STEP if key == ord("-") else GAIN_STEP), 0, GAIN_MAX))
            v4l2_set("gain", gain)
        elif key == ord("a"):
            set_auto_exposure()
            exposure = None

    cap.release()
    cv2.destroyAllWindows()
    if exposure is not None:
        print(f"final exposure={exposure} gain={gain}  "
              f"-> reuse with --exposure {exposure} --gain {gain}")


if __name__ == "__main__":
    main()
