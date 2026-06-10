"""Per-detection drawing shared by the native marker viewer and the web panel.

One implementation of "outline the tag, label it, draw its pose" so the camera
page in the panel and `real.marker_view` can never drift apart in what they
show. Draws in place on a BGR frame.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real.marker_spec import ROLES, TAG_SIZE_MM


def annotate_detections(frame, dets, estimator):
    """Outline each detection, label id (+role), and for tags with a registered
    physical size draw the pose axes and camera-frame position/rotation text."""
    for d in dets:
        quad = d.corners.astype(np.int32)
        cv2.polylines(frame, [quad], True, (0, 255, 0), 2)
        cv2.circle(frame, tuple(quad[0]), 4, (255, 0, 0), -1)   # corner 0 (TL) marks orientation
        cx, cy = d.corners.mean(axis=0).astype(int)
        label = f"{d.id}" + (f" {ROLES[d.id]}" if d.id in ROLES else "")
        cv2.putText(frame, label, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if d.id in TAG_SIZE_MM:
            rvec, tvec = estimator.estimate(d)
            axis_len = TAG_SIZE_MM[d.id] / 1000.0 / 2.0
            cv2.drawFrameAxes(frame, estimator.camera_matrix, estimator.dist_coeffs,
                              rvec, tvec, axis_len, 2)
            roll, pitch, yaw = Rotation.from_rotvec(rvec).as_euler("xyz", degrees=True)
            x, y, z = tvec * 100.0
            cv2.putText(frame, f"{x:+.1f} {y:+.1f} {z:+.1f} cm", (cx - 10, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"rpy {roll:+.0f} {pitch:+.0f} {yaw:+.0f}", (cx - 10, cy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "no size", (cx - 10, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
