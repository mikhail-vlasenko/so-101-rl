"""Reusable marker annotations and stereo status views for real-camera tools.

`annotate_tags` is the configurable base: callers choose per-tag labels/colors.
`annotate_detections` adds metric pose axes/text for the marker viewer and panel.
`compose_stereo_view` / `StereoViewer` standardize the two-camera layout and
status bands while leaving script-specific status content with the caller.

All annotation functions draw on BGR frames. The compose function copies its
inputs, so dataset recorders can save raw frames before producing a preview.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real.marker_spec import ROLES, TAG_SIZE_MM
from real.vision.camera import HEIGHT, WIDTH
from real.vision.stereo_rig import CAMERA_NAMES

GREEN = (0, 220, 0)
YELLOW = (0, 220, 220)
RED = (0, 0, 255)
WHITE = (235, 235, 235)
TABLE_BLUE = (255, 180, 0)


@dataclass(frozen=True)
class TagStyle:
    label: str
    color: tuple[int, int, int] = GREEN
    thickness: int = 2


@dataclass(frozen=True)
class OverlayLine:
    text: str
    color: tuple[int, int, int] = WHITE


def annotate_tags(frame, dets, styles=None):
    """Outline tags in place with optional per-id ``TagStyle`` overrides."""
    styles = {} if styles is None else styles
    for detection in dets:
        style = styles.get(
            detection.id,
            TagStyle(f"{detection.id}" + (
                f" {ROLES[detection.id]}" if detection.id in ROLES else "")))
        quad = detection.corners.astype(np.int32)
        cv2.polylines(frame, [quad], True, style.color, style.thickness)
        cv2.circle(frame, tuple(quad[0]), 4, (255, 0, 0), -1)
        x, y = quad[0]
        cv2.putText(frame, style.label, (int(x), max(24, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, style.color, 2)
    return frame


def annotate_detections(frame, dets, estimator):
    """Outline each detection, label id (+role), and for tags with a registered
    physical size draw the pose axes and camera-frame position/rotation text."""
    annotate_tags(frame, dets)
    for d in dets:
        cx, cy = d.corners.mean(axis=0).astype(int)
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
    return frame


def _draw_lines(frame, lines, y0=30):
    for index, line in enumerate(lines):
        cv2.putText(frame, line.text, (12, y0 + 28 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line.color, 2)


def compose_stereo_view(frames, camera_lines=None, header_lines=(),
                        camera_names=CAMERA_NAMES,
                        view_size=(WIDTH // 2, HEIGHT // 2)):
    """Return a standard side-by-side camera canvas with status text."""
    assert set(frames) == set(camera_names), (set(frames), set(camera_names))
    camera_lines = {} if camera_lines is None else camera_lines
    lines_by_camera = {camera: [] for camera in camera_names}
    for camera, lines in camera_lines.items():
        assert camera in lines_by_camera, camera
        lines_by_camera[camera] = list(lines)
    views = []
    for camera in camera_names:
        view = frames[camera].copy()
        _draw_lines(view, lines_by_camera[camera])
        views.append(cv2.resize(view, view_size))
    stereo = np.hstack(views)
    header_height = 8 + 28 * len(header_lines)
    if header_height == 8:
        return stereo
    canvas = np.zeros((stereo.shape[0] + header_height, stereo.shape[1], 3),
                      dtype=np.uint8)
    canvas[header_height:] = stereo
    _draw_lines(canvas, header_lines, y0=26)
    return canvas


class StereoViewer:
    """Native OpenCV window for a configurable standard stereo canvas."""

    def __init__(self, title, camera_names=CAMERA_NAMES):
        self.title = str(title)
        self.camera_names = tuple(camera_names)
        self._shown = False

    def show(self, frames, camera_lines=None, header_lines=(), delay_ms=1):
        canvas = compose_stereo_view(
            frames, camera_lines, header_lines, self.camera_names)
        cv2.imshow(self.title, canvas)
        self._shown = True
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self):
        if self._shown:
            cv2.destroyWindow(self.title)
            self._shown = False
