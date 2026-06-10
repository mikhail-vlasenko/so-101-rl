"""In-server camera capture for the panel's camera page.

Owns the C922 while streaming: a capture thread reads frames, runs the marker
detector + pose estimator, draws the same overlays as `real.marker_view`
(via `real.overlay.annotate_detections`), and publishes JPEGs into a FrameBox
that the panel serves at /camera/stream.

The service registers itself as a CAMERA holder in the Runner's resource
accounting, so starting the stream blocks native camera tools (and vice
versa) with a clear 409 instead of two processes fighting over /dev/video0.
`stop()` is synchronous: when it returns, the device is released.
"""

from __future__ import annotations

import threading

import cv2

from real.calibrate_camera import FOCUS_ABSOLUTE
from real.camera import open_camera, set_exposure, v4l2_set
from real.detect import make_detector
from real.marker_spec import MARKER_EXPOSURE, MARKER_GAIN
from real.overlay import annotate_detections
from real.pose import PoseEstimator, load_intrinsics

from panel.registry import Resource
from panel.runner import ResourceBusyError, Runner
from panel.streamer import FrameBox

HOLDER_NAME = "the panel's camera stream"
JPEG_QUALITY = 80


class CameraService:
    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._cap = None
        self.box = FrameBox()
        self.family: str | None = None
        self.exposure: int | None = None
        self.gain: int | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, family: str, exposure: int, gain: int) -> None:
        """Claim the camera and start the capture thread. Raises
        ResourceBusyError if anything (including a panel-launched native tool)
        holds the camera, and ValueError on a bad family."""
        with self._lock:
            if self.is_running():
                raise ResourceBusyError(f"camera is in use by {HOLDER_NAME}")
            detector = make_detector(family)  # validates family before claiming
            self._runner.claim_external(HOLDER_NAME, {Resource.CAMERA})
            opened = False
            try:
                self._cap = open_camera(focus=FOCUS_ABSOLUTE,
                                        exposure=exposure, gain=gain)
                opened = True
            finally:
                if not opened:
                    self._runner.release_external(HOLDER_NAME)
            self.family = family
            self.exposure = exposure
            self.gain = gain
            estimator = PoseEstimator(*load_intrinsics())
            self._stop_flag.clear()
            self._thread = threading.Thread(
                target=self._capture_loop, args=(detector, estimator),
                name="panel-camera", daemon=True)
            self._thread.start()

    def _capture_loop(self, detector, estimator) -> None:
        while not self._stop_flag.is_set():
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(gray)
            annotate_detections(frame, dets, estimator)
            seen = sorted(d.id for d in dets)
            cv2.putText(frame, f"{self.family}  detected {len(seen)}: {seen}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"exposure {self.exposure}   gain {self.gain}", (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError("JPEG encoding of camera frame failed")
            self.box.publish(jpeg.tobytes())

    def stop(self) -> None:
        """Synchronously stop the thread and release the device + claim."""
        with self._lock:
            if self._thread is None:
                return
            self._stop_flag.set()
            self._thread.join(timeout=10.0)
            assert not self._thread.is_alive(), "camera thread refused to stop"
            self._thread = None
            self._cap.release()
            self._cap = None
            self._runner.release_external(HOLDER_NAME)

    def set_exposure(self, value: int) -> None:
        set_exposure(value)
        self.exposure = value

    def set_gain(self, value: int) -> None:
        v4l2_set("gain", value)
        self.gain = value


def default_settings() -> dict:
    """Marker-tuned capture defaults shown in the camera page form."""
    return {"exposure": MARKER_EXPOSURE, "gain": MARKER_GAIN}
