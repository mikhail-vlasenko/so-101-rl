"""In-server camera capture for the panel's camera page.

Owns both C922 units while streaming: one capture thread per rig camera reads
frames, runs the marker detector + that unit's pose estimator, draws the same
overlays as `real.vision.marker_view` (via `real.vision.overlay.annotate_detections`),
and publishes JPEGs into a per-camera FrameBox that the panel serves at
/camera/stream/{name}. Each camera is opened by rig name
(`real.vision.stereo_rig`), so the overlay poses use per-lens intrinsics at the
focus they were calibrated at — same configuration the stereo scripts use.

The service registers itself as a CAMERA holder in the Runner's resource
accounting, so starting the stream blocks native camera tools (and vice
versa) with a clear 409 instead of two processes fighting over the devices.
`stop()` is synchronous: when it returns, the devices are released.
"""

from __future__ import annotations

import threading

import cv2

from real.vision.camera import set_exposure, v4l2_set
from real.vision.detect import make_detector
from real.marker_spec import MARKER_EXPOSURE, MARKER_GAIN
from real.vision.overlay import annotate_detections
from real.vision.pose import PoseEstimator
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera, rig_device_index

from panel.registry import Resource
from panel.runner import ResourceBusyError, Runner
from panel.streamer import FrameBox

HOLDER_NAME = "the panel's camera stream"
JPEG_QUALITY = 80


class CameraService:
    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flag = threading.Event()
        self._caps: dict[str, object] = {}
        self._devices: dict[str, int] = {}
        self.boxes = {name: FrameBox() for name in CAMERA_NAMES}
        self.family: str | None = None
        self.exposure: int | None = None
        self.gain: int | None = None

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads.values())

    def start(self, family: str, exposure: int, gain: int) -> None:
        """Claim both cameras and start one capture thread each. Raises
        ResourceBusyError if anything (including a panel-launched native tool)
        holds the cameras, ValueError on a bad family, and RuntimeError if a
        camera isn't connected."""
        with self._lock:
            if self.is_running():
                raise ResourceBusyError(f"cameras are in use by {HOLDER_NAME}")
            # One detector per thread: the AprilTag backend wraps a C detector
            # with internal worker state, so the two loops must not share one.
            detectors = {name: make_detector(family) for name in CAMERA_NAMES}
            self._runner.claim_external(HOLDER_NAME, {Resource.CAMERA})
            estimators = {}
            try:
                for name in CAMERA_NAMES:
                    cap, camera_matrix, dist_coeffs = open_rig_camera(
                        name, exposure=exposure, gain=gain)
                    self._caps[name] = cap
                    self._devices[name] = rig_device_index(name)
                    estimators[name] = PoseEstimator(camera_matrix, dist_coeffs)
            except BaseException:
                # A missing/busy second camera must not leave the first one open.
                self._release_devices()
                self._runner.release_external(HOLDER_NAME)
                raise
            self.family = family
            self.exposure = exposure
            self.gain = gain
            self._stop_flag.clear()
            self._threads = {
                name: threading.Thread(
                    target=self._capture_loop,
                    args=(name, detectors[name], estimators[name]),
                    name=f"panel-camera-{name}", daemon=True)
                for name in CAMERA_NAMES}
            for thread in self._threads.values():
                thread.start()

    def _capture_loop(self, name: str, detector, estimator) -> None:
        cap = self._caps[name]
        box = self.boxes[name]
        while not self._stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera read failed ({name})")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(gray)
            annotate_detections(frame, dets, estimator)
            seen = sorted(d.id for d in dets)
            cv2.putText(frame, f"{name}  {self.family}  detected {len(seen)}: {seen}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"exposure {self.exposure}   gain {self.gain}", (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"JPEG encoding of camera frame failed ({name})")
            box.publish(jpeg.tobytes())

    def _release_devices(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._devices.clear()

    def stop(self) -> None:
        """Synchronously stop the threads and release the devices + claim."""
        with self._lock:
            if not self._threads:
                return
            self._stop_flag.set()
            for name, thread in self._threads.items():
                thread.join(timeout=10.0)
                assert not thread.is_alive(), f"camera thread {name} refused to stop"
            self._threads = {}
            self._release_devices()
            self._runner.release_external(HOLDER_NAME)

    def set_exposure(self, value: int) -> None:
        for device in self._devices.values():
            set_exposure(value, device)
        self.exposure = value

    def set_gain(self, value: int) -> None:
        for device in self._devices.values():
            v4l2_set("gain", value, device)
        self.gain = value


def default_settings() -> dict:
    """Marker-tuned capture defaults shown in the camera page form."""
    return {"exposure": MARKER_EXPOSURE, "gain": MARKER_GAIN}
