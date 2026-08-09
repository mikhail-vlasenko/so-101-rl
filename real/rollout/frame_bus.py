"""Per-camera capture threads with fan-out to multiple consumers.

A V4L2 device can only be opened once, but the rollout needs the same main-cam
frames in two places — the AprilTag marker detector (real/rollout/marker_obs.py)
and the dense-stereo object worker (real/rollout/object_obs.py). Each `CameraFeed`
owns one camera (opened by rig name at its calibrated focus,
real/vision/stereo_rig.py) and republishes every frame with a sequence number;
consumers block on `wait_next` for frames they haven't processed yet, each at
its own pace — a slow consumer skips frames instead of stalling the capture.

Timestamps are capture-time estimates: `t_recv - CAPTURE_TO_READ_S`, the same
convention the age channels trained on (probe run probe_1783181402: the
capture -> cap.read() leg is 39.5 ms).
"""
import threading
import time

from real.vision.stereo_rig import open_rig_camera

# capture -> cap.read() latency (s); see module docstring.
CAPTURE_TO_READ_S = 0.0395


class CameraFeed:
    """One camera's capture thread. start() -> wait_next()/latest() -> stop()."""

    def __init__(self, name):
        self.name = name
        self.camera_matrix = None
        self.dist_coeffs = None
        self._cap = None
        self._thread = None
        self._cond = threading.Condition()
        self._seq = 0
        self._frame = None
        self._t_capture = None
        self._t_recv = None
        self._read_ms = float("nan")
        self._stop = threading.Event()
        self.error = None    # set if the capture loop dies; readers re-raise

    def start(self):
        self._cap, self.camera_matrix, self.dist_coeffs = open_rig_camera(self.name)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Block until the first frame lands so a camera that never delivers
        # fails loud here instead of starving every consumer.
        self.wait_next(0, timeout=5.0)

    def _loop(self):
        try:
            while not self._stop.is_set():
                t_read = time.monotonic()
                ok, frame = self._cap.read()
                t_recv = time.monotonic()
                if not ok:
                    raise RuntimeError(f"camera read failed mid-session ({self.name})")
                with self._cond:
                    self._seq += 1
                    self._frame = frame
                    self._t_capture = t_recv - CAPTURE_TO_READ_S
                    self._t_recv = t_recv
                    self._read_ms = (t_recv - t_read) * 1e3
                    self._cond.notify_all()
        except Exception as exc:   # surface to every consumer; a dead camera
            self.error = exc       # must fail loud, not freeze held frames
            with self._cond:
                self._cond.notify_all()

    def wait_next(self, last_seq, timeout=2.0):
        """Block until a frame newer than `last_seq` arrives; returns
        (seq, t_capture, frame). Re-raises a dead capture thread's error;
        raises on timeout (a stalled camera must not silently freeze the
        consumer's held state)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self.error is not None:
                    raise self.error
                if self._seq > last_seq:
                    return self._seq, self._t_capture, self._frame
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"no new frame from '{self.name}' within {timeout} s")
                self._cond.wait(remaining)

    def latest(self):
        """(seq, t_capture, frame) of the newest frame without blocking;
        seq 0 / None frame before the first capture."""
        if self.error is not None:
            raise self.error
        with self._cond:
            return self._seq, self._t_capture, self._frame

    def stats(self):
        """(staleness_s, read_ms) of the newest frame: how old it is right
        now, and how long its cap.read() blocked."""
        if self.error is not None:
            raise self.error
        with self._cond:
            t_recv = self._t_recv
            read_ms = self._read_ms
        if t_recv is None:
            return float("nan"), float("nan")
        return time.monotonic() - t_recv, read_ms

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()


class FrameBus:
    """The rig's camera feeds, started and stopped together."""

    def __init__(self, names):
        self.feeds = {name: CameraFeed(name) for name in names}

    def start(self):
        for feed in self.feeds.values():
            feed.start()

    def stop(self):
        for feed in self.feeds.values():
            feed.stop()
