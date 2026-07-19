"""Live AprilTag marker observations in the arm base frame for real rollouts.

Replaces the FK stand-in in the rollout loop: a consumer thread on the main
camera's frame feed (real/rollout/frame_bus.py — the same frames the SAM
object tracker consumes) detects the arm tags and maps them into the base
frame through the calibrated extrinsics (`real/calib/extrinsics.py`, produced
by `real/calib/calibrate_qpos.py`), so the policy consumes *measured* tag
poses in the exact `(xyz, axis-angle)` form it saw in sim
(`src/base_env.py::marker_world_poses`).

Per frame the fixed table tag re-anchors the camera (`base_cam_from_table`,
EMA-smoothed), so a bumped camera self-corrects. A tag the camera can't see —
or any frame missing the table tag, where the camera can't be re-anchored at
all — keeps its last measured pose while its age grows, the same
hold-last-pose + age convention training used for undetected tags
(src/base_env.py); a tag never seen this session reads all-zero with age
pinned at MARKER_AGE_CAP_S.

This source serves the ARM tags and the table anchor only. The manipulated
object is tracked tag-free by real/rollout/object_obs.ObjectSource on the
same frame bus.
"""
import threading
import time

import cv2
import numpy as np

from real.calib.extrinsics import (
    PoseEMA,
    base_cam_from_table,
    load_extrinsics,
    mat_to_pos_rotvec,
    quarter_turn_mat,
    rt_to_mat,
)
from real.marker_spec import ARM_TAG_TO_SITE, TABLE_TAG_ID
from real.rollout.frame_bus import CAPTURE_TO_READ_S
from real.vision.detect import make_detector
from real.vision.pose import PoseEstimator
from src.base_env import MARKER_AGE_CAP_S, MARKER_SITE_NAMES, N_MARKERS

# EMA weight for the re-anchored camera pose. The camera is bolted down (static
# within a session), so smoothing base_cam_from_table across frames denoises the
# per-frame table-tag detection jitter that otherwise moves every arm tag in
# common. At 30 fps, 0.05 is a ~0.7 s time constant: heavy denoising while still
# tracking a real bump within a couple of seconds.
CAM_EMA_ALPHA = 0.05


class CameraMarkerSource:
    """Frame-feed consumer → base-frame marker poses. start() → marker_poses() → stop().

    `feed` is the main camera's started CameraFeed; the capture settings
    (per-unit calibrated focus, marker exposure/gain) are the feed's concern
    (real/vision/stereo_rig.py).
    """

    def __init__(self, feed, family: str = "apriltag", on_frame=None):
        self.T_base_table, _, _, self.quarter_turns = load_extrinsics()
        self.feed = feed
        self.detector = make_detector(family)
        # Obs slot order is MARKER_SITE_NAMES; map each slot to its physical tag id.
        site_to_tag = {site: tag for tag, site in ARM_TAG_TO_SITE.items()}
        self.slot_tags = [site_to_tag[name] for name in MARKER_SITE_NAMES]
        self._wanted = set(ARM_TAG_TO_SITE) | {TABLE_TAG_ID}
        # Called from the consumer thread after every processed frame with
        # (t_recv, pos (N,3), rot (N,3), detect_ms) — base-frame poses,
        # undetected tags zeroed. Lets a recorder (sysid/probe_cam_latency.py)
        # timestamp every frame instead of polling for the newest one.
        self._on_frame = on_frame
        # Hold-last-pose state (training convention, src/base_env.py): each
        # tag's most recent detection and its capture time; -inf = never seen
        # (zero pose, age pinned at MARKER_AGE_CAP_S). Written under _lock
        # together with the frame telemetry so a reader gets a consistent
        # snapshot.
        self._pos = np.zeros((N_MARKERS, 3))
        self._rot = np.zeros((N_MARKERS, 3))
        self._last_capture_t = np.full(N_MARKERS, -np.inf)
        # Capture time of the last frame the table anchor tag was seen in; -inf =
        # never. Until it is fresh the camera can't be mapped to the base frame at
        # all (every pose is zeroed/held), so warmup() blocks on it.
        self._table_last_capture_t = -np.inf
        # EMA state for the re-anchored camera pose (see CAM_EMA_ALPHA). Seeded
        # by the first table-tag solve; touched only from the consumer thread.
        self._cam_ema = PoseEMA(CAM_EMA_ALPHA)
        self._recv_t: float | None = None
        self._read_ms = float("nan")
        self._detect_ms = float("nan")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None      # set if the consumer loop dies; readers re-raise it
        self.estimator = None  # built in start() from the feed's intrinsics

    def start(self) -> None:
        assert self.feed.camera_matrix is not None, \
            "start the CameraFeed before the CameraMarkerSource"
        self.estimator = PoseEstimator(self.feed.camera_matrix,
                                       self.feed.dist_coeffs)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Block until the first frame is processed so a dead feed fails loud
        # here instead of serving zeroed poses for the whole rollout.
        deadline = time.monotonic() + 5.0
        while True:
            if self.error is not None:
                raise self.error
            with self._lock:
                if self._recv_t is not None:
                    return
            if time.monotonic() > deadline:
                raise RuntimeError("no processed camera frame within 5 s of start()")
            time.sleep(0.01)

    def _loop(self) -> None:
        try:
            seq = 0
            while not self._stop.is_set():
                seq, t_capture, frame = self.feed.wait_next(seq)
                _, read_ms = self.feed.stats()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t_det = time.monotonic()
                poses = {d.id: self.estimator.estimate(d)
                         for d in self.detector.detect(gray) if d.id in self._wanted}
                detect_ms = (time.monotonic() - t_det) * 1e3
                pos, rot, detected = self._poses_to_base(poses)
                t_recv = t_capture + CAPTURE_TO_READ_S
                with self._lock:
                    self._pos[detected] = pos[detected]
                    self._rot[detected] = rot[detected]
                    self._last_capture_t[detected] = t_capture
                    if TABLE_TAG_ID in poses:
                        self._table_last_capture_t = t_capture
                    self._recv_t = t_recv
                    self._read_ms = read_ms
                    self._detect_ms = detect_ms
                if self._on_frame is not None:
                    self._on_frame(t_recv, pos, rot, detect_ms)
        except Exception as exc:   # surface to the consumer thread; a dead feed
            self.error = exc       # must fail loud, not freeze the last poses

    def _poses_to_base(self, poses: dict):
        """Camera-frame tag poses -> raw per-frame base-frame
        (pos (N,3), rot (N,3), detected (N,) bool); undetected/un-anchored
        tags zeroed with detected=False. These are the per-frame values (used
        as-is for _on_frame telemetry) — the held obs state only folds in the
        detected ones."""
        pos = np.zeros((N_MARKERS, 3))
        rot = np.zeros((N_MARKERS, 3))
        detected = np.zeros(N_MARKERS, dtype=bool)
        if TABLE_TAG_ID not in poses:
            # No table tag this frame -> can't re-anchor the camera.
            return pos, rot, detected
        T_base_cam = self._cam_ema.update(
            base_cam_from_table(self.T_base_table, *poses[TABLE_TAG_ID]))
        for i, tag in enumerate(self.slot_tags):
            if tag in poses:
                # Un-rotate the glue offset so marker_rot matches the sim convention.
                T_cam_tag = rt_to_mat(*poses[tag]) @ quarter_turn_mat(-self.quarter_turns[tag])
                pos[i], rot[i] = mat_to_pos_rotvec(T_base_cam @ T_cam_tag)
                detected[i] = True
        return pos, rot, detected

    def marker_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pos (N,3), rot (N,3), age_s (N,)) in the base frame: each tag's
        most recent detection and its age at this instant, capped at
        MARKER_AGE_CAP_S like training; never-seen tags are zero with age at
        the cap. Re-raises any exception that killed the consumer thread —
        otherwise a dead camera would silently serve the held poses forever."""
        if self.error is not None:
            raise self.error
        with self._lock:
            pos = self._pos.copy()
            rot = self._rot.copy()
            last_capture_t = self._last_capture_t.copy()
        age = np.minimum(MARKER_AGE_CAP_S, time.monotonic() - last_capture_t)
        return pos, rot, age

    def frame_stats(self) -> tuple[float, float, float]:
        """(staleness_s, read_ms, detect_ms) for the most recent processed frame.

        staleness_s = now - the time cap.read() handed us the frame: how old the
        pose the policy is about to consume is (detection compute plus however
        long it then waited for the control loop to pick it up). read_ms is how
        long cap.read() blocked — roughly one frame interval means we're pulling
        fresh frames; near-zero while detect_ms is large means the driver queue
        is feeding backlog and the buffer isn't draining. All NaN before the
        first frame lands."""
        if self.error is not None:
            raise self.error
        with self._lock:
            recv_t = self._recv_t
            read_ms = self._read_ms
            detect_ms = self._detect_ms
        if recv_t is None:
            return float("nan"), float("nan"), float("nan")
        return time.monotonic() - recv_t, read_ms, detect_ms

    def table_age(self) -> float:
        """Seconds since the table anchor tag (id TABLE_TAG_ID) was last
        detected; inf if never seen this session. Re-raises a dead-thread
        error."""
        if self.error is not None:
            raise self.error
        with self._lock:
            last = self._table_last_capture_t
        return time.monotonic() - last

    def warmup(self, timeout_s: float = 5.0, fresh_s: float = 0.25) -> float:
        """Block until the table anchor tag is freshly detected (age <= fresh_s),
        so the episode starts with the camera actually anchored to the base frame
        instead of steering the arm on zeroed/held poses while the sensor and
        detector settle. Re-raises a dead-thread error; raises on timeout so a
        mis-framed or out-of-focus camera fails loud before the arm moves.
        Returns how long it waited (s)."""
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        while True:
            if self.table_age() <= fresh_s:   # re-raises self.error if the thread died
                return time.monotonic() - t0
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"table anchor tag (id {TABLE_TAG_ID}) not detected within "
                    f"{timeout_s:.0f} s of warmup; check camera framing/focus.")
            time.sleep(0.02)

    def stop(self) -> None:
        """Stop the consumer thread; the CameraFeed (and its device) belongs
        to the caller."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
