"""Live AprilTag marker observations in the arm base frame for real rollouts.

Replaces the FK stand-in in the rollout loop: a background thread detects the arm
tags with the camera and maps them into the base frame through the calibrated
extrinsics (`real/extrinsics.py`, produced by `real/calibrate_qpos.py`),
so the policy consumes *measured* tag poses in the exact `(xyz, axis-angle)` form
it saw in sim (`src/base_env.py::marker_world_poses`).

Per frame the fixed table tag re-anchors the camera (`base_cam_from_table`), so a
bumped camera self-corrects. A tag the camera can't see — or any frame missing
the table tag, where the camera can't be re-anchored at all — keeps its last
measured pose while its age grows, the same hold-last-pose + age convention
training used for undetected tags (src/base_env.py); a tag never seen this
session reads all-zero with age pinned at MARKER_AGE_CAP_S.

With track_cube the same pipeline also measures the sponge's tag (CUBE_TAG_ID
on its largest face) and serves it via cube_pose() — the raw tag pose under the
identical hold-last convention, exactly what the sim cube obs carries. No
quarter_turns entry: the tag's yaw about its own normal relative to the sponge
is uncalibrated for now (TODO.md).
"""
import threading
import time

import cv2
import numpy as np

from real.camera import open_camera
from real.detect import make_detector
from real.extrinsics import (
    base_cam_from_table,
    load_extrinsics,
    mat_to_pos_rotvec,
    quarter_turn_mat,
    rt_to_mat,
)
from real.marker_spec import (
    ARM_TAG_TO_SITE,
    CUBE_TAG_ID,
    MARKER_EXPOSURE,
    MARKER_GAIN,
    TABLE_TAG_ID,
)
from real.pose import PoseEstimator, load_intrinsics
from src.base_env import MARKER_AGE_CAP_S, MARKER_SITE_NAMES, N_MARKERS

# The marker-age obs is consume-time minus *capture* time (src/camera_sim.py),
# but the thread only knows when cap.read() returned; the probe
# (sysid/probe_cam_latency.py, run probe_1783181402) measured the
# capture -> cap.read() leg at 39.5 ms, so each frame's capture instant is
# t_recv minus this.
CAPTURE_TO_READ_S = 0.0395


class CameraMarkerSource:
    """Threaded camera → base-frame marker poses. start() → marker_poses() → stop()."""

    def __init__(self, family: str = "apriltag", on_frame=None,
                 track_cube: bool = False):
        # focus from the calibration: the extrinsics (and intrinsics) are only
        # valid at the focus they were solved at, so we open the lens there.
        # quarter_turns un-rotate each tag's measured pose to the sim convention.
        self.T_base_table, _, self.focus, self.quarter_turns = load_extrinsics()
        self.detector = make_detector(family)
        self.estimator = PoseEstimator(*load_intrinsics())
        # Obs slot order is MARKER_SITE_NAMES; map each slot to its physical tag id.
        site_to_tag = {site: tag for tag, site in ARM_TAG_TO_SITE.items()}
        self.slot_tags = [site_to_tag[name] for name in MARKER_SITE_NAMES]
        self._wanted = set(ARM_TAG_TO_SITE) | {TABLE_TAG_ID}
        # Also detect the sponge's tag and maintain its held pose (cube_pose()).
        self.track_cube = bool(track_cube)
        if self.track_cube:
            self._wanted.add(CUBE_TAG_ID)
        # Called from the capture thread after every processed frame with
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
        self._cube_pos = np.zeros(3)
        self._cube_rot = np.zeros(3)
        self._cube_last_capture_t = -np.inf
        self._recv_t: float | None = None
        self._read_ms = float("nan")
        self._detect_ms = float("nan")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None
        self._thread = None
        self.error = None      # set if the capture loop dies; readers re-raise it

    def start(self) -> None:
        self._cap = open_camera(focus=self.focus, exposure=MARKER_EXPOSURE, gain=MARKER_GAIN)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Block until the first frame lands so a camera that never delivers
        # fails loud here instead of serving zeroed poses (and NaN staleness,
        # which no downstream age gate would catch) for the whole rollout.
        deadline = time.monotonic() + 5.0
        while True:
            if self.error is not None:
                raise self.error
            with self._lock:
                if self._recv_t is not None:
                    return
            if time.monotonic() > deadline:
                raise RuntimeError("no camera frame within 5 s of start()")
            time.sleep(0.01)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                t_read = time.monotonic()
                ok, frame = self._cap.read()
                t_recv = time.monotonic()
                if not ok:
                    raise RuntimeError("camera read failed mid-session")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t_det = time.monotonic()
                poses = {d.id: self.estimator.estimate(d)
                         for d in self.detector.detect(gray) if d.id in self._wanted}
                detect_ms = (time.monotonic() - t_det) * 1e3
                pos, rot, detected, cube_pose = self._poses_to_base(poses)
                with self._lock:
                    self._pos[detected] = pos[detected]
                    self._rot[detected] = rot[detected]
                    self._last_capture_t[detected] = t_recv - CAPTURE_TO_READ_S
                    if cube_pose is not None:
                        self._cube_pos, self._cube_rot = cube_pose
                        self._cube_last_capture_t = t_recv - CAPTURE_TO_READ_S
                    self._recv_t = t_recv
                    self._read_ms = (t_recv - t_read) * 1e3
                    self._detect_ms = detect_ms
                if self._on_frame is not None:
                    self._on_frame(t_recv, pos, rot, detect_ms)
        except Exception as exc:   # surface to the consumer thread; a dead camera
            self.error = exc       # must fail loud, not freeze the last poses

    def _poses_to_base(self, poses: dict):
        """Camera-frame tag poses -> raw per-frame base-frame
        (pos (N,3), rot (N,3), detected (N,) bool, cube_pose); undetected/
        un-anchored tags zeroed with detected=False, cube_pose is (pos, rot)
        or None when the cube tag wasn't measured this frame. These are the
        per-frame values (used as-is for _on_frame telemetry) — the held obs
        state only folds in the detected ones."""
        pos = np.zeros((N_MARKERS, 3))
        rot = np.zeros((N_MARKERS, 3))
        detected = np.zeros(N_MARKERS, dtype=bool)
        if TABLE_TAG_ID not in poses:
            # No table tag this frame -> can't re-anchor the camera.
            return pos, rot, detected, None
        T_base_cam = base_cam_from_table(self.T_base_table, *poses[TABLE_TAG_ID])
        for i, tag in enumerate(self.slot_tags):
            if tag in poses:
                # Un-rotate the glue offset so marker_rot matches the sim convention.
                T_cam_tag = rt_to_mat(*poses[tag]) @ quarter_turn_mat(-self.quarter_turns[tag])
                pos[i], rot[i] = mat_to_pos_rotvec(T_base_cam @ T_cam_tag)
                detected[i] = True
        cube_pose = None
        if self.track_cube and CUBE_TAG_ID in poses:
            # Raw tag pose, no quarter_turns un-rotation: the sim trained on
            # the tag site's own pose, and the tag's in-plane glue yaw on the
            # sponge is uncalibrated (TODO.md).
            cube_pose = mat_to_pos_rotvec(T_base_cam @ rt_to_mat(*poses[CUBE_TAG_ID]))
        return pos, rot, detected, cube_pose

    def marker_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pos (N,3), rot (N,3), age_s (N,)) in the base frame: each tag's
        most recent detection and its age at this instant, capped at
        MARKER_AGE_CAP_S like training; never-seen tags are zero with age at
        the cap. Re-raises any exception that killed the capture thread —
        otherwise a dead camera would silently serve the held poses forever."""
        if self.error is not None:
            raise self.error
        with self._lock:
            pos = self._pos.copy()
            rot = self._rot.copy()
            last_capture_t = self._last_capture_t.copy()
        age = np.minimum(MARKER_AGE_CAP_S, time.monotonic() - last_capture_t)
        return pos, rot, age

    def cube_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(pos (3,), rot (3,), age_s) of the cube tag in the base frame —
        marker_poses() semantics: held last detection plus its age, capped at
        MARKER_AGE_CAP_S; zeros with age at the cap when never seen. Re-raises
        any exception that killed the capture thread."""
        assert self.track_cube, "CameraMarkerSource was constructed without track_cube"
        if self.error is not None:
            raise self.error
        with self._lock:
            pos = self._cube_pos.copy()
            rot = self._cube_rot.copy()
            last_capture_t = self._cube_last_capture_t
        age = min(MARKER_AGE_CAP_S, time.monotonic() - last_capture_t)
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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
