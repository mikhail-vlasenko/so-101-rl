"""Live AprilTag marker observations in the arm base frame for real rollouts.

Replaces the FK stand-in in the rollout loop: a background thread detects the arm
tags with the camera and maps them into the base frame through the calibrated
extrinsics (`real/extrinsics.py`, produced by `real/calibrate_qpos.py`),
so the policy consumes *measured* tag poses in the exact `(xyz, axis-angle)` form
it saw in sim (`src/base_env.py::marker_world_poses`).

Per frame the fixed table tag re-anchors the camera (`base_cam_from_table`), so a
bumped camera self-corrects. A tag the camera can't see — or any frame missing
the table tag, where the camera can't be re-anchored at all — is zeroed, the same
convention training used for undetected tags.
"""
import threading

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
from real.marker_spec import ARM_TAG_TO_SITE, MARKER_EXPOSURE, MARKER_GAIN, TABLE_TAG_ID
from real.pose import PoseEstimator, load_intrinsics
from src.base_env import MARKER_SITE_NAMES, N_MARKERS


class CameraMarkerSource:
    """Threaded camera → base-frame marker poses. start() → marker_poses() → stop()."""

    def __init__(self, family: str = "apriltag"):
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
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None
        self._thread = None

    def start(self) -> None:
        self._cap = open_camera(focus=self.focus, exposure=MARKER_EXPOSURE, gain=MARKER_GAIN)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            poses = {d.id: self.estimator.estimate(d)
                     for d in self.detector.detect(gray) if d.id in self._wanted}
            with self._lock:
                self._latest = poses   # whole-dict swap; readers never see a partial update

    def marker_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """(pos (N,3), rot (N,3)) in the base frame; undetected/un-anchored tags zeroed."""
        with self._lock:
            poses = self._latest
        pos = np.zeros((N_MARKERS, 3))
        rot = np.zeros((N_MARKERS, 3))
        if TABLE_TAG_ID not in poses:
            return pos, rot   # no table tag this frame -> can't re-anchor the camera
        T_base_cam = base_cam_from_table(self.T_base_table, *poses[TABLE_TAG_ID])
        for i, tag in enumerate(self.slot_tags):
            if tag in poses:
                # Un-rotate the glue offset so marker_rot matches the sim convention.
                T_cam_tag = rt_to_mat(*poses[tag]) @ quarter_turn_mat(-self.quarter_turns[tag])
                pos[i], rot[i] = mat_to_pos_rotvec(T_base_cam @ T_cam_tag)
        return pos, rot

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
