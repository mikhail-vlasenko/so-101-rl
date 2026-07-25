"""Tag-free object observation source for real rollouts: SAM stereo channels.

`ObjectSource` consumes both cameras' frame feeds (real/rollout/frame_bus.py)
and serves the dual-channel object obs the policy trained on
(src/shape_obs.py — the exact state machine the sim env drives):

- **live**: per camera, a streaming SAM2 tracker (seeded once by a SAM3 text
  prompt, re-prompted after a run of empty masks — re-acquisition) reduces
  each frame's mask to its pixel centroid; the two centroids triangulate to a
  base-frame point (real/vision/stereo.py). Either view losing the object is
  a live miss — the held value ages.
- **precise**: while the live track proves the object static
  (`ObjectChannelDriver.static_now`) and both masks are near their
  static-window baseline area (the occlusion gate), a worker thread runs the
  visual hull (real/tracking/hull_shape.py) and refreshes the center + √M
  with the window's running average (M in the linear domain before the sqrt).
  The hull work never blocks the tick.

Each camera re-anchors itself to the base frame per frame from the table tag
(EMA'd, like the marker pipeline). The `--gui` debug overlay
(`debug_views()`) tints the mask, dots the live centroid, and projects the
served √M ellipsoid into each view — the picture that makes failures visible.
"""
import threading
import time

import cv2
import numpy as np

from real.calib.extrinsics import PoseEMA, base_cam_from_table, load_extrinsics
from real.marker_spec import TABLE_TAG_ID
from real.tracking.hull_shape import hull_estimate
from real.vision.detect import make_detector
from real.vision.pose import PoseEstimator
from real.vision.stereo import pixel_rays, triangulate_rays
from src.shape_obs import (
    MARKER_AGE_CAP_S,
    ObjectChannelDriver,
    sqrtm_from_cov,
    sqrtm_from_upper,
)

# Same table-anchor smoothing as the marker pipeline.
CAM_EMA_ALPHA = 0.05

# Consecutive empty masks in a view before SAM3 re-acquires the object there.
REPROMPT_AFTER_EMPTY = 15


def draw_object_overlay(view, mask, centroid_px, live, center, sqrtm6,
                        camera_matrix, dist_coeffs, T_base_cam):
    """Annotate one camera view in place: green mask tint, red live-centroid
    dot, and the served √M ellipsoid ({d: dᵀM⁻¹d <= 3}, the box-equivalent
    spread) projected as a yellow outline around the precise center."""
    m = mask.astype(np.uint8)
    view[m == 1] = (0.55 * np.array((0, 200, 0)) + 0.45 * view[m == 1]).astype(np.uint8)
    if centroid_px is not None:
        cv2.circle(view, (int(centroid_px[0]), int(centroid_px[1])), 6, (0, 0, 255), -1)
    if np.any(sqrtm6):
        sphere = np.random.default_rng(0).normal(size=(64, 3))
        sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
        pts = center + np.sqrt(3.0) * sphere @ sqrtm_from_upper(sqrtm6).T
        T_cam_base = np.linalg.inv(T_base_cam)
        cam_pts = pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
        if np.all(cam_pts[:, 2] > 0.0):
            px, _ = cv2.projectPoints(cam_pts.reshape(-1, 1, 3), np.zeros(3),
                                      np.zeros(3), camera_matrix, dist_coeffs)
            hull = cv2.convexHull(px.reshape(-1, 2).astype(np.int32))
            cv2.polylines(view, [hull], True, (0, 220, 220), 2)
    return view


class ObjectSource:
    """Frame-feed consumer → dual-channel object obs. start() → object_obs() → stop()."""

    def __init__(self, feeds, prompt: str = "sponge", sam2_model: str = "tiny",
                 family: str = "apriltag", keep_views: bool = False):
        self.feeds = feeds
        self.cameras = tuple(feeds)
        self.prompt = prompt
        self.sam2_model = sam2_model
        self.keep_views = bool(keep_views)
        self.T_base_table, _, _, _ = load_extrinsics()
        self.detector = make_detector(family)
        self._estimators = {}
        self._cam_ema = {name: PoseEMA(CAM_EMA_ALPHA) for name in self.cameras}
        self._T_base_cam = {name: None for name in self.cameras}
        self._sam3 = None
        self._trackers = {}
        self._empty_run = {name: 0 for name in self.cameras}
        # Shared channel state machine (the sim twin drives the same class).
        self._driver = ObjectChannelDriver()
        self._lock = threading.Lock()
        # Occlusion gate baseline: max mask area per camera within the current
        # static window; reset while the object moves. v1 weakness by design:
        # an occlusion present from the window's first frame inflates the
        # baseline's honesty — the estimator eval slices catch it (TODO.md).
        self._window_max_area = {name: 0.0 for name in self.cameras}
        # Hull-refresh handoff: the newest gated frame pair only (the worker
        # always processes the freshest static view), plus a generation
        # counter that resets the worker's window average when motion breaks
        # the static window.
        self._hull_job = None
        self._hull_cond = threading.Condition()
        self._window_gen = 0
        self._stop = threading.Event()
        self._track_thread = None
        self._hull_thread = None
        self.error = None
        # Telemetry (written by the worker threads, read via stats()).
        self._sam_ms = {name: float("nan") for name in self.cameras}
        self._hull_ms = float("nan")
        self._lost_frames = 0
        self._views = {}

    # ------------------------------------------------------------- lifecycle

    def start(self, warmup_timeout_s: float = 60.0) -> None:
        """Prompt SAM3 on both views (fails loud when the object isn't
        found), prime the per-camera SAM2 trackers, and start the tracking +
        hull threads. Blocks until BOTH channels carry a real measurement, so
        the rollout begins on the obs distribution the policy trained against
        (see the precise-channel wait below)."""
        from real.tracking.sam_seg import MaskTracker, load_sam3, text_to_mask

        for name in self.cameras:
            assert self.feeds[name].camera_matrix is not None, \
                f"start the '{name}' CameraFeed before ObjectSource"
            self._estimators[name] = PoseEstimator(self.feeds[name].camera_matrix,
                                                   self.feeds[name].dist_coeffs)
        print(f"ObjectSource: prompting SAM3 with {self.prompt!r} ...", flush=True)
        self._sam3 = load_sam3()
        self._text_to_mask = text_to_mask
        for name in self.cameras:
            _, _, frame = self.feeds[name].latest()
            mask, score = text_to_mask(self._sam3, frame, self.prompt)
            print(f"  {name}: score {score:.2f}, area {int(mask.sum())} px", flush=True)
            self._trackers[name] = MaskTracker(self.sam2_model)
            self._trackers[name].prime(frame, mask)
        self._track_thread = threading.Thread(target=self._track_loop, daemon=True)
        self._hull_thread = threading.Thread(target=self._hull_loop, daemon=True)
        self._track_thread.start()
        self._hull_thread.start()

        deadline = time.monotonic() + warmup_timeout_s
        while True:
            if self.error is not None:
                raise self.error
            _, live_age, _, _, precise_age = self.object_obs()
            # The precise channel is waited on too, not just the live fix: the
            # sim twin seeds pre-episode static evidence at reset
            # (base_env's seed_static, modelling the object settling before a
            # rollout starts), so the policy has never seen the never-measured
            # state — zero √M at the age cap — that the real rig otherwise
            # serves for its first STATIC_DWELL_S plus a hull pass.
            if live_age < 0.5 and precise_age < MARKER_AGE_CAP_S:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"object channels not ready within {warmup_timeout_s:.0f} s "
                    f"of ObjectSource.start() (live_age {live_age:.2f} s, "
                    f"precise_age {precise_age:.2f} s); the precise channel "
                    "needs the object still and unoccluded in both views for a "
                    "settling window — check placement, lighting and prompt")
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        with self._hull_cond:
            self._hull_cond.notify_all()
        for thread in (self._track_thread, self._hull_thread):
            if thread is not None:
                thread.join(timeout=3.0)

    # ------------------------------------------------------------- consumers

    def object_obs(self):
        """(live (3,), live_age, center (3,), sqrtm6 (6,), precise_age) at
        this instant — the cube block of the actor frame. Re-raises any
        exception that killed a worker thread."""
        if self.error is not None:
            raise self.error
        with self._lock:
            return self._driver.serve(time.monotonic())

    def stats(self):
        """Telemetry dict: per-camera SAM track ms, hull ms, lost frames."""
        if self.error is not None:
            raise self.error
        with self._lock:
            return {"sam_ms": dict(self._sam_ms), "hull_ms": self._hull_ms,
                    "lost_frames": self._lost_frames}

    def debug_views(self):
        """{camera: annotated BGR frame} of the newest processed pair (only
        populated with keep_views=True)."""
        if self.error is not None:
            raise self.error
        with self._lock:
            return dict(self._views)

    # --------------------------------------------------------------- workers

    def _anchor(self, name, frame):
        """Update this camera's EMA'd base-frame anchoring from the table tag;
        holds the last anchor when the tag is missed this frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = {d.id: d for d in self.detector.detect(gray)}
        if TABLE_TAG_ID in dets:
            self._T_base_cam[name] = self._cam_ema[name].update(base_cam_from_table(
                self.T_base_table, *self._estimators[name].estimate(dets[TABLE_TAG_ID])))
        return self._T_base_cam[name]

    def _track_view(self, name, frame):
        """(mask, centroid_px | None) for one view; re-prompts SAM3 after a
        run of empty masks (re-acquisition). A failed re-prompt (object fully
        hidden, e.g. inside the gripper) keeps the miss and tries again after
        the next empty run."""
        t0 = time.monotonic()
        mask = self._trackers[name].track(frame)
        self._sam_ms[name] = (time.monotonic() - t0) * 1e3
        if not mask.any():
            self._empty_run[name] += 1
            if self._empty_run[name] >= REPROMPT_AFTER_EMPTY:
                self._empty_run[name] = 0
                try:
                    seed, score = self._text_to_mask(self._sam3, frame, self.prompt)
                except RuntimeError:
                    return mask, None   # nothing matching in view; keep missing
                self._trackers[name].prime(frame, seed)
                mask = seed
                print(f"ObjectSource: re-acquired on '{name}' (score {score:.2f})",
                      flush=True)
        if not mask.any():
            return mask, None
        self._empty_run[name] = 0
        ys, xs = np.nonzero(mask)
        return mask, (float(xs.mean()), float(ys.mean()))

    def _track_loop(self):
        try:
            seqs = {name: 0 for name in self.cameras}
            while not self._stop.is_set():
                frames, times = {}, {}
                for name in self.cameras:
                    seqs[name], t_capture, frames[name] = \
                        self.feeds[name].wait_next(seqs[name])
                    times[name] = t_capture
                t = float(np.mean(list(times.values())))

                masks, centroids, geometry = {}, {}, {}
                for name in self.cameras:
                    masks[name], centroids[name] = self._track_view(name, frames[name])
                    T = self._anchor(name, frames[name])
                    if T is not None:
                        geometry[name] = (self.feeds[name].camera_matrix,
                                          self.feeds[name].dist_coeffs, T)

                live = None
                if all(centroids[n] is not None for n in self.cameras) \
                        and len(geometry) == len(self.cameras):
                    rays = [pixel_rays(np.array([centroids[n]]), *geometry[n])
                            for n in self.cameras]
                    pts, _ = triangulate_rays(*rays[0], *rays[1])
                    live = pts[0]

                with self._lock:
                    if live is None:
                        self._lost_frames += 1
                        # Motion unknown while unseen: break the static window.
                        self._window_gen += 1
                        self._window_max_area = {n: 0.0 for n in self.cameras}
                    else:
                        self._driver.ingest_live(t, live)
                        areas = {n: float(masks[n].sum()) for n in self.cameras}
                        if self._driver.static_now():
                            for n in self.cameras:
                                self._window_max_area[n] = max(
                                    self._window_max_area[n], areas[n])
                            vis_frac = [areas[n] / self._window_max_area[n]
                                        for n in self.cameras]
                            gate = self._driver.gate_open(vis_frac)
                        else:
                            self._window_gen += 1
                            self._window_max_area = {n: 0.0 for n in self.cameras}
                            gate = False
                    if self.keep_views:
                        _, _, center, sqrtm6, _ = self._driver.serve(t)
                        for n in self.cameras:
                            if n in geometry:
                                self._views[n] = draw_object_overlay(
                                    frames[n].copy(), masks[n], centroids[n],
                                    live, center, sqrtm6, *geometry[n])
                if live is not None and gate:
                    with self._hull_cond:
                        self._hull_job = (self._window_gen, t, masks, geometry)
                        self._hull_cond.notify()
        except Exception as exc:
            self.error = exc

    def _hull_loop(self):
        """Consume gated frame pairs, run the visual hull, and refresh the
        precise channel with the static window's running average — M averaged
        in the linear domain before the sqrt. Never blocks the tick."""
        try:
            window_gen = -1
            centers, Ms = [], []
            while not self._stop.is_set():
                with self._hull_cond:
                    while self._hull_job is None and not self._stop.is_set():
                        self._hull_cond.wait(0.5)
                    if self._stop.is_set():
                        return
                    gen, t, masks, geometry = self._hull_job
                    self._hull_job = None
                if gen != window_gen:
                    window_gen = gen
                    centers, Ms = [], []
                t0 = time.monotonic()
                center, M = hull_estimate(masks, geometry, self.cameras)
                hull_ms = (time.monotonic() - t0) * 1e3
                if center is None:
                    continue
                centers.append(center)
                Ms.append(M)
                with self._lock:
                    self._hull_ms = hull_ms
                    self._driver.ingest_precise(
                        t, np.mean(centers, axis=0),
                        sqrtm_from_cov(np.mean(Ms, axis=0)))
        except Exception as exc:
            self.error = exc
