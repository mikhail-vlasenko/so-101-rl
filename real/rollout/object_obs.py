"""Asynchronous tag-free SAM + dense-stereo object observations for rollouts.

``ObjectSource`` is a consumer of the existing :mod:`real.rollout.frame_bus`;
it never opens a camera and therefore cannot contend with the arm-marker
consumer.  SAM3 remains resident on the GPU after the initial text prompt,
while one SAM2 tracker per view follows the sponge at camera rate.  The mask
centroids provide the held live position and its static gate.

When that gate opens, the newest eligible full-resolution frame pair replaces
any older pending pair.  A separate CPU worker rectifies the images and masks,
runs the frozen Stage 2 StereoSGBM candidate, filters and voxel-downsamples the
cloud, then publishes the shared :class:`src.bps.BPSObsState`.  This work never
blocks ``ArmLoop``.  Empty masks and clouds below the configured validity gate
are ordinary misses; a dead camera, model error, or dense-worker exception is
re-raised by every public read and aborts the rollout.

Both cameras solve the complete two-tag table board.  Dense jobs remain
withheld until their relative pose passes the stereo calibration's known-good
anchor reference and movement limits.  A failed check instructs the operator
to repeat the checkerboard calibration rather than serving geometry from a
moved rig.

Run the non-arm smoke test with the cameras and a still sponge in view:

    python -m real.rollout.object_obs --seconds 5
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from types import MappingProxyType

import cv2
import numpy as np

from real.calib.align_stereo_rig import (
    camera_movement_warning,
    relative_pose_change,
)
from real.calib.calibrate_stereo import load_limits
from real.calib.extrinsics import mat_inv
from real.calib.table_anchor import TableAnchorTracker
from real.marker_spec import TABLE_TAG_IDS
from real.tracking.dense_stereo import (
    DenseStereoConfig,
    StereoPreprocessor,
    T_base_rectified_main,
    cloud_to_bps,
    disparity_range_from_depths,
    disparity_to_cloud,
    load_config,
    sgbm_disparities,
    voxel_downsample,
)
from real.tracking.sam_seg import (
    MaskTracker,
    load_sam3,
    mask_centroid,
    text_to_mask,
)
from real.tracking.shape_dataset import CausalMeanPosition, load_workspace_bounds
from real.vision.detect import make_detector
from real.vision.stereo import pixel_rays, triangulate_rays
from real.vision.stereo_rig import CAMERA_NAMES
from src.bps import BPSConfig, BPSMeasurement, BPSObsState, load_bps_config
from src.shape_obs import MARKER_AGE_CAP_S, ObjectChannelDriver


REPROMPT_AFTER_EMPTY = 15


@dataclass(frozen=True)
class DenseStereoJob:
    """One immutable static-gated pair handed to the CPU worker."""

    capture_t: float
    frames: MappingProxyType
    masks: MappingProxyType
    T_base_main: np.ndarray


@dataclass(frozen=True)
class DenseStereoResult:
    measurement: BPSMeasurement | None
    cloud_base: np.ndarray
    valid_count: int
    valid_fraction: float
    correspondence_rejected_fraction: float
    overall_rejected_fraction: float
    inference_ms: float


@dataclass(frozen=True)
class ObjectSourceStats:
    sam_ms: float
    dense_ms: float
    cloud_age_s: float
    valid_count: int
    valid_fraction: float
    correspondence_rejected_fraction: float
    overall_rejected_fraction: float
    lost_frames: int
    dense_refreshes: int
    dense_misses: int
    rig_movement_mm: float
    rig_movement_deg: float


def _readonly(array: np.ndarray, *, copy: bool) -> np.ndarray:
    output = np.array(array, copy=copy)
    output.setflags(write=False)
    return output


def current_relative_camera_pose(T_base_cameras: dict[str, np.ndarray]) -> np.ndarray:
    """Return ``T_aux_main`` from complete base-frame camera poses."""
    if set(T_base_cameras) != set(CAMERA_NAMES):
        raise ValueError(f"camera poses must contain exactly {CAMERA_NAMES}")
    return mat_inv(T_base_cameras["aux"]) @ T_base_cameras["main"]


def validate_rig_placement(T_base_cameras: dict[str, np.ndarray],
                           preprocessor: StereoPreprocessor) -> tuple[float, float]:
    """Validate the live two-tag rig pose against the calibrated reference."""
    reference = preprocessor.rectification.anchor_reference_T_aux_main
    if reference is None:
        raise RuntimeError(
            "stereo calibration has no table-anchor movement reference; run "
            "`python -m real.calib.align_stereo_rig --stereo-calibration "
            "real/vision/stereo_calibration.yaml "
            "--record-stereo-anchor-reference` with the calibrated rig")
    current = current_relative_camera_pose(T_base_cameras)
    movement_mm, movement_deg = relative_pose_change(current, reference)
    limits = load_limits()
    warning = camera_movement_warning(
        movement_mm, movement_deg,
        limits.camera_movement_warning_translation_mm,
        limits.camera_movement_warning_rotation_deg,
    )
    if warning is not None:
        raise RuntimeError(warning)
    return movement_mm, movement_deg


def workspace_rectified_depths(workspace_xy: tuple[np.ndarray, np.ndarray],
                               workspace_z_m: tuple[float, float],
                               T_base_rectified: np.ndarray) -> np.ndarray:
    """Rectified-main depths of the complete configured object workspace."""
    low, high = (np.asarray(bound, dtype=np.float64) for bound in workspace_xy)
    corners_base = np.array([
        [x, y, z]
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in workspace_z_m
    ], dtype=np.float64)
    T_rectified_base = mat_inv(T_base_rectified)
    corners_rectified = (
        corners_base @ T_rectified_base[:3, :3].T
        + T_rectified_base[:3, 3]
    )
    depths = corners_rectified[:, 2]
    if np.any(depths <= 0.0):
        raise RuntimeError("configured object workspace crosses behind the main camera")
    return depths


class DenseStereoProcessor:
    """Frozen rectification/matcher/filter/BPS path used by the worker."""

    def __init__(self, workspace_xy: tuple[np.ndarray, np.ndarray],
                 dense_config: DenseStereoConfig | None = None,
                 bps_config: BPSConfig | None = None,
                 preprocessor: StereoPreprocessor | None = None):
        self.config = load_config() if dense_config is None else dense_config
        self.bps_config = load_bps_config() if bps_config is None else bps_config
        self.preprocessor = (
            StereoPreprocessor(config=self.config)
            if preprocessor is None else preprocessor
        )
        self.workspace_xy = tuple(
            np.asarray(bound, dtype=np.float64).copy() for bound in workspace_xy)
        self._disparity_range: tuple[int, int] | None = None

    def configure_for_pose(self, T_base_main: np.ndarray) -> tuple[int, int]:
        T_base_rectified = T_base_rectified_main(
            T_base_main, self.preprocessor.rectification)
        depths = workspace_rectified_depths(
            self.workspace_xy, self.config.workspace_z_m, T_base_rectified)
        self._disparity_range = disparity_range_from_depths(
            depths, self.preprocessor.rectification, self.config)
        return self._disparity_range

    def process(self, job: DenseStereoJob) -> DenseStereoResult:
        if self._disparity_range is None:
            raise RuntimeError("dense processor used before rig placement validation")
        started = time.perf_counter()
        images = {
            name: self.preprocessor.rectify_image(name, job.frames[name])
            for name in CAMERA_NAMES
        }
        masks = {
            name: self.preprocessor.rectify_mask(name, job.masks[name])
            for name in CAMERA_NAMES
        }
        if any(int(masks[name].sum()) < self.config.min_mask_area_px
               for name in CAMERA_NAMES):
            return self._miss(
                np.empty((0, 3)), 0, 0.0, 1.0, started)

        minimum, count = self._disparity_range
        disparity, reverse = sgbm_disparities(
            images["main"], images["aux"],
            self.config.frozen_sgbm_candidate, minimum, count)
        cloud = disparity_to_cloud(
            disparity, reverse, masks["main"], masks["aux"],
            self.preprocessor.geometry.Q,
            T_base_rectified_main(
                job.T_base_main, self.preprocessor.rectification),
            self.config, self.workspace_xy)
        filtered_count = int(cloud.points_base.shape[0])
        valid_fraction = float(np.clip(
            filtered_count / max(cloud.left_mask_pixels, 1), 0.0, 1.0))
        correspondence_rejected_fraction = float(np.clip(
            cloud.correspondence_rejected / max(cloud.left_mask_pixels, 1),
            0.0, 1.0))
        points, _ = voxel_downsample(
            cloud.points_base, cloud.confidence, self.config.voxel_size_m)
        valid_count = int(points.shape[0])
        overall_rejected_fraction = 1.0 - valid_fraction
        if valid_count < self.config.min_deployment_valid_points:
            return self._miss(
                points, valid_count, correspondence_rejected_fraction,
                overall_rejected_fraction, started, valid_fraction)

        measurement = cloud_to_bps(points, valid_fraction, self.bps_config)
        measurement = BPSMeasurement(
            distances=_readonly(measurement.distances, copy=True),
            center_base=_readonly(measurement.center_base, copy=True),
            valid_fraction=measurement.valid_fraction,
        )
        return DenseStereoResult(
            measurement=measurement,
            cloud_base=_readonly(points, copy=True),
            valid_count=valid_count,
            valid_fraction=valid_fraction,
            correspondence_rejected_fraction=correspondence_rejected_fraction,
            overall_rejected_fraction=overall_rejected_fraction,
            inference_ms=(time.perf_counter() - started) * 1e3,
        )

    @staticmethod
    def _miss(points: np.ndarray, valid_count: int,
              correspondence_rejected_fraction: float,
              overall_rejected_fraction: float, started: float,
              valid_fraction: float = 0.0) -> DenseStereoResult:
        return DenseStereoResult(
            measurement=None,
            cloud_base=_readonly(points, copy=True),
            valid_count=valid_count,
            valid_fraction=valid_fraction,
            correspondence_rejected_fraction=correspondence_rejected_fraction,
            overall_rejected_fraction=overall_rejected_fraction,
            inference_ms=(time.perf_counter() - started) * 1e3,
        )


class ObjectSource:
    """FrameBus consumer serving the live centroid and held BPS observation."""

    def __init__(self, feeds, workspace_xy: tuple[np.ndarray, np.ndarray],
                 prompt: str = "sponge", sam2_model: str = "tiny",
                 family: str = "apriltag"):
        if set(feeds) != set(CAMERA_NAMES):
            raise ValueError(f"feeds must contain exactly {CAMERA_NAMES}")
        self.feeds = feeds
        self.prompt = prompt
        self.sam2_model = sam2_model
        self.detector = make_detector(family)
        self.processor = DenseStereoProcessor(workspace_xy)
        self._anchors: dict[str, TableAnchorTracker] = {}
        self._trackers: dict[str, MaskTracker] = {}
        self._sam3 = None
        self._driver = ObjectChannelDriver()
        self._static_filter = CausalMeanPosition(
            self.processor.config.static_mean_window_s)
        self._bps_state = BPSObsState()
        self._state_lock = threading.Lock()
        self._processor_lock = threading.Lock()
        self._stop = threading.Event()
        self._job_condition = threading.Condition()
        self._job: tuple[int, DenseStereoJob] | None = None
        self._job_generation = 0
        self._executor: ThreadPoolExecutor | None = None
        self._futures: tuple[Future, Future] = ()
        self._rig_validated = False
        self._paired_anchor_updates = 0
        self._window_max_area = {name: 0.0 for name in CAMERA_NAMES}
        self._empty_run = {name: 0 for name in CAMERA_NAMES}
        self._sam_ms = {name: float("nan") for name in CAMERA_NAMES}
        self._dense_ms = float("nan")
        self._valid_count = 0
        self._valid_fraction = 0.0
        self._correspondence_rejected_fraction = 0.0
        self._overall_rejected_fraction = 1.0
        self._lost_frames = 0
        self._dense_refreshes = 0
        self._dense_misses = 0
        self._rig_movement_mm = float("nan")
        self._rig_movement_deg = float("nan")
        self._last_bps_capture_t = -np.inf
        self._latest_cloud = _readonly(np.empty((0, 3)), copy=False)

    def start(self, warmup_timeout_s: float = 60.0) -> None:
        """Start both workers and wait for validated live and BPS channels."""
        for name in CAMERA_NAMES:
            feed = self.feeds[name]
            if feed.camera_matrix is None:
                raise RuntimeError(f"start the '{name}' CameraFeed before ObjectSource")
            self._anchors[name] = TableAnchorTracker(
                feed.camera_matrix, feed.dist_coeffs)

        print(f"ObjectSource: prompting SAM3 with {self.prompt!r} ...", flush=True)
        if self._sam3 is None:
            self._sam3 = load_sam3()
        frames = {}
        masks = {}
        for name in CAMERA_NAMES:
            _, _, frames[name] = self.feeds[name].latest()
            masks[name], score = text_to_mask(
                self._sam3, frames[name], self.prompt)
            print(
                f"  {name}: score {score:.2f}, area {int(masks[name].sum())} px",
                flush=True,
            )

        # Do not load/prime either SAM 2 tracker until both prompt masks are
        # available. An interactive retry after one camera misses therefore
        # reuses the resident SAM 3 model and leaves no half-started worker.
        for name in CAMERA_NAMES:
            tracker = MaskTracker(self.sam2_model)
            tracker.prime(frames[name], masks[name])
            self._trackers[name] = tracker

        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="object-source")
        self._futures = (
            self._executor.submit(self._track_loop),
            self._executor.submit(self._dense_loop),
        )

        self.prepare_episode(warmup_timeout_s)

    def prepare_episode(self, timeout_s: float = 60.0) -> tuple[float, float]:
        """Revalidate the live rig and wait for a post-validation BPS cloud.

        Each call requires a fresh run of accepted, paired table-board solves;
        a placement that passed earlier in the process is never reused for a
        new episode.  Dense jobs are withheld while the check runs, and the
        ready gate only accepts a cloud captured after the new validation.
        Returns the measured relative-camera movement in ``(mm, degrees)``.
        """
        self._raise_worker_errors()
        required_pairs = load_limits().min_detected_pairs
        with self._state_lock:
            initial_pairs = self._paired_anchor_updates
            self._rig_validated = False
        target_pairs = initial_pairs + required_pairs
        deadline = time.monotonic() + timeout_s

        poses = None
        while poses is None:
            self._raise_worker_errors()
            with self._state_lock:
                paired = self._paired_anchor_updates
                if paired >= target_pairs:
                    current = {
                        name: self._anchors[name].value()
                        for name in CAMERA_NAMES
                        if self._anchors[name].seeded
                    }
                    if len(current) == len(CAMERA_NAMES):
                        poses = current
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"stereo rig did not produce {required_pairs} new paired "
                    f"table-anchor solves within {timeout_s:.0f} s "
                    f"({paired - initial_pairs}/{required_pairs}); keep both "
                    "complete table tags visible in both cameras")
            if poses is None:
                time.sleep(0.02)

        movement = validate_rig_placement(poses, self.processor.preprocessor)
        with self._processor_lock:
            disparity_range = self.processor.configure_for_pose(poses["main"])
        validated_t = time.monotonic()
        with self._state_lock:
            self._rig_movement_mm, self._rig_movement_deg = movement
            self._rig_validated = True
        print(
            "ObjectSource: episode stereo rig placement PASS "
            f"({movement[0]:.2f} mm / {movement[1]:.3f} deg), "
            f"disparity [{disparity_range[0]}, {sum(disparity_range)})",
            flush=True,
        )

        while True:
            self._raise_worker_errors()
            (live, live_age), bps = self.object_obs()
            with self._state_lock:
                bps_capture_t = self._last_bps_capture_t
            if (live_age < 0.5 and bps.age_s < MARKER_AGE_CAP_S
                    and bps_capture_t >= validated_t):
                return movement
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"object source not episode-ready within {timeout_s:.0f} s "
                    f"(live_age={live_age:.2f} s, "
                    f"cloud_age={bps.age_s:.2f} s); keep both complete table tags "
                    "and the still, unoccluded sponge visible in both cameras")
            time.sleep(0.02)

    def stop(self) -> None:
        self._stop.set()
        with self._job_condition:
            self._job_condition.notify_all()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.detector.close()

    def object_obs(self):
        """Return ``((live, live_age), BPSObservation)`` at this instant."""
        self._raise_worker_errors()
        now = time.monotonic()
        with self._state_lock:
            return self._driver.serve(now), self._bps_state.serve(now)

    def stats(self) -> ObjectSourceStats:
        """Immutable latency, validity, loss, refresh and rig telemetry."""
        self._raise_worker_errors()
        now = time.monotonic()
        with self._state_lock:
            sam_values = np.asarray(tuple(self._sam_ms.values()), dtype=float)
            sam_ms = (float(np.nansum(sam_values))
                      if np.any(np.isfinite(sam_values)) else float("nan"))
            cloud_age = self._bps_state.serve(now).age_s
            return ObjectSourceStats(
                sam_ms=sam_ms,
                dense_ms=self._dense_ms,
                cloud_age_s=cloud_age,
                valid_count=self._valid_count,
                valid_fraction=self._valid_fraction,
                correspondence_rejected_fraction=(
                    self._correspondence_rejected_fraction),
                overall_rejected_fraction=self._overall_rejected_fraction,
                lost_frames=self._lost_frames,
                dense_refreshes=self._dense_refreshes,
                dense_misses=self._dense_misses,
                rig_movement_mm=self._rig_movement_mm,
                rig_movement_deg=self._rig_movement_deg,
            )

    def latest_cloud(self) -> np.ndarray:
        """Copy of the latest filtered voxel cloud for diagnostics/viewers."""
        self._raise_worker_errors()
        with self._state_lock:
            return self._latest_cloud.copy()

    def _raise_worker_errors(self) -> None:
        for future in self._futures:
            if future.done():
                future.result()

    def _track_view(self, name: str, frame: np.ndarray):
        started = time.perf_counter()
        mask = self._trackers[name].track(frame)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        if not mask.any():
            self._empty_run[name] += 1
            if self._empty_run[name] >= REPROMPT_AFTER_EMPTY:
                self._empty_run[name] = 0
                try:
                    mask, score = text_to_mask(self._sam3, frame, self.prompt)
                except RuntimeError:
                    return mask, None, elapsed_ms
                self._trackers[name].prime(frame, mask)
                print(
                    f"ObjectSource: re-acquired on '{name}' (score {score:.2f})",
                    flush=True)
        centroid = mask_centroid(mask)
        if centroid is not None:
            self._empty_run[name] = 0
        return mask, centroid, elapsed_ms

    def _track_loop(self) -> None:
        sequences = {name: 0 for name in CAMERA_NAMES}
        while not self._stop.is_set():
            # Wait for both feeds to advance, then take the newest pair so a
            # slower consumer skips backlog instead of making stereo stale.
            for name in CAMERA_NAMES:
                sequences[name], _, _ = self.feeds[name].wait_next(sequences[name])
            snapshots = {name: self.feeds[name].latest() for name in CAMERA_NAMES}
            sequences = {name: snapshots[name][0] for name in CAMERA_NAMES}
            capture_t = float(np.mean([snapshots[name][1] for name in CAMERA_NAMES]))
            frames = {name: snapshots[name][2] for name in CAMERA_NAMES}

            masks = {}
            centroids = {}
            detections = {}
            sam_ms = {}
            for name in CAMERA_NAMES:
                masks[name], centroids[name], sam_ms[name] = self._track_view(
                    name, frames[name])
                gray = cv2.cvtColor(frames[name], cv2.COLOR_BGR2GRAY)
                detections[name] = {
                    detection.id: detection
                    for detection in self.detector.detect(gray)
                    if detection.id in TABLE_TAG_IDS
                }
            with self._state_lock:
                anchor_updates = {
                    name: self._anchors[name].observe(detections[name])
                    for name in CAMERA_NAMES
                }
                poses = {
                    name: self._anchors[name].value()
                    for name in CAMERA_NAMES
                    if self._anchors[name].seeded
                }
                if all(anchor_updates.values()):
                    self._paired_anchor_updates += 1

            live = None
            if (all(centroids[name] is not None for name in CAMERA_NAMES)
                    and len(poses) == len(CAMERA_NAMES)):
                rays = [
                    pixel_rays(
                        np.asarray([centroids[name]]),
                        self.feeds[name].camera_matrix,
                        self.feeds[name].dist_coeffs,
                        poses[name],
                    )
                    for name in CAMERA_NAMES
                ]
                points, _ = triangulate_rays(*rays[0], *rays[1])
                live = points[0]

            queue_dense = False
            with self._state_lock:
                self._sam_ms = sam_ms
                if live is None:
                    self._lost_frames += 1
                    self._static_filter.clear()
                    self._window_max_area = {name: 0.0 for name in CAMERA_NAMES}
                else:
                    gate_point = self._static_filter.update(capture_t, live)
                    self._driver.ingest_live(
                        capture_t, live, gate_point=gate_point)
                    areas = {name: float(masks[name].sum()) for name in CAMERA_NAMES}
                    if self._driver.static_now():
                        for name in CAMERA_NAMES:
                            self._window_max_area[name] = max(
                                self._window_max_area[name], areas[name])
                        visible_fractions = [
                            areas[name] / max(self._window_max_area[name], 1.0)
                            for name in CAMERA_NAMES
                        ]
                        queue_dense = (
                            self._rig_validated
                            and self._driver.gate_open(visible_fractions)
                            and all(areas[name] >= self.processor.config.min_mask_area_px
                                    for name in CAMERA_NAMES)
                        )
                    else:
                        self._window_max_area = {
                            name: 0.0 for name in CAMERA_NAMES}
            if queue_dense:
                self._queue_dense_job(capture_t, frames, masks, poses["main"])

    def _queue_dense_job(self, capture_t: float, frames: dict[str, np.ndarray],
                         masks: dict[str, np.ndarray],
                         T_base_main: np.ndarray) -> None:
        job = DenseStereoJob(
            capture_t=capture_t,
            frames=MappingProxyType({
                name: _readonly(frames[name], copy=True) for name in CAMERA_NAMES
            }),
            masks=MappingProxyType({
                name: _readonly(masks[name], copy=True) for name in CAMERA_NAMES
            }),
            T_base_main=_readonly(T_base_main, copy=True),
        )
        with self._job_condition:
            self._job_generation += 1
            self._job = (self._job_generation, job)
            self._job_condition.notify()

    def _dense_loop(self) -> None:
        consumed_generation = 0
        while not self._stop.is_set():
            with self._job_condition:
                while (not self._stop.is_set()
                       and (self._job is None
                            or self._job[0] == consumed_generation)):
                    self._job_condition.wait(0.5)
                if self._stop.is_set():
                    return
                consumed_generation, job = self._job
            with self._processor_lock:
                result = self.processor.process(job)
            with self._state_lock:
                self._dense_ms = result.inference_ms
                self._valid_count = result.valid_count
                self._valid_fraction = result.valid_fraction
                self._correspondence_rejected_fraction = (
                    result.correspondence_rejected_fraction)
                self._overall_rejected_fraction = result.overall_rejected_fraction
                self._latest_cloud = result.cloud_base
                if result.measurement is None:
                    self._dense_misses += 1
                else:
                    self._bps_state.ingest(job.capture_t, result.measurement)
                    self._last_bps_capture_t = job.capture_t
                    self._dense_refreshes += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-arm live smoke test for the Stage 5 object source")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Seconds to sample after startup warmup")
    parser.add_argument("--prompt", default="sponge", help="SAM3 text prompt")
    parser.add_argument("--sam2-model", choices=("tiny", "base+"), default="tiny")
    return parser.parse_args()


def main() -> int:
    from real.rollout.frame_bus import FrameBus

    args = parse_args()
    if args.seconds <= 0.0:
        raise ValueError("--seconds must be positive")
    bus = FrameBus(CAMERA_NAMES)
    source = ObjectSource(
        bus.feeds, load_workspace_bounds(), args.prompt, args.sam2_model)
    try:
        bus.start()
        source.start()
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            (live, live_age), bps = source.object_obs()
            stats = source.stats()
            print(
                f"live=({live[0]:+.3f},{live[1]:+.3f},{live[2]:+.3f}) "
                f"ages={live_age:.2f}/{bps.age_s:.2f}s "
                f"points={stats.valid_count} valid={stats.valid_fraction:.1%} "
                f"reject={stats.correspondence_rejected_fraction:.1%}/"
                f"{stats.overall_rejected_fraction:.1%} "
                f"sam={stats.sam_ms:.1f}ms dense={stats.dense_ms:.1f}ms "
                f"refresh/miss={stats.dense_refreshes}/{stats.dense_misses}",
                flush=True,
            )
            time.sleep(0.25)
        stats = source.stats()
        if stats.dense_refreshes == 0:
            raise RuntimeError("live smoke test produced no dense BPS refresh")
    finally:
        source.stop()
        bus.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
