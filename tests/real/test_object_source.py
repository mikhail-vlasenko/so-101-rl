"""Stage 5 online dense worker contracts without camera/GPU hardware."""

from concurrent.futures import Future
from dataclasses import replace
import threading
import time
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from real.calib.calibrate_stereo import StereoRectification
from real.rollout.object_obs import (
    DenseStereoJob,
    DenseStereoProcessor,
    ObjectSource,
    current_relative_camera_pose,
    validate_rig_placement,
    workspace_rectified_depths,
)
from real.tracking.dense_stereo import (
    CloudResult,
    DenseStereoConfig,
    FastFoundationCandidate,
    ProcessingGeometry,
    SGBMCandidate,
)
from real.tracking.sam_seg import SAMPromptNoMatchError


def _config():
    candidate = SGBMCandidate(3, 5, 50, 1, 1)
    return DenseStereoConfig(
        processing_scale=0.5,
        processing_height_multiple=32,
        tag_inpaint_dilate_px=0,
        tag_inpaint_radius_px=2,
        static_mean_window_s=0.15,
        held_out_fraction=0.25,
        min_static_window_frames=5,
        development_frame_stride=8,
        disparity_margin_px=4,
        lr_max_error_px=1.5,
        point_sample_count=4,
        voxel_size_m=0.01,
        workspace_z_m=(-1.0, 10.0),
        depth_mad_scale=4.0,
        depth_mad_floor_m=0.01,
        min_mask_area_px=10,
        sgbm_candidates=(candidate,),
        frozen_sgbm_candidate=candidate,
        min_deployment_valid_points=2,
        fast_max_disp_multiple=32,
        fast_candidates=(FastFoundationCandidate("23-36-37", 8),),
    )


def _rectification():
    main = np.array([[100.0, 0.0, 4.0, 0.0],
                     [0.0, 100.0, 3.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]])
    aux = main.copy()
    aux[0, 3] = -10.0
    return StereoRectification(
        image_size=(8, 6),
        T_aux_main=np.eye(4),
        rotations={"main": np.eye(3), "aux": np.eye(3)},
        projections={"main": main, "aux": aux},
        Q=np.eye(4),
        valid_rois={"main": (0, 0, 8, 6), "aux": (0, 0, 8, 6)},
        anchor_reference_T_aux_main=None,
    )


class IdentityPreprocessor:
    def __init__(self, rectification=None):
        self.rectification = _rectification() if rectification is None else rectification
        self.geometry = ProcessingGeometry((2, 2), 0, np.eye(4))

    def rectify_image(self, name, image):
        return image

    def rectify_mask(self, name, mask):
        return mask


def _job():
    frames = {name: np.zeros((2, 2, 3), dtype=np.uint8)
              for name in ("main", "aux")}
    masks = {name: np.ones((2, 2), dtype=bool)
             for name in ("main", "aux")}
    return DenseStereoJob(
        capture_t=1.0,
        frames=MappingProxyType(frames),
        masks=MappingProxyType(masks),
        T_base_main=np.eye(4),
    )


def test_start_reuses_sam3_and_builds_no_trackers_until_both_masks_exist(
        monkeypatch):
    class Feed:
        camera_matrix = np.eye(3)
        dist_coeffs = np.zeros(5)

        def latest(self):
            return 1, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)

    class Tracker:
        def __init__(self, model):
            self.model = model

        def prime(self, frame, mask):
            pass

    source = ObjectSource.__new__(ObjectSource)
    source.feeds = {name: Feed() for name in ("main", "aux")}
    source.prompt = "sponge"
    source.sam2_model = "tiny"
    source._anchors = {}
    source._trackers = {}
    source._sam3 = None
    source._track_loop = lambda: None
    source._dense_loop = lambda: None
    source.prepare_episode = lambda timeout: None

    loaded = []
    detections = iter((
        (np.ones((2, 2), dtype=bool), 0.9),
        None,
        (np.ones((2, 2), dtype=bool), 0.8),
        (np.ones((2, 2), dtype=bool), 0.7),
    ))

    def detect(sam3, frame, prompt):
        found = next(detections)
        if found is None:
            raise SAMPromptNoMatchError("no match")
        return found

    monkeypatch.setattr(
        "real.rollout.object_obs.TableAnchorTracker", lambda *args: object())
    monkeypatch.setattr(
        "real.rollout.object_obs.load_sam3", lambda: loaded.append(True) or object())
    monkeypatch.setattr("real.rollout.object_obs.text_to_mask", detect)
    monkeypatch.setattr("real.rollout.object_obs.MaskTracker", Tracker)

    with pytest.raises(SAMPromptNoMatchError, match="no match"):
        source.start()
    assert source._trackers == {}

    source.start()
    source._executor.shutdown(wait=True)

    assert loaded == [True]
    assert set(source._trackers) == {"main", "aux"}


def test_relative_pose_uses_complete_main_and_aux_transforms():
    main = np.eye(4)
    aux = np.eye(4)
    aux[0, 3] = 0.11

    relative = current_relative_camera_pose({"main": main, "aux": aux})

    assert np.isclose(relative[0, 3], -0.11)
    with pytest.raises(ValueError, match="exactly"):
        current_relative_camera_pose({"main": main})


def test_rig_validation_accepts_reference_and_rejects_movement():
    main = np.eye(4)
    aux = np.eye(4)
    aux[0, 3] = 0.11
    poses = {"main": main, "aux": aux}
    reference = current_relative_camera_pose(poses)
    base = _rectification()
    rectification = StereoRectification(
        image_size=base.image_size,
        T_aux_main=base.T_aux_main,
        rotations=base.rotations,
        projections=base.projections,
        Q=base.Q,
        valid_rois=base.valid_rois,
        anchor_reference_T_aux_main=reference,
    )
    preprocessor = IdentityPreprocessor(rectification)

    movement = validate_rig_placement(poses, preprocessor)
    assert movement == pytest.approx((0.0, 0.0))

    moved_aux = aux.copy()
    moved_aux[1, 3] = 0.002
    with pytest.raises(RuntimeError, match="cameras moved.*calibrate_stereo"):
        validate_rig_placement({"main": main, "aux": moved_aux}, preprocessor)


def test_workspace_depths_use_base_to_rectified_transform():
    T_base_rectified = np.eye(4)
    T_base_rectified[2, 3] = -0.5

    depths = workspace_rectified_depths(
        (np.array([0.1, -0.1]), np.array([0.3, 0.1])),
        (0.0, 0.2), T_base_rectified)

    np.testing.assert_array_equal(np.unique(depths), (0.5, 0.7))


def test_processor_publishes_immutable_bps_and_validity(monkeypatch):
    config = replace(
        _config(), workspace_z_m=(1.0, 2.0), min_mask_area_px=1,
        min_deployment_valid_points=2)
    processor = DenseStereoProcessor(
        (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        dense_config=config, preprocessor=IdentityPreprocessor())
    processor.configure_for_pose(np.eye(4))
    points = np.array([[0.00, 0.00, 0.02],
                       [0.02, 0.00, 0.02],
                       [0.00, 0.02, 0.02]])

    monkeypatch.setattr(
        "real.rollout.object_obs.sgbm_disparities",
        lambda *args: (np.ones((2, 2), dtype=np.float32),
                       -np.ones((2, 2), dtype=np.float32)))
    monkeypatch.setattr(
        "real.rollout.object_obs.disparity_to_cloud",
        lambda *args: CloudResult(points, np.ones(3, dtype=np.float32), 4, 1))

    result = processor.process(_job())

    assert result.measurement is not None
    assert result.valid_count == 3
    assert result.valid_fraction == pytest.approx(0.75)
    assert result.correspondence_rejected_fraction == pytest.approx(0.25)
    assert result.overall_rejected_fraction == pytest.approx(0.25)
    assert not result.cloud_base.flags.writeable
    assert not result.measurement.distances.flags.writeable
    assert not result.measurement.center_base.flags.writeable


def test_processor_treats_too_few_points_as_measurement_miss(monkeypatch):
    config = replace(
        _config(), workspace_z_m=(1.0, 2.0), min_mask_area_px=1,
        min_deployment_valid_points=2)
    processor = DenseStereoProcessor(
        (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        dense_config=config, preprocessor=IdentityPreprocessor())
    processor.configure_for_pose(np.eye(4))
    point = np.array([[0.0, 0.0, 0.02]])
    monkeypatch.setattr(
        "real.rollout.object_obs.sgbm_disparities",
        lambda *args: (np.ones((2, 2), dtype=np.float32),
                       -np.ones((2, 2), dtype=np.float32)))
    monkeypatch.setattr(
        "real.rollout.object_obs.disparity_to_cloud",
        lambda *args: CloudResult(point, np.ones(1, dtype=np.float32), 4, 0))

    result = processor.process(_job())

    assert result.measurement is None
    assert result.valid_count == 1
    assert result.valid_fraction == pytest.approx(0.25)


def test_pending_dense_job_is_replaced_by_newest_pair():
    source = ObjectSource.__new__(ObjectSource)
    source._active = threading.Event()
    source._active.set()
    source._job_condition = threading.Condition()
    source._job_generation = 0
    source._job = None
    frames = {name: np.zeros((2, 2, 3), dtype=np.uint8)
              for name in ("main", "aux")}
    masks = {name: np.ones((2, 2), dtype=bool)
             for name in ("main", "aux")}

    source._queue_dense_job(1.0, frames, masks, np.eye(4))
    source._queue_dense_job(2.0, frames, masks, np.eye(4))

    generation, job = source._job
    assert generation == 2
    assert job.capture_t == 2.0
    assert not job.frames["main"].flags.writeable


def test_pause_discards_dense_work_and_requires_fresh_rig_validation():
    source = ObjectSource.__new__(ObjectSource)
    source._active = threading.Event()
    source._active.set()
    source._stop = threading.Event()
    source._state_lock = threading.Lock()
    source._rig_validated = True
    source._job_condition = threading.Condition()
    source._job_generation = 1
    source._job = (1, _job())

    source.pause()

    assert not source._active.is_set()
    assert not source._rig_validated
    assert source._job is None

    frames = {name: np.zeros((2, 2, 3), dtype=np.uint8)
              for name in ("main", "aux")}
    masks = {name: np.ones((2, 2), dtype=bool)
             for name in ("main", "aux")}
    source._queue_dense_job(2.0, frames, masks, np.eye(4))
    assert source._job is None

    source.resume()
    source._queue_dense_job(3.0, frames, masks, np.eye(4))
    assert source._job[1].capture_t == 3.0


def test_worker_exception_is_rethrown_by_public_reads():
    source = ObjectSource.__new__(ObjectSource)
    failed = Future()
    failed.set_exception(RuntimeError("dense worker died"))
    source._futures = (failed,)

    with pytest.raises(RuntimeError, match="dense worker died"):
        source.object_obs()


def test_prepare_episode_revalidates_fresh_pairs_and_waits_for_new_cloud(
        monkeypatch):
    class Anchor:
        seeded = True

        def value(self):
            return np.eye(4)

    class Processor:
        preprocessor = object()

        def __init__(self):
            self.configured = 0

        def configure_for_pose(self, pose):
            self.configured += 1
            return 3, 4

    source = ObjectSource.__new__(ObjectSource)
    source._futures = ()
    source._state_lock = threading.Lock()
    source._processor_lock = threading.Lock()
    source._paired_anchor_updates = 0
    source._rig_validated = False
    source._anchors = {name: Anchor() for name in ("main", "aux")}
    source.processor = Processor()
    source._rig_movement_mm = float("nan")
    source._rig_movement_deg = float("nan")
    source._last_bps_capture_t = -np.inf
    source.object_obs = lambda: (
        (np.zeros(3), 0.0), SimpleNamespace(age_s=0.0))

    validations = []
    monkeypatch.setattr(
        "real.rollout.object_obs.load_limits",
        lambda: SimpleNamespace(min_detected_pairs=1))
    monkeypatch.setattr(
        "real.rollout.object_obs.validate_rig_placement",
        lambda poses, preprocessor: validations.append(poses) or (0.1, 0.2))

    def publish_new_episode_evidence():
        time.sleep(0.01)
        with source._state_lock:
            source._paired_anchor_updates += 1
        while True:
            with source._state_lock:
                if source._rig_validated:
                    source._last_bps_capture_t = time.monotonic()
                    return
            time.sleep(0.001)

    for _ in range(2):
        worker = threading.Thread(target=publish_new_episode_evidence)
        worker.start()
        movement = source.prepare_episode(timeout_s=1.0)
        worker.join()
        assert movement == (0.1, 0.2)

    assert len(validations) == 2
    assert source.processor.configured == 2
    assert source._paired_anchor_updates == 2
