"""Stage 2 metrics and viewer panels must describe the actual cloud."""

import cv2
import numpy as np

from real.tracking.eval_dense_stereo import (
    _cache_image_path,
    box_surface_coverage,
    closest_box_surface,
    visibility_sequence_report,
)
from real.tracking.view_dense_stereo import confidence_map, diagnostic_mosaic


def test_closest_box_surface_handles_inside_and_outside_points():
    center = np.array([0.2, 0.0, 0.03])
    rotation = np.eye(3)
    half_extents = np.array([0.03, 0.02, 0.01])
    points = np.array([
        [0.24, 0.0, 0.03],
        [0.2, 0.0, 0.03],
    ])

    closest, faces = closest_box_surface(
        points, center, rotation, half_extents)

    np.testing.assert_allclose(closest[0], [0.23, 0.0, 0.03])
    np.testing.assert_allclose(closest[1], [0.2, 0.0, 0.04])
    np.testing.assert_array_equal(faces, [1, 5])


def test_surface_coverage_reports_physical_area_of_one_complete_face():
    half_extents = np.array([0.02, 0.01, 0.005])
    cell = 0.01
    xs = np.array([-0.015, -0.005, 0.005, 0.015])
    ys = np.array([-0.005, 0.005])
    points = np.array([[x, y, half_extents[2]] for x in xs for y in ys])

    fraction, area = box_surface_coverage(
        points, np.zeros(3), np.eye(3), half_extents, cell)

    face_area = 4.0 * half_extents[0] * half_extents[1]
    total_area = 8.0 * (
        half_extents[0] * half_extents[1]
        + half_extents[0] * half_extents[2]
        + half_extents[1] * half_extents[2])
    np.testing.assert_allclose(area, face_area)
    np.testing.assert_allclose(fraction, face_area / total_area)


def test_confidence_map_requires_masked_left_right_consistency():
    disparity = np.full((3, 6), 2.0, dtype=np.float32)
    reverse = np.full((3, 6), -2.0, dtype=np.float32)
    left_mask = np.zeros((3, 6), dtype=bool)
    right_mask = np.zeros((3, 6), dtype=bool)
    left_mask[1, 3:5] = True
    right_mask[1, 1:3] = True
    reverse[1, 2] = -1.0

    confidence = confidence_map(
        disparity, reverse, left_mask, right_mask, max_error_px=1.5)

    assert confidence[1, 3] == 1.0
    np.testing.assert_allclose(confidence[1, 4], 1.0 - 1.0 / 1.5)
    assert np.count_nonzero(confidence) == 2


def test_diagnostic_mosaic_contains_six_processed_size_panels():
    height, width = 32, 48
    images = {
        "main": np.full((height, width, 3), 80, dtype=np.uint8),
        "aux": np.full((height, width, 3), 100, dtype=np.uint8),
    }
    masks = {
        "main": np.ones((height, width), dtype=bool),
        "aux": np.ones((height, width), dtype=bool),
    }
    points = np.array([[0.2, 0.0, 0.01]])

    mosaic = diagnostic_mosaic(
        images, masks, np.full((height, width), 100.0),
        np.ones((height, width)), points, np.ones(1),
        np.array([0.2, 0.0, 0.0125]), np.eye(3),
        np.array([0.03, 0.02, 0.0125]), (80, 64),
        (np.array([0.1, -0.1]), np.array([0.3, 0.1])), 42)

    assert mosaic.shape == (3 * height, 2 * width, 3)
    assert np.any(mosaic)


def test_visibility_sequence_ignores_initial_absence_and_measures_reacquisition(
        tmp_path):
    records = [
        {"k": index + 1, "t": {"main": float(index), "aux": float(index)}}
        for index in range(6)
    ]
    visible = [False, False, True, True, False, True]
    for record, is_visible in zip(records, visible):
        for camera in ("main", "aux"):
            path = _cache_image_path(
                tmp_path, "masks", record, camera, "png")
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.full((4, 4), 255 if is_visible else 0, dtype=np.uint8)
            assert cv2.imwrite(str(path), mask)

    report = visibility_sequence_report(
        tmp_path, records, [False, False, False, True, False, True],
        min_mask_area_px=4)

    assert report["disappearance_events"] == 1
    assert report["reacquisition_events"] == 1
    assert report["complete_disappearance_duration_s"]["max"] == 1.0
    assert report["static_refresh_delay_after_reacquisition_s"]["max"] == 0.0
    assert report["held_cloud_age_s"]["max"] == 1.0
