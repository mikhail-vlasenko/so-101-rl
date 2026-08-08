"""Mechanical stereo-alignment metric and workspace-projection contracts."""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from real.calib.align_stereo_rig import (
    AlignmentLimits,
    CameraCoverage,
    _current_measurement,
    _viewer_lines,
    coverage_guidance,
    evaluate_alignment,
    load_alignment_limits,
    load_workspace_corners,
    project_workspace,
)
from real.vision.overlay import GREEN, RED


def _limits() -> AlignmentLimits:
    return AlignmentLimits(
        capture_frames=100,
        sample_window=30,
        min_samples=20,
        baseline_min_m=0.10,
        baseline_max_m=0.12,
        max_height_delta_m=0.005,
        max_relative_axis_deg=2.0,
        min_workspace_margin_px=50.0,
        workspace_z_low_m=0.0,
        workspace_z_high_m=0.15,
        min_pair_detection_fraction=0.8,
        max_translation_spread_mean_mm=0.5,
        max_rotation_spread_mean_deg=0.1,
    )


def _coverage(margins=(80.0, 80.0, 80.0, 80.0)) -> CameraCoverage:
    return CameraCoverage(
        pixels=np.zeros((8, 2)),
        margins_px=np.asarray(margins, dtype=np.float64),
        all_in_front=True,
    )


def test_configured_parallel_rig_passes_every_gate():
    main = np.eye(4)
    aux = np.eye(4)
    aux[:3, 3] = (0.11, 0.0, 0.002)
    report = evaluate_alignment(
        {"main": main, "aux": aux},
        {"main": _coverage(), "aux": _coverage()},
        {"main": (0.2, 0.04), "aux": (0.3, 0.05)},
        pair_detection_fraction=1.0,
        sample_count=30,
        limits=_limits(),
    )
    assert report.passed
    assert report.failures == ()


def test_failed_gates_report_concrete_adjustments():
    main = np.eye(4)
    aux = np.eye(4)
    aux[:3, :3] = Rotation.from_euler(
        "xyz", (3.0, -4.0, 5.0), degrees=True).as_matrix()
    aux[:3, 3] = (0.13, 0.0, -0.01)
    report = evaluate_alignment(
        {"main": main, "aux": aux},
        {"main": _coverage((80, 10, 80, -5)), "aux": _coverage()},
        {"main": (0.8, 0.04), "aux": (0.3, 0.2)},
        pair_detection_fraction=0.5,
        sample_count=10,
        limits=_limits(),
    )
    assert not report.passed
    message = "\n".join(report.failures)
    assert "baseline" in message
    assert "raise aux" in message
    assert "pitch" in message and "yaw" in message and "roll" in message
    assert "right workspace margin" in message
    assert "bottom workspace margin" in message
    assert "translation spread" in message
    assert "rotation spread" in message
    assert "visibility" in message
    assert "paired pose samples" in message


def test_workspace_projection_returns_ordered_pixel_margins():
    points = np.array([
        [-0.1, -0.1, 1.0], [-0.1, 0.1, 1.0],
        [0.1, -0.1, 1.0], [0.1, 0.1, 1.0],
    ])
    camera_matrix = np.array([
        [100.0, 0.0, 320.0],
        [0.0, 100.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    result = project_workspace(
        points, np.eye(4), camera_matrix, np.zeros(5), (640, 480))
    np.testing.assert_allclose(result.pixels.min(axis=0), (310.0, 230.0))
    np.testing.assert_allclose(result.pixels.max(axis=0), (330.0, 250.0))
    np.testing.assert_allclose(result.margins_px, (310.0, 309.0, 230.0, 229.0))
    assert result.all_in_front


def test_margin_guidance_maps_image_edges_to_physical_aiming():
    result = _coverage((100.0, 10.0, 100.0, 10.0))
    assert coverage_guidance(result, 50.0, (1280, 720)) == "aim right, tilt down"


def test_workspace_xy_comes_from_task_configs():
    limits = load_alignment_limits()
    corners = load_workspace_corners(limits)
    np.testing.assert_allclose(corners.min(axis=0), (0.10, -0.10, 0.0))
    np.testing.assert_allclose(corners.max(axis=0), (0.30, 0.10, 0.15))


def test_missing_anchor_frames_evict_stale_measurements_from_window():
    main = np.eye(4)
    aux = np.eye(4)
    aux[0, 3] = 0.11
    sample = {"main": main, "aux": aux}
    measurements = deque([sample] * 20 + [None] * 10, maxlen=30)
    camera_matrix = np.array([
        [100.0, 0.0, 320.0],
        [0.0, 100.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    workspace = np.array([
        [x, y, z]
        for x in (-0.1, 0.1)
        for y in (-0.1, 0.1)
        for z in (1.0, 1.1)
    ])
    result = _current_measurement(
        measurements,
        {"main": camera_matrix, "aux": camera_matrix},
        {"main": np.zeros(5), "aux": np.zeros(5)},
        {"main": (640, 480), "aux": (640, 480)},
        workspace,
        _limits(),
    )
    assert result is not None
    report = result[1]
    assert report.sample_count == 20
    assert np.isclose(report.pair_detection_fraction, 2.0 / 3.0)


def test_anchor_search_and_measured_view_have_same_header_height():
    limits = _limits()
    image_sizes = {"main": (1280, 720), "aux": (1280, 720)}
    visible = {"main": True, "aux": True}
    search_header, _ = _viewer_lines(None, limits, image_sizes, visible)

    main = np.eye(4)
    aux = np.eye(4)
    aux[0, 3] = 0.11
    report = evaluate_alignment(
        {"main": main, "aux": aux},
        {"main": _coverage(), "aux": _coverage()},
        {"main": (0.2, 0.04), "aux": (0.3, 0.05)},
        pair_detection_fraction=1.0,
        sample_count=30,
        limits=limits,
    )
    measured_header, _ = _viewer_lines(report, limits, image_sizes, visible)
    assert len(search_header) == len(measured_header) == 2


def test_metric_header_colors_each_angle_independently():
    limits = _limits()
    main = np.eye(4)
    aux = np.eye(4)
    aux[:3, :3] = Rotation.from_euler(
        "xyz", (1.0, 3.0, -1.5), degrees=True).as_matrix()
    aux[:3, 3] = (0.11, 0.0, 0.002)
    report = evaluate_alignment(
        {"main": main, "aux": aux},
        {"main": _coverage(), "aux": _coverage()},
        {"main": (0.2, 0.04), "aux": (0.3, 0.05)},
        pair_detection_fraction=1.0,
        sample_count=30,
        limits=limits,
    )
    header, _ = _viewer_lines(
        report, limits,
        {"main": (1280, 720), "aux": (1280, 720)},
        {"main": True, "aux": True},
    )
    assert [span.color for span in header[1].spans] == [
        GREEN, GREEN, GREEN, RED, GREEN,
    ]
