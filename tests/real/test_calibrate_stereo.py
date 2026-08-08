import cv2
from dataclasses import replace
import numpy as np

from real.calib.calibrate_camera import object_points
from real.calib.calibrate_stereo import (
    StereoCalibrationLimits,
    calibration_failures,
    solve_stationary_board,
)
from real.calib.extrinsics import mat_inv, rt_to_mat


def _synthetic_case():
    pattern = (7, 9)
    points = object_points(pattern).astype(np.float64)
    mats = {
        "main": np.array([[960.0, 0.0, 640.0],
                          [0.0, 958.0, 360.0],
                          [0.0, 0.0, 1.0]]),
        "aux": np.array([[955.0, 0.0, 625.0],
                         [0.0, 957.0, 350.0],
                         [0.0, 0.0, 1.0]]),
    }
    dists = {name: np.zeros(5) for name in mats}
    T_main_board = rt_to_mat(
        np.array([0.10, -0.04, 0.03]), np.array([-0.06, -0.08, 0.48]))
    T_aux_main = rt_to_mat(
        np.array([0.003, -0.008, 0.005]), np.array([-0.109, 0.002, 0.001]))
    poses = {"main": T_main_board, "aux": T_aux_main @ T_main_board}
    samples = {}
    for name in mats:
        T = poses[name]
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        corners, _ = cv2.projectPoints(points, rvec, T[:3, 3], mats[name], dists[name])
        samples[name] = [corners.astype(np.float32) for _ in range(4)]
    return points, samples, mats, dists, T_aux_main


def test_stationary_board_recovers_relative_transform_and_rectifies():
    points, samples, mats, dists, expected = _synthetic_case()
    samples["main"][1] = samples["main"][1][::-1].copy()
    samples["aux"][2] = samples["aux"][2][::-1].copy()
    result = solve_stationary_board(points, samples, mats, dists, (1280, 720))

    delta = result.T_aux_main @ mat_inv(expected)
    delta_rvec, _ = cv2.Rodrigues(delta[:3, :3])
    assert np.linalg.norm(delta[:3, 3]) < 1e-6
    assert np.linalg.norm(delta_rvec) < 1e-6
    assert result.reprojection_rmse_px["main"] < 1e-3
    assert result.reprojection_rmse_px["aux"] < 1e-3
    assert result.vertical_p95_px < 1e-3


def test_quality_gate_reports_each_failed_contract():
    points, samples, mats, dists, _ = _synthetic_case()
    result = solve_stationary_board(points, samples, mats, dists, (1280, 720))
    result = replace(
        result,
        reprojection_rmse_px={"main": 0.10, "aux": 0.20},
        vertical_p95_px=0.30,
    )
    limits = StereoCalibrationLimits(
        capture_frames=4,
        min_detected_pairs=4,
        max_reprojection_rmse_px=0.15,
        max_rectified_vertical_p95_px=0.20,
        camera_movement_warning_translation_mm=1.0,
        camera_movement_warning_rotation_deg=0.1,
    )

    failures = calibration_failures(result, limits)

    assert any("aux reprojection" in failure for failure in failures)
    assert any("rectified vertical" in failure for failure in failures)
