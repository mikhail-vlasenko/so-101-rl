"""Contract tests for real/vision/stereo.py: back-projection and ray triangulation.

Synthetic round-trip with the full camera model: project known base-frame
points through two realistic distorted cameras with cv2.projectPoints, then
recover them via pixel_rays + triangulate_rays. Exact (noise-free) inputs must
come back to sub-0.1 mm — any convention slip (distortion handling, rotation
direction, OpenCV camera axes) shows up as centimetres, not tenths.
"""
import numpy as np
import pytest

from real.calib.extrinsics import mat_inv, mat_to_rt
from real.vision.stereo import pixel_rays, triangulate_rays

K = np.array([[970.0, 0.0, 640.0],
              [0.0, 968.0, 360.0],
              [0.0, 0.0, 1.0]])
DIST = np.array([0.04, -0.17, -0.002, -0.0006, 0.12])


def look_at(cam_pos, target):
    """T_base_cam for an OpenCV camera (+x right, +y down, +z forward) at
    `cam_pos` looking at `target`, with the image kept roughly upright."""
    z = np.asarray(target, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = cam_pos
    return T


def project(points_base, T_base_cam):
    import cv2
    rvec, tvec = mat_to_rt(mat_inv(T_base_cam))
    px, _ = cv2.projectPoints(np.asarray(points_base, dtype=np.float64), rvec, tvec, K, DIST)
    return px.reshape(-1, 2)


# Rig-like geometry: two cameras ~20 cm apart, ~0.45 m from the workspace.
T_CAM1 = look_at([0.10, -0.42, 0.20], [0.10, 0.0, 0.05])
T_CAM2 = look_at([0.30, -0.40, 0.22], [0.10, 0.0, 0.05])

POINTS = np.array([
    [0.10, -0.05, 0.02],
    [0.15, 0.05, 0.00],
    [0.02, 0.02, 0.10],
    [0.12, -0.02, 0.025],
])


def test_roundtrip_recovers_points():
    o1, d1 = pixel_rays(project(POINTS, T_CAM1), K, DIST, T_CAM1)
    o2, d2 = pixel_rays(project(POINTS, T_CAM2), K, DIST, T_CAM2)
    recovered, gaps = triangulate_rays(o1, d1, o2, d2)
    assert np.allclose(recovered, POINTS, atol=1e-4)
    assert np.all(gaps < 1e-4)


def test_pixel_error_moves_point_but_reports_gap():
    px1 = project(POINTS, T_CAM1)
    px1[0] += [3.0, 0.0]   # 3 px of centroid slop in one view
    o1, d1 = pixel_rays(px1, K, DIST, T_CAM1)
    o2, d2 = pixel_rays(project(POINTS, T_CAM2), K, DIST, T_CAM2)
    recovered, gaps = triangulate_rays(o1, d1, o2, d2)
    err = np.linalg.norm(recovered[0] - POINTS[0])
    assert 5e-4 < err < 2e-2      # a few mm of 3D error, not metres
    assert gaps[0] > 5 * gaps[1:].max()   # and the gap flags the bad pair


def test_parallel_rays_raise():
    d = np.array([[0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="parallel"):
        triangulate_rays(np.zeros(3), d, np.array([0.2, 0.0, 0.0]), d)
