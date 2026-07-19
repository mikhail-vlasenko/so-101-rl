"""Synthetic-render tests for the visual-hull estimator
(real/tracking/hull_shape.py): a known box is projected into two synthetic
pinhole views (a convex box's silhouette is the convex hull of its projected
corners — exact), and the hull estimate must recover center and shape within
the two-view hull's inherent overestimate."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.tracking.hull_shape import HULL_VOXEL_PITCH_M, hull_estimate
from src.shape_obs import box_sqrtm, sqrtm_from_cov

W, H = 1280, 720
K = np.array([[950.0, 0.0, 640.0],
              [0.0, 950.0, 360.0],
              [0.0, 0.0, 1.0]])
DIST = np.zeros(5)

HALF = np.array([0.03, 0.02, 0.0125])


def _camera_at(pos, target):
    """T_base_cam of an OpenCV camera (+z into the scene, +y down) at `pos`
    looking at `target`."""
    z = np.asarray(target, float) - np.asarray(pos, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T


def _box_corners(center, R, half):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)], dtype=float)
    return center + (signs * half) @ R.T


def _silhouette(corners, T_base_cam):
    """Exact box silhouette: convex hull of the projected corners, filled."""
    T_cam_base = np.linalg.inv(T_base_cam)
    cam_pts = corners @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    assert np.all(cam_pts[:, 2] > 0.0), "box behind the synthetic camera"
    px, _ = cv2.projectPoints(cam_pts.reshape(-1, 1, 3), np.zeros(3),
                              np.zeros(3), K, DIST)
    hull = cv2.convexHull(px.reshape(-1, 2).astype(np.float32))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    return mask.astype(bool)


def _rig(center):
    """Two-camera geometry ~45 degrees apart, both aimed at `center`."""
    cams = {
        "main": _camera_at([0.11, -0.41, 0.18], center),
        "aux": _camera_at([0.43, -0.37, 0.18], center),
    }
    return {name: (K, DIST, T) for name, T in cams.items()}


def _estimate(center, R):
    geometry = _rig(center)
    corners = _box_corners(center, R, HALF)
    masks = {name: _silhouette(corners, T_base_cam)
             for name, (_, _, T_base_cam) in geometry.items()}
    return hull_estimate(masks, geometry, ("main", "aux"))


def test_hull_recovers_center():
    for yaw in (0.0, 0.5, 1.2):
        center = np.array([0.22, 0.02, HALF[2]])
        R = Rotation.from_euler("z", yaw).as_matrix()
        c, M = _estimate(center, R)
        assert c is not None
        # The two-view hull is a superset of the box, roughly symmetric about
        # it, so the centroid lands near the true center.
        assert np.linalg.norm(c - center) < 0.015, (yaw, c - center)


def test_hull_shape_bounds_the_box():
    """The visual hull contains the box, so its per-direction spread is at
    least the box's; with only two views it overestimates along the
    unconstrained direction but must stay bounded."""
    center = np.array([0.22, 0.02, HALF[2]])
    R = Rotation.from_euler("z", 0.4).as_matrix()
    c, M = _estimate(center, R)
    S = sqrtm_from_cov(M)
    gt = np.sort(np.linalg.eigvalsh(box_sqrtm(R, HALF)))
    est = np.sort(np.linalg.eigvalsh(S))
    # Contains the box (allow one voxel pitch of discretization slack)...
    assert np.all(est >= gt - HULL_VOXEL_PITCH_M), (est, gt)
    # ...but no runaway smear: within ~2.5x per axis.
    assert np.all(est <= 2.5 * gt + HULL_VOXEL_PITCH_M), (est, gt)


def test_hull_vertical_extent_discriminates_resting_face():
    """Standing the box on different faces must change the hull's vertical
    spread — the 'which way is it shorter' signal. A tall box is
    well-constrained (its top face pins the height); a flat box smears ~1.5x
    with cameras only ~20 degrees above the table (measured on this synthetic
    rig) — still comfortably separable."""
    center_flat = np.array([0.22, 0.02, HALF[2]])
    c_flat, M_flat = _estimate(center_flat, np.eye(3))
    R_tall = Rotation.from_euler("y", np.pi / 2).as_matrix()  # x-axis vertical
    center_tall = np.array([0.22, 0.02, HALF[0]])
    c_tall, M_tall = _estimate(center_tall, R_tall)
    ez = np.array([0.0, 0.0, 1.0])
    spread_flat = np.sqrt(ez @ M_flat @ ez)
    spread_tall = np.sqrt(ez @ M_tall @ ez)
    # True vertical spreads: hz/sqrt(3) flat, hx/sqrt(3) tall.
    true_flat = HALF[2] / np.sqrt(3)
    true_tall = HALF[0] / np.sqrt(3)
    assert true_flat <= spread_flat <= 1.7 * true_flat, (spread_flat, true_flat)
    np.testing.assert_allclose(spread_tall, true_tall, rtol=0.15)
    assert spread_tall > 1.4 * spread_flat


def test_hull_empty_mask_returns_none():
    center = np.array([0.22, 0.02, HALF[2]])
    geometry = _rig(center)
    corners = _box_corners(center, np.eye(3), HALF)
    masks = {"main": _silhouette(corners, geometry["main"][2]),
             "aux": np.zeros((H, W), dtype=bool)}
    c, M = hull_estimate(masks, geometry, ("main", "aux"))
    assert c is None and M is None


def test_hull_disjoint_cones_return_none():
    """Two masks whose cones cannot intersect (opposite image corners) must
    yield an empty hull, not a crash."""
    center = np.array([0.22, 0.02, HALF[2]])
    geometry = _rig(center)
    m1 = np.zeros((H, W), dtype=bool)
    m1[:40, :40] = True
    m2 = np.zeros((H, W), dtype=bool)
    m2[-40:, -40:] = True
    c, M = hull_estimate({"main": m1, "aux": m2}, geometry, ("main", "aux"))
    assert c is None and M is None
