"""Sparse two-camera triangulation in the arm base frame.

The binocular pipeline (vision_multicam_longterm.md) triangulates *matched
image points* across the two independently-anchored cameras — never dense
stereo. Each camera is mapped into the base frame by its own EMA-smoothed
two-tag board tracker (`real.calib.table_anchor.TableAnchorTracker`), so the two
cameras need no
stereo calibration and no rigid coupling: a pixel in either view back-projects
to a base-frame ray, and a point seen in both views is the closest-approach
midpoint of its two rays. The ray-pair gap is the cross-view consistency
signal — rays that don't nearly intersect mean one view's measurement (or
anchoring) is off.
"""
import cv2
import numpy as np


def pixel_rays(pixels, camera_matrix, dist_coeffs, T_base_cam):
    """Back-project distorted pixel coords into unit rays in the base frame.

    `pixels` is (N, 2) raw image coordinates (distortion is removed here, so
    pass detector output as-is). Returns `(origin (3,), dirs (N, 3))`: all rays
    share the camera centre as origin; directions are unit vectors in the base
    frame.
    """
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, camera_matrix, dist_coeffs).reshape(-1, 2)
    dirs_cam = np.hstack([norm, np.ones((len(norm), 1))])
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    return T_base_cam[:3, 3].copy(), dirs_cam @ T_base_cam[:3, :3].T


def triangulate_rays(o1, d1, o2, d2):
    """Closest-approach midpoints of paired rays: (points (N, 3), gaps (N,)).

    `o1`/`o2` are the two ray origins (3,), `d1`/`d2` (N, 3) unit directions;
    row i of `d1` corresponds to row i of `d2`. `gaps` is the miss distance
    between each ray pair at closest approach — the cross-view consistency
    metric. Raises on (near-)parallel pairs, where the intersection is
    unconstrained along the rays.
    """
    d1 = np.asarray(d1, dtype=np.float64).reshape(-1, 3)
    d2 = np.asarray(d2, dtype=np.float64).reshape(-1, 3)
    w = np.asarray(o1, dtype=np.float64) - np.asarray(o2, dtype=np.float64)
    b = np.sum(d1 * d2, axis=1)
    d1w = d1 @ w
    d2w = d2 @ w
    denom = 1.0 - b ** 2
    if np.any(denom < 1e-9):
        raise ValueError("near-parallel ray pair: triangulation is unconstrained")
    t1 = (b * d2w - d1w) / denom
    t2 = (d2w - b * d1w) / denom
    p1 = o1 + t1[:, None] * d1
    p2 = o2 + t2[:, None] * d2
    return (p1 + p2) / 2.0, np.linalg.norm(p1 - p2, axis=1)
