"""Visual-hull shape estimator v1: two silhouettes -> body center + M.

The estimator behind the precise obs channel (plan decision 3): the two
cameras' masks define silhouette cones; voxels inside BOTH cones (and inside
the workspace box, above the table) approximate the object volume; its
centroid and second-moment matrix M are the estimate. `sqrtm_from_cov(M)`
(src/shape_obs.py) turns M into the √M the policy consumes — multi-frame
averaging over a static window happens on M in the linear domain BEFORE the
square root (real/tracking/eval_estimator.py offline, the ObjectSource
refresh worker online).

No learned parts, no per-object knowledge: masks + per-unit intrinsics + the
per-session table-tag anchoring are the only inputs. Scored exclusively by
real/tracking/eval_estimator.py against tag GT — tune nothing by eye.
"""
import cv2
import numpy as np

from real.calib.extrinsics import mat_inv
from real.vision.stereo import pixel_rays, triangulate_rays

# Voxel pitch (m). ~4 mm resolves the 25 mm sponge axis into ~6 cells while
# keeping the local grid ~2e5 voxels.
HULL_VOXEL_PITCH_M = 0.004

# The voxel grid is centered on the triangulated mask-centroid point and
# clipped to the workspace; this half-size must cover the whole hull. The
# silhouette-cone intersection smears at most ~object_diameter/sin(view
# separation) ≈ 9 cm along the mean viewing direction for the ~45°-separated
# rig, so 12 cm of margin is safe for hand-sized objects.
HULL_LOCAL_HALF_M = 0.12

# Workspace box in the base frame (m): generous bounds around the reachable
# table area; z >= 0 is the table plane — nothing real extends below it.
WORKSPACE_LOW = np.array([-0.10, -0.40, 0.0])
WORKSPACE_HIGH = np.array([0.60, 0.40, 0.35])


def _mask_lookup(mask, pts_px):
    """Bool (N,) — each pixel coordinate lands inside the mask. Points outside
    the image are outside the silhouette."""
    h, w = mask.shape
    px = np.rint(pts_px).astype(int)
    inside = ((px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h))
    out = np.zeros(len(px), dtype=bool)
    sel = np.nonzero(inside)[0]
    out[sel] = mask[px[sel, 1], px[sel, 0]]
    return out


def _in_silhouette(points, mask, camera_matrix, dist_coeffs, T_base_cam):
    """Bool (N,) — base-frame points inside this camera's silhouette cone."""
    T_cam_base = mat_inv(T_base_cam)
    cam_pts = points @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    ok = cam_pts[:, 2] > 0.0  # behind the camera is outside the cone
    out = np.zeros(len(points), dtype=bool)
    if not ok.any():
        return out
    px, _ = cv2.projectPoints(cam_pts[ok].reshape(-1, 1, 3), np.zeros(3),
                              np.zeros(3), camera_matrix, dist_coeffs)
    out[np.nonzero(ok)[0]] = _mask_lookup(mask, px.reshape(-1, 2))
    return out


def _grid_around(center):
    """Voxel-center grid of pitch HULL_VOXEL_PITCH_M in the local box around
    `center`, clipped to the workspace box (z >= table included)."""
    lo = np.maximum(center - HULL_LOCAL_HALF_M, WORKSPACE_LOW)
    hi = np.minimum(center + HULL_LOCAL_HALF_M, WORKSPACE_HIGH)
    axes = [np.arange(lo[k] + HULL_VOXEL_PITCH_M / 2, hi[k], HULL_VOXEL_PITCH_M)
            for k in range(3)]
    if any(len(a) == 0 for a in axes):
        return np.zeros((0, 3))
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _seed_point(masks, geometry, cameras):
    """Triangulated point of the two mask centroids — only used to center the
    local voxel grid, so the visible-surface bias is irrelevant here."""
    rays = []
    for name in cameras:
        ys, xs = np.nonzero(masks[name])
        if xs.size == 0:
            return None
        mat, dist, T_base_cam = geometry[name]
        rays.append(pixel_rays(np.array([[xs.mean(), ys.mean()]]), mat, dist,
                               T_base_cam))
    pts, _ = triangulate_rays(*rays[0], *rays[1])
    return pts[0]


def hull_estimate(masks, geometry, cameras):
    """One frame pair -> (centroid (3,), M (3,3)) of the visual hull, or
    (None, None) when the cones don't intersect (empty mask, broken
    anchoring) — the caller holds its last estimate, same as any missed
    detection.

    `masks` maps camera name -> bool HxW; `geometry` maps camera name ->
    (camera_matrix, dist_coeffs, T_base_cam). M includes the voxel cubes' own
    second moment (pitch^2/12 per axis), making it the exact second moment of
    the voxelized volume rather than of the center points.
    """
    seed = _seed_point(masks, geometry, cameras)
    if seed is None:
        return None, None
    points = _grid_around(seed)
    if len(points) == 0:
        return None, None
    inside = np.ones(len(points), dtype=bool)
    for name in cameras:
        inside &= _in_silhouette(points, masks[name], *geometry[name])
    if not inside.any():
        return None, None
    hull = points[inside]
    centroid = hull.mean(axis=0)
    centered = hull - centroid
    M = (centered.T @ centered) / len(hull) \
        + np.eye(3) * (HULL_VOXEL_PITCH_M ** 2 / 12.0)
    return centroid, M
