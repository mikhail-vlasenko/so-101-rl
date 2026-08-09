"""MuJoCo visible-surface sampling shared by live and dense sponge channels."""

from __future__ import annotations

import numpy as np
import mujoco


# Group 1 is reserved for camera stand-ins.  Rays begin inside those geoms, so
# they alone are masked; floor, arm, gripper and ring geometry all occlude.
OCCLUDER_GEOMGROUP = np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8)


def unit_box_surface_points(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic interior grid on all six faces of the unit box."""
    if grid_size < 2:
        raise ValueError("surface grid size must be at least 2")
    # Preserve the established 3x3 live-centroid geometry exactly.  Denser BPS
    # sampling approaches the edges while still avoiding duplicated edges and
    # corners shared by adjacent faces.
    fractions = (np.asarray((-0.7, 0.0, 0.7)) if grid_size == 3
                 else np.linspace(-0.95, 0.95, grid_size))
    points = []
    normals = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1.0, 1.0):
            for a in fractions:
                for b in fractions:
                    point = np.zeros(3)
                    point[axis] = sign
                    point[u] = a
                    point[v] = b
                    normal = np.zeros(3)
                    normal[axis] = sign
                    points.append(point)
                    normals.append(normal)
    return np.asarray(points), np.asarray(normals)


def box_surface_points_world(data, geom_id: int, half_extents: np.ndarray,
                             grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the current randomized box's physical faces in base coordinates."""
    unit_points, unit_normals = unit_box_surface_points(grid_size)
    return transform_box_surface_points_world(
        data, geom_id, half_extents, unit_points, unit_normals)


def transform_box_surface_points_world(
        data, geom_id: int, half_extents: np.ndarray,
        unit_points: np.ndarray, unit_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a precomputed unit-box surface grid at the current geom pose."""
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    center = data.geom_xpos[geom_id]
    points = center + (unit_points * np.asarray(half_extents)) @ rotation.T
    return points, unit_normals @ rotation.T


def visible_surface_mask(model, data, cam, excluded_body_id: int,
                         points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, int]:
    """Visible sample mask and facing-surface count for one calibrated camera."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if points.shape != normals.shape:
        raise ValueError("surface points and normals must have matching shapes")
    to_cam = cam.pos - points
    facing = np.einsum("ij,ij->i", normals, to_cam) > 0.0
    n_facing = int(facing.sum())
    visible = np.zeros(points.shape[0], dtype=bool)
    candidates = facing & cam.in_view(points)
    n_rays = int(candidates.sum())
    if n_rays == 0:
        return visible, n_facing
    candidate_indices = np.flatnonzero(candidates)
    candidate_points = points[candidates]
    rays = candidate_points - cam.pos
    distances = np.linalg.norm(rays, axis=1)
    geom_ids = np.empty(n_rays, dtype=np.int32)
    hits = np.empty(n_rays, dtype=np.float64)
    mujoco.mj_multiRay(
        model,
        data,
        cam.pos,
        np.ascontiguousarray(rays / distances[:, None]).ravel(),
        OCCLUDER_GEOMGROUP,
        1,
        excluded_body_id,
        geom_ids,
        hits,
        None,
        n_rays,
        float(distances.max()),
    )
    visible[candidate_indices[(hits < 0.0) | (hits >= distances)]] = True
    return visible, n_facing


def visible_surface(model, data, cam, excluded_body_id: int,
                    points: np.ndarray, normals: np.ndarray):
    """Existing live-channel contract: visible/facing fraction and centroid."""
    visible, n_facing = visible_surface_mask(
        model, data, cam, excluded_body_id, points, normals)
    if not np.any(visible):
        return 0.0, None
    return float(np.count_nonzero(visible) / n_facing), points[visible].mean(axis=0)
