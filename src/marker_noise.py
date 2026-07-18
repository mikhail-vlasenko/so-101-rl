"""Anisotropic (camera-frame) solvePnP position noise for simulated AprilTag obs.

Real monocular tag localization is accurate in the image plane but poor along the
camera->tag ray: the range comes from the tag's apparent size, a weak cue, so the
position error is a squashed ellipsoid (sub-mm lateral, several mm in depth), not
the round ball an isotropic iid Gaussian models. This module draws that ellipsoid.

The two sigmas are derived from the calibrated camera geometry, the per-tag
distance, and the printed tag size -- not hand-tuned magnitudes:

    lateral_sigma = (Z / f) * px_noise                       # image plane, well constrained
    depth_sigma   = (Z**2 / (f * W)) * px_noise * depth_factor  # range, from apparent size

with Z the tag->camera distance, f the focal length (px), W the tag edge (m).
Only two DR knobs remain: `px_noise` (effective solvePnP corner error, px) and
`depth_factor` (range-error amplification over the image-plane baseline). f comes
from real/vision/camera_intrinsics.yaml and W from real/marker_spec.py -- the same
calibration the real pipeline (real/vision/pose.py, real/rollout/marker_obs.py) uses, so the
noise is grounded in the physical setup rather than duplicated constants.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

_INTRINSICS_PATH = Path(__file__).resolve().parent.parent / "real" / "vision" / "camera_intrinsics.yaml"


class CameraIntrinsics(NamedTuple):
    """Calibrated pinhole intrinsics (px) and image size from
    real/vision/camera_intrinsics.yaml -- the single source of truth shared with the
    real solvePnP pipeline (real/vision/pose.py). fx/fy/cx/cy define the camera matrix;
    width/height are the frame bounds the field-of-view check tests against."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def focal_px(self) -> float:
        """Mean focal length (px), the scalar the anisotropic pos-noise model uses."""
        return 0.5 * (self.fx + self.fy)


def load_camera_intrinsics(path: Path = _INTRINSICS_PATH) -> CameraIntrinsics:
    """Load the calibrated intrinsics -- the single read of the camera YAML shared
    by the noise model (focal length) and the FOV visibility check (full matrix)."""
    with open(path) as f:
        d = yaml.safe_load(f)
    m = d["camera_matrix"]
    return CameraIntrinsics(fx=m[0][0], fy=m[1][1], cx=m[0][2], cy=m[1][2],
                            width=int(d["image_width"]), height=int(d["image_height"]))


def load_focal_px(path: Path = _INTRINSICS_PATH) -> float:
    """Mean focal length (px) from the calibrated camera matrix -- the single
    source of truth shared with the real solvePnP pipeline (real/vision/pose.py)."""
    return load_camera_intrinsics(path).focal_px


def pos_noise_sigmas(tag_pos, cam_pos, tag_size_m, focal_px, px_noise, depth_factor):
    """(lateral_sigma, depth_sigma, depth_dir) of one tag's camera-frame noise
    ellipsoid. depth_dir is the unit camera->tag ray (the axis the depth sigma
    applies to); the two orthogonal directions get lateral_sigma. `tag_pos` is the
    TRUE position so the depth axis is the real line of sight, not a circular
    function of the noise itself. Single source of the depth-vs-lateral formula,
    shared by the sampler below and the tests."""
    ray = tag_pos - cam_pos
    dist = np.linalg.norm(ray)
    depth_dir = ray / dist
    lateral_sigma = (dist / focal_px) * px_noise
    depth_sigma = (dist * dist / (focal_px * tag_size_m)) * px_noise * depth_factor
    return lateral_sigma, depth_sigma, depth_dir


def anisotropic_pos_noise(rng, tag_pos, cam_pos, tag_size_m, focal_px,
                          px_noise, depth_factor):
    """One tag's world-frame position noise (3,): small lateral (image plane),
    large depth (along the camera->tag ray)."""
    lateral_sigma, depth_sigma, depth_dir = pos_noise_sigmas(
        tag_pos, cam_pos, tag_size_m, focal_px, px_noise, depth_factor)
    g = rng.normal(0.0, 1.0, 3)
    g_depth = (g @ depth_dir) * depth_dir
    return lateral_sigma * (g - g_depth) + depth_sigma * g_depth
