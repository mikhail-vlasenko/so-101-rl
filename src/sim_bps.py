"""Synthetic partial-surface clouds for the simulated precise BPS channel.

This is geometry sampling, not RGB/depth rendering: it samples the randomized
sponge faces, applies both calibrated camera FOVs and the same MuJoCo occlusion
rays as the live channel, then models correspondence noise/dropout before the
shared voxel and BPS transforms.  The generator is suitable for vectorized
environments because it runs only on scheduled static captures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from src.bps import (
    CONFIG_PATH,
    BPSConfig,
    BPSMeasurement,
    encode_bps,
    voxel_first_indices,
)
from src.surface_cloud import (
    transform_box_surface_points_world,
    unit_box_surface_points,
    visible_surface_mask,
)


@dataclass(frozen=True)
class SyntheticCloudConfig:
    point_noise_sigma_m: float
    point_dropout_probability: float
    whole_view_loss_probability: float
    voxel_size_m: float

    def __post_init__(self):
        if self.point_noise_sigma_m < 0.0:
            raise ValueError("point_noise_sigma_m must be non-negative")
        for name, value in (
            ("point_dropout_probability", self.point_dropout_probability),
            ("whole_view_loss_probability", self.whole_view_loss_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")


def clean_synthetic_cloud_config() -> SyntheticCloudConfig:
    """No-DR cloud settings with the repository-owned voxel size."""
    with CONFIG_PATH.open() as stream:
        voxel_size_m = float(
            yaml.safe_load(stream)["dense_stereo_feasibility"]["voxel_size_m"])
    return SyntheticCloudConfig(0.0, 0.0, 0.0, voxel_size_m)


@dataclass(frozen=True)
class SyntheticBPSCapture:
    measurement: BPSMeasurement
    points_base: np.ndarray
    left_visible_count: int
    correspondence_count: int


class SyntheticBPSGenerator:
    """Generate the sim twin of one successful real dense-stereo refresh."""

    def __init__(self, bps_config: BPSConfig, cloud_config: SyntheticCloudConfig):
        self.bps_config = bps_config
        self.cloud_config = cloud_config
        self._unit_points, self._unit_normals = unit_box_surface_points(
            bps_config.synthetic_surface_grid_size)

    @property
    def unit_points(self) -> np.ndarray:
        return self._unit_points

    @property
    def unit_normals(self) -> np.ndarray:
        return self._unit_normals

    def whole_view_lost(self, rng: np.random.Generator) -> bool:
        """Draw the per-job complete dense-stereo failure before raycasting."""
        return bool(rng.random() < self.cloud_config.whole_view_loss_probability)

    def capture(self, model, data, cameras, cube_geom_id: int,
                cube_body_id: int, half_extents: np.ndarray,
                rng: np.random.Generator) -> SyntheticBPSCapture | None:
        if len(cameras) != 2:
            raise ValueError("dense stereo requires exactly two cameras")
        if self.whole_view_lost(rng):
            return None

        points, normals = transform_box_surface_points_world(
            data,
            cube_geom_id,
            half_extents,
            self._unit_points,
            self._unit_normals,
        )
        masks = tuple(
            visible_surface_mask(model, data, camera, cube_body_id, points, normals)[0]
            for camera in cameras
        )
        return self.capture_visible(points, masks, rng)

    def capture_visible(self, points: np.ndarray,
                        masks: tuple[np.ndarray, np.ndarray],
                        rng: np.random.Generator) -> SyntheticBPSCapture | None:
        """Degrade and encode a dense surface whose visibility is resolved.

        The caller owns the whole-view-loss draw so it can avoid preparing or
        raycasting dense points when that job is absent.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(masks) != 2:
            raise ValueError("dense stereo requires exactly two visibility masks")
        masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in masks)
        if any(mask.shape != (points.shape[0],) for mask in masks):
            raise ValueError("dense visibility masks must match the surface points")
        left_visible_count = int(np.count_nonzero(masks[0]))
        shared = masks[0] & masks[1]
        shared_indices = np.flatnonzero(shared)
        if shared_indices.size == 0 or left_visible_count == 0:
            return None
        keep = rng.random(shared_indices.size) >= \
            self.cloud_config.point_dropout_probability
        retained = points[shared_indices[keep]].copy()
        correspondence_count = retained.shape[0]
        if correspondence_count == 0:
            return None
        if self.cloud_config.point_noise_sigma_m > 0.0:
            retained += rng.normal(
                0.0, self.cloud_config.point_noise_sigma_m, retained.shape)
        voxel_indices = voxel_first_indices(retained, self.cloud_config.voxel_size_m)
        retained = retained[voxel_indices]
        valid_fraction = correspondence_count / left_visible_count
        # A coarse synthetic surface grid can contain more shared samples than
        # left-only samples only if the caller supplies inconsistent masks;
        # clipping also protects float ratios at the exact boundary.
        valid_fraction = float(np.clip(valid_fraction, 0.0, 1.0))
        measurement = encode_bps(retained, valid_fraction, self.bps_config)
        return SyntheticBPSCapture(
            measurement=measurement,
            points_base=retained,
            left_visible_count=left_visible_count,
            correspondence_count=correspondence_count,
        )
