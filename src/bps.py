"""Fixed Basis Point Set contract for the sponge's precise observation.

Dense stereo and simulation both produce a voxel-downsampled visible-surface
cloud in the arm-base frame.  This module is the only place that turns that
cloud into the policy-facing representation: its measured centroid plus the
distances from a fixed, base-aligned 4x4x4 basis to the centered cloud.

The transform is deliberately deterministic.  Input points are sorted before
the centroid and nearest-neighbour reductions, so permuting a cloud cannot
change even the bytes of the resulting float32 observation.  Empty clouds are
measurement misses and must be handled by :class:`BPSObsState`, not encoded as
a fabricated cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from src.shape_obs import MARKER_AGE_CAP_S


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"
BPS_DISTANCE_DIM = 64
BPS_AGE_CAP_S = MARKER_AGE_CAP_S
BPS_OBS_DIM = BPS_DISTANCE_DIM + 3 + 1 + 1


@dataclass(frozen=True)
class BPSConfig:
    basis_axis_m: tuple[float, float, float, float]
    distance_cap_m: float
    synthetic_surface_grid_size: int

    def __post_init__(self):
        axis = self.basis_axis_m
        if len(axis) != 4 or tuple(sorted(axis)) != axis or len(set(axis)) != 4:
            raise ValueError(
                "bps.basis_axis_m must contain four strictly increasing values")
        if self.distance_cap_m <= 0.0:
            raise ValueError("bps.distance_cap_m must be positive")
        if self.synthetic_surface_grid_size < 2:
            raise ValueError("bps.synthetic_surface_grid_size must be at least 2")


@dataclass(frozen=True)
class BPSMeasurement:
    """One successful precise-cloud measurement, ready for hold/age state."""

    distances: np.ndarray
    center_base: np.ndarray
    valid_fraction: float


@dataclass(frozen=True)
class BPSObservation:
    """Currently served precise block, including held-measurement metadata."""

    distances: np.ndarray
    center_base: np.ndarray
    age_s: float
    valid_fraction: float

    def flat(self) -> np.ndarray:
        """Policy block: distances, center, age, valid fraction."""
        block = np.concatenate((
            self.distances,
            self.center_base,
            np.array([self.age_s, self.valid_fraction], dtype=np.float32),
        )).astype(np.float32)
        assert block.shape == (BPS_OBS_DIM,)
        return block


def load_bps_config(path: Path = CONFIG_PATH) -> BPSConfig:
    """Load and validate the resolved values that define the BPS contract."""
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)["bps"]
    axis = tuple(float(value) for value in data["basis_axis_m"])
    config = BPSConfig(
        basis_axis_m=axis,
        distance_cap_m=float(data["distance_cap_m"]),
        synthetic_surface_grid_size=int(data["synthetic_surface_grid_size"]),
    )
    return config


def basis_points(config: BPSConfig) -> np.ndarray:
    """The lexicographically ordered Cartesian product ``(x, y, z)``."""
    basis = np.asarray(
        list(itertools.product(config.basis_axis_m, repeat=3)),
        dtype=np.float64,
    )
    assert basis.shape == (BPS_DISTANCE_DIM, 3)
    return basis


def bps_fingerprint(config: BPSConfig) -> str:
    """SHA-256 identity stored by future BPS policy checkpoints.

    Include the fully expanded ordered basis rather than merely its four axis
    values, making ordering part of the compatibility contract.
    """
    payload = {
        "basis_m": basis_points(config).tolist(),
        "distance_cap_m": config.distance_cap_m,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_bps_fingerprint(model) -> str | None:
    """Return the BPS identity persisted in an SB3 policy checkpoint."""
    return model.policy_kwargs.get("bps_fingerprint")


def validate_checkpoint_bps(model, config: BPSConfig) -> None:
    """Fail loudly if a policy was built for a different BPS contract."""
    expected = bps_fingerprint(config)
    actual = checkpoint_bps_fingerprint(model)
    if actual != expected:
        raise ValueError(
            "BPS checkpoint mismatch: policy has "
            f"{actual!r}, runtime requires {expected!r}. Distill the policy "
            "onto the current observation layout; do not edit the checkpoint."
        )


def voxel_first_indices(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Indices of the first input point in each occupied metric voxel.

    This is shared by the real StereoSGBM filter and the synthetic cloud path,
    keeping their voxel convention identical without importing camera code
    into the simulator.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("cloud points must be finite")
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    voxels = np.floor(points / voxel_size_m).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    first.sort()
    return first


def encode_bps(points_base: np.ndarray, valid_fraction: float,
               config: BPSConfig) -> BPSMeasurement:
    """Reduce one non-empty visible cloud to the fixed normalized BPS block."""
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("an empty cloud is a measurement miss, not a BPS input")
    if not np.all(np.isfinite(points)):
        raise ValueError("cloud points must be finite")
    if not np.isfinite(valid_fraction) or not 0.0 <= valid_fraction <= 1.0:
        raise ValueError("valid_fraction must be finite and in [0, 1]")

    # np.mean is reduction-order-sensitive.  Canonical lexicographic ordering
    # makes point-order invariance exact instead of merely numerically close.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    center = ordered.mean(axis=0)
    centered = ordered - center
    delta = basis_points(config)[:, None, :] - centered[None, :, :]
    raw_distances = np.sqrt(np.min(np.einsum("bij,bij->bi", delta, delta), axis=1))
    distances = np.clip(raw_distances / config.distance_cap_m, 0.0, 1.0)
    return BPSMeasurement(
        distances=distances.astype(np.float32),
        center_base=center.astype(np.float32),
        valid_fraction=float(np.float32(valid_fraction)),
    )


class BPSObsState:
    """Hold-last/age state for the precise BPS channel.

    Whole-view loss and invalid clouds are represented by not calling
    :meth:`ingest`.  Before the first valid measurement the exact contract is
    zero distances, zero center, zero valid fraction and age at the cap.
    """

    def __init__(self, age_cap_s: float = BPS_AGE_CAP_S):
        if age_cap_s <= 0.0:
            raise ValueError("age_cap_s must be positive")
        self.age_cap_s = float(age_cap_s)
        self._distances = np.zeros(BPS_DISTANCE_DIM, dtype=np.float32)
        self._center = np.zeros(3, dtype=np.float32)
        self._valid_fraction = 0.0
        self._measurement_t = -np.inf

    def ingest(self, t: float, measurement: BPSMeasurement | None) -> None:
        if measurement is None:
            return
        distances = np.asarray(measurement.distances, dtype=np.float32)
        center = np.asarray(measurement.center_base, dtype=np.float32)
        if distances.shape != (BPS_DISTANCE_DIM,) or center.shape != (3,):
            raise ValueError("invalid BPS measurement shape")
        if not np.all(np.isfinite(distances)) or np.any((distances < 0.0)
                                                        | (distances > 1.0)):
            raise ValueError("BPS distances must be finite and in [0, 1]")
        if not np.all(np.isfinite(center)):
            raise ValueError("BPS center must be finite")
        if not 0.0 <= measurement.valid_fraction <= 1.0:
            raise ValueError("BPS valid_fraction must be in [0, 1]")
        self._distances = distances.copy()
        self._center = center.copy()
        self._valid_fraction = float(measurement.valid_fraction)
        self._measurement_t = float(t)

    def serve(self, t: float) -> BPSObservation:
        age = float(np.clip(t - self._measurement_t, 0.0, self.age_cap_s))
        return BPSObservation(
            distances=self._distances.copy(),
            center_base=self._center.copy(),
            age_s=age,
            valid_fraction=self._valid_fraction,
        )
