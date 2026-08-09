"""Fixed BPS transform, metadata, and held-state contract."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from src.bps import (
    BPS_DISTANCE_DIM,
    BPS_AGE_CAP_S,
    BPSConfig,
    BPSObsState,
    basis_points,
    bps_fingerprint,
    encode_bps,
    load_bps_config,
    validate_checkpoint_bps,
    voxel_first_indices,
)


CONFIG = BPSConfig(
    basis_axis_m=(-0.04, -0.01, 0.01, 0.04),
    distance_cap_m=0.08,
    synthetic_surface_grid_size=15,
)


def _symmetric_cloud():
    axis = np.array([-0.03, -0.015, 0.015, 0.03])
    return np.asarray(list(itertools.product(axis, repeat=3)), dtype=np.float64)


def test_config_owns_exact_lexicographic_basis_and_fingerprint():
    configured = load_bps_config()
    assert configured == CONFIG
    expected = np.asarray(list(itertools.product(CONFIG.basis_axis_m, repeat=3)))
    np.testing.assert_array_equal(basis_points(configured), expected)
    assert basis_points(configured).shape == (BPS_DISTANCE_DIM, 3)
    assert bps_fingerprint(configured) == bps_fingerprint(CONFIG)
    changed = BPSConfig(CONFIG.basis_axis_m, 0.081, 15)
    assert bps_fingerprint(changed) != bps_fingerprint(CONFIG)


def test_checkpoint_fingerprint_rejects_missing_or_changed_contract():
    valid = SimpleNamespace(policy_kwargs={"bps_fingerprint": bps_fingerprint(CONFIG)})
    validate_checkpoint_bps(valid, CONFIG)
    missing = SimpleNamespace(policy_kwargs={})
    with pytest.raises(ValueError, match="BPS checkpoint mismatch"):
        validate_checkpoint_bps(missing, CONFIG)
    changed = BPSConfig(CONFIG.basis_axis_m, 0.081, 15)
    with pytest.raises(ValueError, match="BPS checkpoint mismatch"):
        validate_checkpoint_bps(valid, changed)


def test_transform_is_exactly_point_order_invariant():
    rng = np.random.default_rng(7)
    points = rng.normal(0.0, 0.02, size=(257, 3)) + (0.2, 0.01, 0.04)
    reference = encode_bps(points, 0.75, CONFIG)
    shuffled = encode_bps(points[rng.permutation(len(points))], 0.75, CONFIG)
    np.testing.assert_array_equal(shuffled.distances, reference.distances)
    np.testing.assert_array_equal(shuffled.center_base, reference.center_base)


def test_distances_are_normalized_finite_and_clip_at_cap():
    at_center = encode_bps(np.zeros((1, 3)), 1.0, CONFIG)
    expected = np.linalg.norm(basis_points(CONFIG), axis=1) / CONFIG.distance_cap_m
    np.testing.assert_allclose(at_center.distances, np.clip(expected, 0.0, 1.0))
    assert np.all(np.isfinite(at_center.distances))
    assert np.all((at_center.distances >= 0.0) & (at_center.distances <= 1.0))
    clipped = encode_bps(np.array([[-0.2, 0.0, 0.0],
                                   [0.2, 0.0, 0.0]]), 1.0, CONFIG)
    assert np.all(clipped.distances == 1.0)

    with pytest.raises(ValueError, match="empty cloud"):
        encode_bps(np.empty((0, 3)), 0.0, CONFIG)
    with pytest.raises(ValueError, match="finite"):
        encode_bps(np.array([[np.nan, 0.0, 0.0]]), 1.0, CONFIG)
    with pytest.raises(ValueError, match="valid_fraction"):
        encode_bps(np.zeros((1, 3)), 1.1, CONFIG)


def test_submillimetre_jitter_changes_distances_smoothly():
    points = _symmetric_cloud()
    reference = encode_bps(points, 1.0, CONFIG)
    rng = np.random.default_rng(3)
    jitter = rng.uniform(-0.00025, 0.00025, size=points.shape)
    changed = encode_bps(points + jitter, 1.0, CONFIG)
    delta = np.abs(changed.distances - reference.distances)
    assert np.max(delta) < 0.01
    assert np.any(delta > 0.0)


def test_removing_centroid_preserving_points_degrades_predictably():
    # Nested symmetric shells keep the measured centroid fixed. Removing the
    # inner shell can only increase each basis-to-cloud nearest distance.
    outer = np.asarray(list(itertools.product((-0.03, 0.03), repeat=3)))
    inner = np.asarray(list(itertools.product((-0.01, 0.01), repeat=3)))
    full = encode_bps(np.concatenate((outer, inner)), 1.0, CONFIG)
    sparse = encode_bps(outer, 0.5, CONFIG)
    assert np.all(sparse.distances >= full.distances)
    assert np.any(sparse.distances > full.distances)


def test_never_measured_and_held_age_contract():
    state = BPSObsState()
    unseen = state.serve(100.0)
    np.testing.assert_array_equal(unseen.distances, np.zeros(BPS_DISTANCE_DIM))
    np.testing.assert_array_equal(unseen.center_base, np.zeros(3))
    assert unseen.valid_fraction == 0.0
    assert unseen.age_s == BPS_AGE_CAP_S

    measurement = encode_bps(_symmetric_cloud() + (0.2, 0.0, 0.03), 0.6, CONFIG)
    state.ingest(10.0, measurement)
    fresh = state.serve(10.0)
    assert fresh.age_s == 0.0
    state.ingest(10.1, None)
    held = state.serve(10.25)
    np.testing.assert_array_equal(held.distances, fresh.distances)
    np.testing.assert_array_equal(held.center_base, fresh.center_base)
    assert held.valid_fraction == fresh.valid_fraction
    assert held.age_s == pytest.approx(0.25)


def test_voxel_selection_contract_is_deterministic():
    points = np.array([[0.001, 0.001, 0.001],
                       [0.009, 0.009, 0.009],
                       [0.011, 0.001, 0.001]])
    np.testing.assert_array_equal(voxel_first_indices(points, 0.01), [0, 2])
