"""Synthetic-data tests for the sponge tag placement solver
(real/tracking/tag_body_calib.py): known (u, v, yaw) placements generate
pairwise relative-transform measurements; the solver must recover them."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_inv
from real.tracking.tag_body_calib import (
    face_frame,
    solve_tag_placements,
    tag_body_transform,
)

HALF = np.array([0.03, 0.02, 0.0125])

FACES = {1: "+z", 3: "-z", 4: "+y", 5: "-y", 6: "+x", 7: "-x"}


def test_face_frames_are_right_handed_outward():
    for face in ("+x", "-x", "+y", "-y", "+z", "-z"):
        origin, R = face_frame(face, HALF)
        a, b, n = R[:, 0], R[:, 1], R[:, 2]
        np.testing.assert_allclose(np.cross(a, b), n, atol=1e-12)
        # Origin sits at the face center: along the normal at the half extent.
        np.testing.assert_allclose(origin @ n, np.abs(origin).max(), atol=1e-12)
        assert origin @ n > 0.0  # outward


def test_tag_body_transform_geometry():
    """A tag's +z is the outward face normal; (u, v) moves it in-plane."""
    T = tag_body_transform("+z", HALF, u=0.005, v=-0.003, yaw=0.4)
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.005, -0.003, HALF[2]], atol=1e-12)
    T = tag_body_transform("-x", HALF, u=0.0, v=0.0, yaw=0.0)
    np.testing.assert_allclose(T[:3, 2], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [-HALF[0], 0.0, 0.0], atol=1e-12)


def _synthetic_pairs(rng, truth, n_pairs, noise_pos=0.0, noise_rot=0.0):
    """Random co-visibility pairs with optional measurement noise."""
    T = {tag: tag_body_transform(FACES[tag], HALF, *params)
         for tag, params in truth.items()}
    ids = sorted(truth)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(ids, size=2, replace=False)
        T_meas = mat_inv(T[i]) @ T[j]
        if noise_pos > 0.0:
            T_noise = np.eye(4)
            T_noise[:3, :3] = Rotation.from_rotvec(
                rng.normal(0, noise_rot, 3)).as_matrix()
            T_noise[:3, 3] = rng.normal(0, noise_pos, 3)
            T_meas = T_meas @ T_noise
        pairs.append((int(i), int(j), T_meas))
    return pairs


def _random_truth(rng):
    return {tag: (rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004),
                  rng.uniform(-0.3, 0.3)) for tag in FACES}


def test_solver_recovers_exact_placements():
    rng = np.random.default_rng(0)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=60)
    solved, res = solve_tag_placements(pairs, FACES, HALF)
    for tag, (u, v, yaw) in truth.items():
        su, sv, syaw = solved[tag]
        assert abs(su - u) < 1e-4, tag
        assert abs(sv - v) < 1e-4, tag
        assert abs(syaw - yaw) < 1e-3, tag
    assert np.sqrt((res[:, :3] ** 2).sum(1).mean()) < 1e-4


def test_solver_tolerates_measurement_noise():
    """With ~1 mm / ~0.5 deg pair noise the recovered placements stay within
    a millimeter and a degree — the accuracy the GT pipeline needs."""
    rng = np.random.default_rng(1)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=300,
                             noise_pos=0.001, noise_rot=0.01)
    solved, _ = solve_tag_placements(pairs, FACES, HALF)
    for tag, (u, v, yaw) in truth.items():
        su, sv, syaw = solved[tag]
        assert abs(su - u) < 1e-3, tag
        assert abs(sv - v) < 1e-3, tag
        assert abs(syaw - yaw) < np.radians(1.0), tag


def test_solver_transform_roundtrip():
    """The full T_body_tag rebuilt from solved params maps tag-frame points to
    the same body-frame points as the ground truth."""
    rng = np.random.default_rng(2)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=80)
    solved, _ = solve_tag_placements(pairs, FACES, HALF)
    probe = np.array([0.01, -0.01, 0.0, 1.0])
    for tag in truth:
        T_true = tag_body_transform(FACES[tag], HALF, *truth[tag])
        T_est = tag_body_transform(FACES[tag], HALF, *solved[tag])
        np.testing.assert_allclose(T_est @ probe, T_true @ probe, atol=2e-4)


def test_solver_rejects_undeclared_tag():
    rng = np.random.default_rng(3)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=10)
    with pytest.raises(AssertionError):
        solve_tag_placements(pairs, {1: "+z"}, HALF)
