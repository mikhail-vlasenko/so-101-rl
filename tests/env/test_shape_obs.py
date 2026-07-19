"""src/shape_obs.py contract: √M math, the dual-channel hold-last/age state
machine, and the static gate — the module both the sim env and the real
rollout import, so these tests pin the shared convention itself."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shape_obs import (
    MARKER_AGE_CAP_S,
    STATIC_DWELL_S,
    STATIC_SPEED_MAX_M_S,
    ObjectObsState,
    box_sqrtm,
    is_static,
    sqrtm_from_cov,
    sqrtm_from_upper,
    sqrtm_upper,
)

HALF = np.array([0.03, 0.02, 0.0125])


# ---------------------------------------------------------------- √M math


def test_box_sqrtm_identity_rotation():
    S = box_sqrtm(np.eye(3), HALF)
    np.testing.assert_allclose(S, np.diag(HALF / np.sqrt(3.0)), atol=1e-12)


def test_box_sqrtm_matches_eigh_route():
    """Closed form vs the covariance route: √M from box_sqrtm must equal
    sqrtm_from_cov of M = R diag(h²)/3 Rᵀ for random rotations."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = Rotation.random(random_state=rng).as_matrix()
        M = (R * (HALF ** 2 / 3.0)) @ R.T
        np.testing.assert_allclose(box_sqrtm(R, HALF), sqrtm_from_cov(M), atol=1e-12)


def test_box_sqrtm_square_is_second_moment():
    """(√M)² = M: the directional spread contract dᵀ(√M)²d = dᵀMd."""
    rng = np.random.default_rng(1)
    R = Rotation.random(random_state=rng).as_matrix()
    S = box_sqrtm(R, HALF)
    M = (R * (HALF ** 2 / 3.0)) @ R.T
    np.testing.assert_allclose(S @ S, M, atol=1e-12)


def test_box_sqrtm_invariant_under_box_symmetries():
    """A 180° flip about any box axis is a symmetry of the box — √M must not
    change (the whole reason the obs carries √M instead of a rotation)."""
    rng = np.random.default_rng(2)
    R = Rotation.random(random_state=rng).as_matrix()
    S = box_sqrtm(R, HALF)
    for axis in range(3):
        flip = -np.eye(3)
        flip[axis, axis] = 1.0
        np.testing.assert_allclose(box_sqrtm(R @ flip, HALF), S, atol=1e-12)


def test_sqrtm_from_cov_clamps_tiny_negatives():
    cov = np.diag([1e-4, 1e-6, -1e-18])
    S = sqrtm_from_cov(cov)
    assert np.all(np.isfinite(S))
    w = np.linalg.eigvalsh(S)
    assert np.all(w >= 0.0)


def test_sqrtm_upper_order_and_roundtrip():
    S = np.array([[1.0, 4.0, 5.0],
                  [4.0, 2.0, 6.0],
                  [5.0, 6.0, 3.0]])
    u = sqrtm_upper(S)
    np.testing.assert_array_equal(u, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # xx,yy,zz,xy,xz,yz
    np.testing.assert_array_equal(sqrtm_from_upper(u), S)


# ------------------------------------------------------- state machine


def test_never_seen_serves_zeros_at_cap():
    s = ObjectObsState()
    live, live_age, center, sqrtm6, precise_age = s.serve(123.4)
    np.testing.assert_array_equal(live, np.zeros(3))
    np.testing.assert_array_equal(center, np.zeros(3))
    np.testing.assert_array_equal(sqrtm6, np.zeros(6))
    assert live_age == MARKER_AGE_CAP_S
    assert precise_age == MARKER_AGE_CAP_S


def test_live_hold_last_and_age():
    s = ObjectObsState()
    s.ingest_live(10.0, np.array([1.0, 2.0, 3.0]))
    live, age, *_ = s.serve(10.05)
    np.testing.assert_array_equal(live, [1.0, 2.0, 3.0])
    assert age == pytest.approx(0.05)
    # A miss (None) leaves the held value; age keeps growing.
    s.ingest_live(10.1, None)
    live, age, *_ = s.serve(10.2)
    np.testing.assert_array_equal(live, [1.0, 2.0, 3.0])
    assert age == pytest.approx(0.2)
    # Ages cap; a fresh measurement resets them.
    _, age, *_ = s.serve(20.0)
    assert age == MARKER_AGE_CAP_S
    s.ingest_live(20.0, np.array([4.0, 5.0, 6.0]))
    live, age, *_ = s.serve(20.0)
    np.testing.assert_array_equal(live, [4.0, 5.0, 6.0])
    assert age == 0.0


def test_precise_channel_independent_of_live():
    s = ObjectObsState()
    S = box_sqrtm(np.eye(3), HALF)
    s.ingest_precise(5.0, np.array([0.2, 0.0, 0.02]), S)
    live, live_age, center, sqrtm6, precise_age = s.serve(5.3)
    assert live_age == MARKER_AGE_CAP_S          # live never seen
    np.testing.assert_array_equal(center, [0.2, 0.0, 0.02])
    np.testing.assert_array_equal(sqrtm6, sqrtm_upper(S))
    assert precise_age == pytest.approx(0.3)


def test_serve_clips_small_negative_age():
    """Frame schedules can land a capture an epsilon past the obs instant."""
    s = ObjectObsState()
    s.ingest_live(10.0, np.zeros(3))
    _, age, *_ = s.serve(10.0 - 1e-9)
    assert age == 0.0


# ---------------------------------------------------------- static gate


def _track(speed, n=20, dt=0.05):
    """A straight-line track at constant speed sampled every dt."""
    t = np.arange(n) * dt
    p = np.zeros((n, 3))
    p[:, 0] = speed * t
    return t, p


def test_is_static_true_for_still_object():
    t, p = _track(0.0)
    assert is_static(t, p)


def test_is_static_false_for_moving_object():
    t, p = _track(5.0 * STATIC_SPEED_MAX_M_S)
    assert not is_static(t, p)


def test_is_static_boundary_speed():
    t, p = _track(0.99 * STATIC_SPEED_MAX_M_S)
    assert is_static(t, p)
    t, p = _track(1.01 * STATIC_SPEED_MAX_M_S)
    assert not is_static(t, p)


def test_is_static_needs_full_dwell():
    """A short still track must not count: the window has to span the dwell."""
    n_short = int(0.5 * STATIC_DWELL_S / 0.05)
    t, p = _track(0.0, n=n_short)
    assert not is_static(t, p)
    assert not is_static([1.0], np.zeros((1, 3)))
    # Exactly one dwell span passes (the reset seeding relies on this).
    assert is_static([0.0, STATIC_DWELL_S], np.zeros((2, 3)))


def test_is_static_judges_only_trailing_window():
    """Motion older than the trailing dwell window is forgiven — the object
    was carried, then set down and left alone for a full dwell."""
    dt = 0.05
    n_move = 10
    n_still = int(STATIC_DWELL_S / dt) + 2
    t = np.arange(n_move + n_still) * dt
    p = np.zeros((len(t), 3))
    p[:n_move, 0] = np.linspace(0.0, 0.5, n_move)      # fast carry
    p[n_move:, 0] = 0.5                                 # at rest
    assert is_static(t, p)
    # Motion INSIDE the trailing window still blocks.
    p[-2, 0] = 0.5 + 10.0 * STATIC_SPEED_MAX_M_S * dt
    assert not is_static(t, p)
