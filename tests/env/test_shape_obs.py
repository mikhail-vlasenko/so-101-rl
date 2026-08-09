"""Shared live-channel hold/age and static-gate contract."""

import numpy as np
import pytest

from src.shape_obs import (
    MARKER_AGE_CAP_S,
    STATIC_DWELL_S,
    STATIC_SPEED_MAX_M_S,
    ObjectObsState,
    is_static,
)


def test_never_seen_serves_zeros_at_cap():
    live, age = ObjectObsState().serve(123.4)
    np.testing.assert_array_equal(live, np.zeros(3))
    assert age == MARKER_AGE_CAP_S


def test_live_hold_last_and_age():
    state = ObjectObsState()
    state.ingest_live(10.0, np.array([1.0, 2.0, 3.0]))
    live, age = state.serve(10.05)
    np.testing.assert_array_equal(live, [1.0, 2.0, 3.0])
    assert age == pytest.approx(0.05)
    state.ingest_live(10.1, None)
    live, age = state.serve(10.2)
    np.testing.assert_array_equal(live, [1.0, 2.0, 3.0])
    assert age == pytest.approx(0.2)
    assert state.serve(20.0)[1] == MARKER_AGE_CAP_S


def test_serve_clips_small_negative_age():
    state = ObjectObsState()
    state.ingest_live(10.0, np.zeros(3))
    assert state.serve(10.0 - 1e-9)[1] == 0.0


def _track(speed, n=20, dt=0.05):
    t = np.arange(n) * dt
    p = np.zeros((n, 3))
    p[:, 0] = speed * t
    return t, p


def test_is_static_true_for_still_object():
    assert is_static(*_track(0.0))


def test_is_static_false_for_moving_object():
    assert not is_static(*_track(5.0 * STATIC_SPEED_MAX_M_S))


def test_is_static_boundary_speed():
    assert is_static(*_track(0.99 * STATIC_SPEED_MAX_M_S))
    assert not is_static(*_track(1.01 * STATIC_SPEED_MAX_M_S))


def test_is_static_needs_full_dwell():
    n_short = int(0.5 * STATIC_DWELL_S / 0.05)
    assert not is_static(*_track(0.0, n=n_short))
    assert not is_static([1.0], np.zeros((1, 3)))
    assert is_static([0.0, STATIC_DWELL_S], np.zeros((2, 3)))


def test_is_static_judges_only_trailing_window():
    dt = 0.05
    n_move = 10
    n_still = int(STATIC_DWELL_S / dt) + 2
    t = np.arange(n_move + n_still) * dt
    p = np.zeros((len(t), 3))
    p[:n_move, 0] = np.linspace(0.0, 0.5, n_move)
    p[n_move:, 0] = 0.5
    assert is_static(t, p)
    p[-2, 0] += 10.0 * STATIC_SPEED_MAX_M_S * dt
    assert not is_static(t, p)
