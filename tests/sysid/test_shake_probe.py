"""Contract tests for the shake-probe sweep profile (pure math, no hardware)."""

import numpy as np

from scripts.shake_probe import sweep_offset

A, V, DWELL = 150, 100.0, 2.0
DT = 0.005


def _profile(dwell_s: float) -> tuple[np.ndarray, np.ndarray]:
    period = 4 * A / V + 2 * dwell_s
    ts = np.arange(0, 2 * period, DT)
    return ts, np.array([sweep_offset(t, A, V, dwell_s) for t in ts])


def test_starts_at_zero_and_spans_full_range():
    _, offs = _profile(DWELL)
    assert offs[0] == 0
    assert offs.max() == A
    assert offs.min() == -A


def test_continuous_no_lunges():
    _, offs = _profile(DWELL)
    # per-sample jump bounded by the commanded slope (+1 raw for rounding)
    assert np.max(np.abs(np.diff(offs))) <= V * DT + 1


def test_dwells_at_extremes():
    ts, offs = _profile(DWELL)
    period = 4 * A / V + 2 * DWELL
    cycle = offs[ts < period]
    for extreme in (A, -A):
        t_at = np.sum(cycle == extreme) * DT
        assert abs(t_at - DWELL) < 0.05, (extreme, t_at)


def test_zero_dwell_is_plain_triangle():
    ts, offs = _profile(0.0)
    assert offs[0] == 0
    assert offs.max() == A
    assert offs.min() == -A
    # strictly no flat segments longer than the rounding plateau
    longest_flat = max(
        len(list(g)) for g in np.split(offs, np.where(np.diff(offs) != 0)[0] + 1)
    )
    assert longest_flat * DT < 0.1
