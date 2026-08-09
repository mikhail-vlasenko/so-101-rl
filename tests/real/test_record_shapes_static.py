"""The capture viewer's static label must tolerate tag-pose jitter."""

import numpy as np

from real.tracking.shape_dataset import CausalMeanPosition
from src.shape_obs import STATIC_DWELL_S, is_static


def _filtered_track(points, dt=1.0 / 30.0):
    filt = CausalMeanPosition(0.15)
    times = np.arange(len(points), dtype=np.float64) * dt
    filtered = [filt.update(t, point) for t, point in zip(times, points)]
    return times, np.asarray(filtered)


def test_static_label_rejects_submillimetre_alternating_tag_jitter():
    count = int(np.ceil(STATIC_DWELL_S * 30.0)) + 8
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = np.where(np.arange(count) % 2 == 0, -0.0006, 0.0006)

    times, filtered = _filtered_track(points)

    assert is_static(times, filtered)


def test_static_label_still_rejects_deliberate_motion():
    count = int(np.ceil(STATIC_DWELL_S * 30.0)) + 8
    times = np.arange(count, dtype=np.float64) / 30.0
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = 0.10 * times

    times, filtered = _filtered_track(points)

    assert not is_static(times, filtered)
