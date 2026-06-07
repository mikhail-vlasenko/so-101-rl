"""Contract tests for the shared real-arm command shaping in real.twin.control.

Pure functions, no hardware: clamp limits the per-tick jump, and the sub-target
stream emits exactly n_interp waypoints sliding monotonically to the target.
"""
import numpy as np

from real.twin.constants import MAX_RAW_DELTA_PER_STEP
from real.twin.control import clamp_raw_delta, stream_sub_targets


def test_clamp_limits_large_jump():
    prev = np.full(6, 2048, dtype=np.int64)
    target = prev + 1000  # far beyond the per-tick cap
    out = clamp_raw_delta(prev, target)
    assert np.all(out - prev == MAX_RAW_DELTA_PER_STEP)


def test_clamp_passes_in_range_delta():
    prev = np.full(6, 2048, dtype=np.int64)
    target = prev + np.array(
        [1, -2, 3, 0, MAX_RAW_DELTA_PER_STEP, -MAX_RAW_DELTA_PER_STEP], dtype=np.int64
    )
    assert np.array_equal(clamp_raw_delta(prev, target), target)


def test_stream_emits_n_waypoints_ending_at_target():
    prev = np.full(6, 2048, dtype=np.int64)
    target = prev + 7
    writes: list[np.ndarray] = []
    stream_sub_targets(prev, target, n_interp=7, sub_dt=0.0,
                       write=lambda r: writes.append(r.copy()))
    assert len(writes) == 7
    assert np.array_equal(writes[-1], target)
    # Each waypoint lies within [prev, target] and never moves backward.
    for earlier, later in zip(writes, writes[1:]):
        assert np.all(later >= earlier)
    assert np.all(writes[0] >= prev) and np.all(writes[-1] <= target)


def test_stream_single_waypoint_hits_target():
    prev = np.zeros(6, dtype=np.int64)
    target = np.full(6, 10, dtype=np.int64)
    writes: list[np.ndarray] = []
    stream_sub_targets(prev, target, n_interp=1, sub_dt=0.0,
                       write=lambda r: writes.append(r.copy()))
    assert len(writes) == 1
    assert np.array_equal(writes[0], target)
