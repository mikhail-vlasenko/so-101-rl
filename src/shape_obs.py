"""Tag-free live object observation and static-refresh gate.

The fast object channel is the triangulated centroid of the two cameras'
segmentation masks.  Sim and real both feed this module so hold-last/age and
the static gate cannot drift between them.  The precise visible-surface
observation is owned separately by :mod:`src.bps`; this module deliberately
contains no shape estimator or precise-channel representation.
"""

import numpy as np


# Never-measured channels are zero with age pinned at this cap.  Arm markers
# and both object channels use the same convention.
MARKER_AGE_CAP_S = 1.0

# A dense refresh is allowed only after the live centroid has remained slow
# for a complete dwell and both views meet their visibility gate.
STATIC_SPEED_MAX_M_S = 0.02
STATIC_DWELL_S = 0.5
VISIBLE_FRACTION_MIN = 0.8


def is_static(times, centroids):
    """Whether the trailing live-centroid track proves the object static."""
    times = np.asarray(times, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    assert times.shape[0] == centroids.shape[0], (times.shape, centroids.shape)
    if times.shape[0] < 2:
        return False
    if times[-1] - times[0] < STATIC_DWELL_S - 1e-9:
        return False
    start = np.searchsorted(times, times[-1] - STATIC_DWELL_S, side="right") - 1
    start = max(start, 0)
    t = times[start:]
    p = centroids[start:]
    dt = np.diff(t)
    assert np.all(dt > 0.0), "times must be strictly ascending"
    speeds = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    return bool(np.all(speeds <= STATIC_SPEED_MAX_M_S))


class ObjectObsState:
    """Hold-last/age state for the fast live centroid."""

    def __init__(self):
        self._live = np.zeros(3)
        self._live_t = -np.inf

    def ingest_live(self, t, centroid):
        if centroid is None:
            return
        centroid = np.asarray(centroid, dtype=np.float64)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            raise ValueError("live centroid must be a finite (3,) vector")
        self._live = centroid.copy()
        self._live_t = float(t)

    def serve(self, t):
        """Return ``(live (3,), age_s)`` at time ``t``."""
        age = float(np.clip(t - self._live_t, 0.0, MARKER_AGE_CAP_S))
        return self._live.copy(), age


class ObjectChannelDriver:
    """Live-channel state plus the shared static-gate history.

    ``ingest_live`` refreshes the held centroid and extends the gate history.
    ``gate_open`` combines the history decision with per-view visibility.  A
    caller that gets a true result may publish a BPS measurement to its own
    :class:`src.bps.BPSObsState`.
    """

    def __init__(self):
        self.state = ObjectObsState()
        self._hist_t: list[float] = []
        self._hist_p: list[np.ndarray] = []

    def seed_static(self, t, point):
        assert not self._hist_t, "seed_static must precede the first ingest"
        self._hist_t.append(float(t))
        self._hist_p.append(np.asarray(point, dtype=np.float64).copy())

    def ingest_live(self, t, live, gate_point=None):
        if live is None:
            return
        self.state.ingest_live(t, live)
        point = np.asarray(live if gate_point is None else gate_point,
                           dtype=np.float64).copy()
        if self._hist_t and t == self._hist_t[-1]:
            self._hist_p[-1] = point
        else:
            self._hist_t.append(float(t))
            self._hist_p.append(point)
        cutoff = t - 2.0 * STATIC_DWELL_S
        while len(self._hist_t) > 2 and self._hist_t[0] < cutoff:
            self._hist_t.pop(0)
            self._hist_p.pop(0)

    def static_now(self):
        return is_static(self._hist_t, self._hist_p)

    def gate_open(self, vis_frac):
        return bool(np.all(np.asarray(vis_frac) >= VISIBLE_FRACTION_MIN)
                    and self.static_now())

    def serve(self, t):
        return self.state.serve(t)
