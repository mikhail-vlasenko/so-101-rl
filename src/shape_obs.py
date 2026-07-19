"""Dual-channel tag-free object observation: shared sim/real state machine.

The manipulated object is observed through two channels built from the two
cameras' segmentation masks (real: SAM stereo, real/rollout/object_obs.py; sim:
visible-surface sampling, src/base_env.py), both under the hold-last-pose + age
convention the arm markers use:

- **live**: triangulated 3D centroid of the two views' visible-mask centroids —
  fast, biased toward the visible surface, refreshed every frame both cameras
  see the object.
- **precise**: estimated body center + the shape tensor **√M**, refreshed only
  when the object is static (`is_static` on the live-centroid history) and
  well visible in both views (`VISIBLE_FRACTION_MIN`).

√M is the principal square root of the volume's second-moment matrix
M = R·diag(hx²,hy²,hz²)/3·Rᵀ: symmetric, units of meters, invariant under all
box symmetries, continuous over SO(3), defined for any object shape.
dᵀ(√M)²d is the squared spread along direction d.

This module is the single source of truth for the channel state machine and
the static-gate constants: `src/base_env.py` (sim) and
`real/rollout/object_obs.py` (real) both drive an `ObjectObsState` and must
never fork the convention (pinned by the contract test in
tests/env/test_shape_obs.py). Pure numpy — no mujoco, no cv2.
"""

import numpy as np

# Undetected channels keep their last measurement while their age grows, capped
# here; never-measured channels read all-zero with age pinned at the cap. The
# arm markers share the identical convention (src/base_env.py,
# real/rollout/marker_obs.py).
MARKER_AGE_CAP_S = 1.0

# Precise-refresh gate: the object counts as static when its live-centroid
# speed stays below STATIC_SPEED_MAX_M_S over a trailing STATIC_DWELL_S window
# (is_static), and as well visible when the per-view visible fraction is at
# least VISIBLE_FRACTION_MIN (sim: unblocked fraction of camera-facing surface
# points; real: mask area vs its static baseline).
STATIC_SPEED_MAX_M_S = 0.02
STATIC_DWELL_S = 0.5
VISIBLE_FRACTION_MIN = 0.8

# Upper-triangle order served for the symmetric √M — pinned here, everything
# (obs layout, obs_norm, estimators, logs) reads the same order.
SQRTM_UPPER_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")


def box_sqrtm(R, half_extents):
    """√M of a box: R·diag(hx,hy,hz)/√3·Rᵀ, (3,3) symmetric.

    Closed form of sqrtm_from_cov for a uniform box of half extents
    `half_extents` rotated by `R` (3,3): the box's second moment is
    diag(h²)/3, so its principal square root has eigenvalues h/√3.
    """
    R = np.asarray(R, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    return (R * (h / np.sqrt(3.0))) @ R.T


def sqrtm_from_cov(cov):
    """Principal square root of a symmetric PSD (3,3) covariance via eigh.

    Tiny negative eigenvalues (numerical noise from a degenerate voxel set)
    are clamped to zero instead of producing NaNs.
    """
    w, V = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def sqrtm_upper(S):
    """(6,) upper triangle of a symmetric (3,3), in SQRTM_UPPER_ORDER."""
    S = np.asarray(S, dtype=np.float64)
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[0, 2], S[1, 2]])


def sqrtm_from_upper(u):
    """Inverse of sqrtm_upper: (6,) in SQRTM_UPPER_ORDER -> symmetric (3,3)."""
    xx, yy, zz, xy, xz, yz = np.asarray(u, dtype=np.float64)
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]])


def is_static(times, centroids):
    """True when the live-centroid track proves the object static: the samples
    span at least STATIC_DWELL_S and every consecutive-sample speed inside the
    trailing STATIC_DWELL_S window stays at or below STATIC_SPEED_MAX_M_S.

    `times` (N,) ascending, `centroids` (N, 3) — the detected live
    measurements only (misses simply don't appear; a long detection gap makes
    the single bridging speed the judgment, which is the honest reading of the
    data available).
    """
    times = np.asarray(times, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    assert times.shape[0] == centroids.shape[0], (times.shape, centroids.shape)
    if times.shape[0] < 2:
        return False
    # Float slack: a window seeded exactly one dwell back must count as spanning.
    if times[-1] - times[0] < STATIC_DWELL_S - 1e-9:
        return False
    # Trailing window plus the last sample before it, so the speeds cover the
    # whole dwell even when samples straddle the window edge.
    start = np.searchsorted(times, times[-1] - STATIC_DWELL_S, side="right") - 1
    start = max(start, 0)
    t = times[start:]
    p = centroids[start:]
    dt = np.diff(t)
    assert np.all(dt > 0.0), "times must be strictly ascending"
    speeds = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    return bool(np.all(speeds <= STATIC_SPEED_MAX_M_S))


class ObjectObsState:
    """Pure dual-channel hold-last/age state machine for one object.

    Feed measurements with `ingest_live` / `ingest_precise` (any monotonic
    time base; sim uses sim seconds, real uses time.monotonic()), read the
    served observation with `serve`. Channels never seen read zeros with age
    pinned at MARKER_AGE_CAP_S. The caller owns the refresh *gating* (static +
    visible for precise); this class only holds and ages.
    """

    def __init__(self):
        self._live = np.zeros(3)
        self._live_t = -np.inf
        self._center = np.zeros(3)
        self._sqrtm6 = np.zeros(6)
        self._precise_t = -np.inf

    def ingest_live(self, t, centroid):
        """One live-channel frame at time `t`: `centroid` (3,) when both views
        measured the object, None on a miss (the held state simply ages)."""
        if centroid is None:
            return
        self._live = np.asarray(centroid, dtype=np.float64).copy()
        self._live_t = float(t)

    def ingest_precise(self, t, center, sqrtm):
        """One precise refresh at time `t`: body center (3,) and √M as a
        symmetric (3,3) (stored as its sqrtm_upper 6-vector)."""
        self._center = np.asarray(center, dtype=np.float64).copy()
        self._sqrtm6 = sqrtm_upper(sqrtm)
        self._precise_t = float(t)

    def serve(self, t):
        """(live (3,), live_age, center (3,), sqrtm6 (6,), precise_age) at
        time `t`. Ages are seconds since each channel's last ingest, clipped
        below at 0 (frame schedules may land an epsilon past the obs instant)
        and capped at MARKER_AGE_CAP_S."""
        live_age = float(np.clip(t - self._live_t, 0.0, MARKER_AGE_CAP_S))
        precise_age = float(np.clip(t - self._precise_t, 0.0, MARKER_AGE_CAP_S))
        return (self._live.copy(), live_age,
                self._center.copy(), self._sqrtm6.copy(), precise_age)


class ObjectChannelDriver:
    """The full dual-channel update logic on top of ObjectObsState: live
    ingestion, the static-gate history it feeds, and the precise-refresh gate.

    This is the single implementation both twins drive — the sim env
    (src/base_env._ingest_frame) and the real rollout
    (real/rollout/object_obs.ObjectSource) — so the hold-last/age/static-gate
    behavior cannot fork. The contract test in tests/real/test_object_twin.py
    pins the sim env to it.

    The caller supplies measurements; this class owns state:
    - `ingest_live(t, live, gate_point=None)` records a live measurement (None
      = miss, state just ages) and extends the gate history with `gate_point`
      (defaults to `live`; the sim passes its pre-noise measurement — injected
      DR noise is not real static jitter and must not blind the gate).
    - `gate_open(vis_frac)` answers whether a precise refresh is allowed NOW:
      the object is well visible in every view and its gate-history track
      proves it static (is_static).
    - `ingest_precise(t, center, sqrtm)` folds a precise estimate in; cheap
      estimators call it right after a passing gate, the real hull worker
      whenever its window average updates.
    - `seed_static(t, point)`: pre-window static evidence — the sim's
      pre-episode world and the real rig's settle-before-start warmup both
      know the object stood still before the first frame.
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
        # One measurement per instant — a re-processed frame at the same time
        # replaces rather than duplicates, keeping is_static's times strictly
        # ascending.
        if self._hist_t and t == self._hist_t[-1]:
            self._hist_p[-1] = point
        else:
            self._hist_t.append(float(t))
            self._hist_p.append(point)
        # Trim to the trailing dwell window (plus the pre-window sample
        # is_static keeps for edge coverage).
        cutoff = t - 2.0 * STATIC_DWELL_S
        while len(self._hist_t) > 2 and self._hist_t[0] < cutoff:
            self._hist_t.pop(0)
            self._hist_p.pop(0)

    def static_now(self):
        """Whether the gate-history track currently proves the object static
        (the visibility half of the gate is the caller's measurement)."""
        return is_static(self._hist_t, self._hist_p)

    def gate_open(self, vis_frac):
        return bool(np.all(np.asarray(vis_frac) >= VISIBLE_FRACTION_MIN)
                    and self.static_now())

    def ingest_precise(self, t, center, sqrtm):
        self.state.ingest_precise(t, center, sqrtm)

    def serve(self, t):
        return self.state.serve(t)
