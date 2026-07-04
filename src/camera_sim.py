"""Discrete-frame camera simulation for observation latency (sim2real).

The real marker pipeline delivers poses from discrete camera frames: a frame
samples the world at its capture instant, becomes available to the control
loop `delay` seconds later (mid-exposure -> sensor readout -> USB transfer ->
MJPG decode -> AprilTag detection; measure it with sysid/probe_cam_latency.py),
and each policy tick then consumes the newest available frame
(real/marker_obs.py). With a 30 fps camera against 15 Hz control the consumed
pose is `delay` plus zero-to-one frame interval old, and that staleness
follows a sawtooth that drifts as the camera and control clocks beat against
each other. Joint encoders have no analog of any of this: the bus read is ~2 ms,
effectively fresh at a 66.7 ms tick, so only camera-derived observations
(markers, cube) go through this model.

CameraSim reproduces the timing skeleton while staying agnostic of what a
"frame" contains: the env records an opaque world-state snapshot every physics
substep (`record`), CameraSim captures frames from that history on a
fixed-period schedule — per-episode random phase, per-episode pipeline delay
sampled from `delay_range_s`, per-frame capture-time jitter whose random walk
stands in for real clock drift — and `observe` returns the newest frame whose
availability time has passed. Each frame is processed exactly once, at capture
time, through the env's `capture_fn` (detector dropout and measurement noise
freeze per frame, like a real detection).
"""

from collections import deque


class CameraSim:
    """Frame-schedule bookkeeping for one episode; states/frames are opaque."""

    # Float slack (s) so a frame due exactly at an obs time counts as due:
    # sim time and the frame schedule accumulate through different float sums.
    _EPS = 1e-6

    def __init__(self, frame_s: float, delay_range_s: tuple[float, float],
                 jitter_s: float, control_dt: float):
        frame_s = float(frame_s)
        delay_lo, delay_hi = float(delay_range_s[0]), float(delay_range_s[1])
        jitter_s = float(jitter_s)
        assert frame_s > 0.0, f"frame_s must be positive, got {frame_s}"
        assert 0.0 <= delay_lo <= delay_hi, \
            f"delay range must satisfy 0 <= lo <= hi, got ({delay_lo}, {delay_hi})"
        assert 0.0 <= jitter_s < frame_s / 4, \
            f"jitter_s={jitter_s} must be well under frame_s={frame_s}"
        self.frame_s = frame_s
        self.delay_lo = delay_lo
        self.delay_hi = delay_hi
        self.jitter_s = jitter_s
        # History must cover every capture time still unprocessed at the next
        # observe(): captures due then lie within one control period of it, so
        # one period plus a frame interval of slack bounds the lookback.
        self._keep_s = control_dt + frame_s
        self._random_phase = True
        self._hist: deque = deque()      # (t, state), t ascending
        self._pending: deque = deque()   # (avail_t, frame), avail_t ascending
        self._frame = None               # newest consumed frame
        self._next_capture_t = 0.0
        self._delay = 0.0

    @classmethod
    def synchronous(cls, control_dt: float) -> "CameraSim":
        """Zero-latency camera (dr=none / plain env construction): one frame is
        captured exactly at every control tick and available immediately, so
        marker/cube obs equal the current state and reset consumes no RNG."""
        cam = cls(frame_s=control_dt, delay_range_s=(0.0, 0.0), jitter_s=0.0,
                  control_dt=control_dt)
        cam._random_phase = False
        return cam

    def reset(self, rng, t: float, state, capture_fn):
        """Start an episode at sim time `t` with the world in `state`. The
        pre-episode world is static, so the frame in flight at reset shows
        exactly the reset state: a fresh initial frame, returned as consumed."""
        self._hist.clear()
        self._pending.clear()
        self._hist.append((t, state))
        self._frame = capture_fn(state)
        if self._random_phase:
            self._delay = rng.uniform(self.delay_lo, self.delay_hi)
            self._next_capture_t = t + rng.uniform(0.0, self.frame_s)
        else:
            self._delay = 0.0
            self._next_capture_t = t + self.frame_s
        return self._frame

    def record(self, t: float, state) -> None:
        """Log the world state at sim time `t` (call once per physics substep)."""
        self._hist.append((t, state))
        keep_after = t - self._keep_s
        while self._hist[0][0] < keep_after:
            self._hist.popleft()

    def _state_at(self, t: float):
        """Recorded state nearest to `t` — history spacing is one substep, so
        the lookup error is at most half a substep."""
        return min(self._hist, key=lambda entry: abs(entry[0] - t))[1]

    def observe(self, t: float, rng, capture_fn):
        """Advance the schedule to sim time `t` and return the frame the policy
        sees: the newest one whose capture time + pipeline delay has passed."""
        while self._next_capture_t <= t + self._EPS:
            capture_t = self._next_capture_t
            frame = capture_fn(self._state_at(capture_t))
            self._pending.append((capture_t + self._delay, frame))
            step = self.frame_s
            if self.jitter_s > 0.0:
                step = max(self.frame_s / 2.0, step + rng.normal(0.0, self.jitter_s))
            self._next_capture_t = capture_t + step
        while self._pending and self._pending[0][0] <= t + self._EPS:
            self._frame = self._pending.popleft()[1]
        return self._frame
