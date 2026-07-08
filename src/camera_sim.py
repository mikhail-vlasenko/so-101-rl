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
stands in for real clock drift — and `observe` returns every frame whose
availability time has passed since the last call, each tagged with its capture
time so the consumer can age it. Each frame is processed exactly once, at
capture time, through the env's `capture_fn` (detector dropout and measurement
noise freeze per frame, like a real detection).
"""

import math
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
        self._pending: deque = deque()   # (avail_t, capture_t, frame), avail_t ascending
        self._next_capture_t = 0.0
        self._delay = 0.0

    @property
    def pipeline_delay_s(self) -> float:
        """This episode's sampled capture->available delay (drawn in reset;
        0.0 for a synchronous camera). A privileged DR latent for the
        asymmetric critic (SO101BaseEnv._priv_tail)."""
        return self._delay

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
        pre-episode world is static, so every frame captured before `t` shows
        exactly the reset state. The schedule is extrapolated backward to
        (a) the newest capture whose pipeline delay has elapsed by `t` —
        returned as (capture_t, frame), aging exactly like a mid-episode one
        (delay + sawtooth phase) — and (b) the captures still in the pipeline
        at `t`, queued so they land during the first ticks like on a real
        camera. Each frame is capture_fn-processed once, as always."""
        self._hist.clear()
        self._pending.clear()
        self._hist.append((t, state))
        if self._random_phase:
            self._delay = rng.uniform(self.delay_lo, self.delay_hi)
            self._next_capture_t = t + rng.uniform(0.0, self.frame_s)
        else:
            self._delay = 0.0
            self._next_capture_t = t + self.frame_s
        phase = self._next_capture_t - t
        n_back = math.ceil((phase + self._delay) / self.frame_s - self._EPS)
        capture_t = self._next_capture_t - n_back * self.frame_s
        for k in range(1, n_back):
            in_flight_t = capture_t + k * self.frame_s
            self._pending.append((in_flight_t + self._delay, in_flight_t,
                                  capture_fn(state)))
        return capture_t, capture_fn(state)

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
        """Advance the schedule to sim time `t` and return the frames that
        became available since the last call, oldest first, as
        (capture_t, frame) pairs — empty if no new frame is due yet. The
        caller keeps the newest as its current frame and folds every one into
        any per-tag held state, exactly like the real capture thread updates
        per frame (real/marker_obs.py)."""
        while self._next_capture_t <= t + self._EPS:
            capture_t = self._next_capture_t
            frame = capture_fn(self._state_at(capture_t))
            self._pending.append((capture_t + self._delay, capture_t, frame))
            step = self.frame_s
            if self.jitter_s > 0.0:
                step = max(self.frame_s / 2.0, step + rng.normal(0.0, self.jitter_s))
            self._next_capture_t = capture_t + step
        available = []
        while self._pending and self._pending[0][0] <= t + self._EPS:
            _, capture_t, frame = self._pending.popleft()
            available.append((capture_t, frame))
        return available
