"""Discrete-frame camera simulation for observation latency (sim2real).

The real marker pipeline delivers poses from discrete camera frames: a frame
samples the world at its capture instant, becomes available to the control
loop `delay` seconds later (mid-exposure -> sensor readout -> USB transfer ->
MJPG decode -> AprilTag detection; measure it with sysid/probe_cam_latency.py),
and each policy tick then consumes the newest available frame
(real/rollout/marker_obs.py). With a 30 fps camera against 15 Hz control the consumed
pose is `delay` plus zero-to-one frame interval old, and that staleness
follows a sawtooth that drifts as the camera and control clocks beat against
each other. Joint encoders have no analog of any of this: the bus read is ~2 ms,
effectively fresh at a 66.7 ms tick, so only camera-derived observations
(markers, cube) go through this model.

CameraSim reproduces the timing skeleton while staying agnostic of what a
"frame" contains: the env records opaque world-state snapshots (`record`),
CameraSim captures frames from that history on a fixed-period schedule —
per-episode random phase, per-episode pipeline delay sampled from
`delay_range_s`, per-frame capture-time jitter whose random walk stands in for
real clock drift — and `observe` returns every frame whose availability time
has passed since the last call, each tagged with its capture time so the
consumer can age it. Each frame is processed exactly once, at capture time,
through the env's `capture_fn` (detector dropout and measurement noise freeze
per frame, like a real detection).

The env only snapshots the substeps `needs_state` claims. A snapshot is
expensive (occlusion raycasts against every camera, src/base_env.py
`_capture_camera_state`) while the schedule consumes one per frame: at 30 fps
against a 150 Hz substep rate, snapshotting unconditionally built and threw
away four of every five.
"""

import math
from collections import deque


class CameraSim:
    """Frame-schedule bookkeeping for one episode; states/frames are opaque."""

    # Float slack (s) so a frame due exactly at an obs time counts as due:
    # sim time and the frame schedule accumulate through different float sums.
    _EPS = 1e-6

    def __init__(self, frame_s: float, delay_range_s: tuple[float, float],
                 jitter_s: float, control_dt: float, substep_s: float):
        frame_s = float(frame_s)
        delay_lo, delay_hi = float(delay_range_s[0]), float(delay_range_s[1])
        jitter_s = float(jitter_s)
        substep_s = float(substep_s)
        assert frame_s > 0.0, f"frame_s must be positive, got {frame_s}"
        assert 0.0 <= delay_lo <= delay_hi, \
            f"delay range must satisfy 0 <= lo <= hi, got ({delay_lo}, {delay_hi})"
        assert 0.0 <= jitter_s < frame_s / 4, \
            f"jitter_s={jitter_s} must be well under frame_s={frame_s}"
        # A substep coarser than half a frame interval would let two captures
        # claim the same snapshot, collapsing the schedule onto the substep grid.
        assert 0.0 < substep_s <= frame_s / 2, \
            f"substep_s={substep_s} must be positive and at most frame_s/2={frame_s / 2}"
        self.frame_s = frame_s
        self.delay_lo = delay_lo
        self.delay_hi = delay_hi
        self.jitter_s = jitter_s
        # A capture instant is served by the snapshot nearest it in time, so a
        # substep claims every instant within half a substep of itself.
        self._half_substep = substep_s / 2.0
        # History must cover every capture time still unprocessed at the next
        # observe(): captures due then lie within one control period of it, so
        # one period plus a frame interval of slack bounds the lookback.
        self._keep_s = control_dt + frame_s
        self._random_phase = True
        self._hist: deque = deque()      # (t, state), t ascending
        self._due: deque = deque()       # capture times snapshotted, frame not yet built
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
    def synchronous(cls, control_dt: float, substep_s: float) -> "CameraSim":
        """Zero-latency camera (dr=none / plain env construction): one frame is
        captured exactly at every control tick and available immediately, so
        marker/cube obs equal the current state and reset consumes no RNG."""
        cam = cls(frame_s=control_dt, delay_range_s=(0.0, 0.0), jitter_s=0.0,
                  control_dt=control_dt, substep_s=substep_s)
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
        self._due.clear()
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

    def needs_state(self, t: float, rng) -> bool:
        """Whether the substep ending at sim time `t` is the nearest sample to
        at least one upcoming capture instant — i.e. whether the caller must
        snapshot the world now and hand it to `record`.

        Claims every instant it answers for, queueing it for `observe` to turn
        into a frame and advancing the schedule past it (this is where a
        capture's jitter is drawn). Instants are claimed strictly in order and
        never revisited, so the substep that claims one is the closest sample
        it will ever have: an earlier substep would have claimed it already,
        and a later one is further away.
        """
        claimed = False
        while self._next_capture_t <= t + self._half_substep + self._EPS:
            self._due.append(self._next_capture_t)
            step = self.frame_s
            if self.jitter_s > 0.0:
                step = max(self.frame_s / 2.0, step + rng.normal(0.0, self.jitter_s))
            self._next_capture_t += step
            claimed = True
        return claimed

    def record(self, t: float, state) -> None:
        """Log the world state at sim time `t` (call for every substep
        `needs_state` claimed, and no others)."""
        self._hist.append((t, state))
        keep_after = t - self._keep_s
        while self._hist[0][0] < keep_after:
            self._hist.popleft()

    def _state_at(self, t: float):
        """Recorded state nearest to `t` — history spacing is one substep, so
        the lookup error is at most half a substep."""
        return min(self._hist, key=lambda entry: abs(entry[0] - t))[1]

    def observe(self, t: float, capture_fn):
        """Build a frame for every capture instant claimed since the last call
        and return those whose availability time has passed, oldest first, as
        (capture_t, frame) pairs — empty if no new frame is due yet. The
        caller keeps the newest as its current frame and folds every one into
        any per-tag held state, exactly like the real capture thread updates
        per frame (real/rollout/marker_obs.py)."""
        while self._due:
            capture_t = self._due.popleft()
            frame = capture_fn(self._state_at(capture_t))
            self._pending.append((capture_t + self._delay, capture_t, frame))
        available = []
        while self._pending and self._pending[0][0] <= t + self._EPS:
            _, capture_t, frame = self._pending.popleft()
            available.append((capture_t, frame))
        return available
