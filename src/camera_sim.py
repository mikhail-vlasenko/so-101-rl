"""Discrete-frame camera simulation for observation latency (sim2real).

The real pipeline derives three observations from discrete camera frames at
different rates: AprilTag arm markers first, the dual-SAM live centroid after
both views have run, and dense BPS after StereoSGBM. A frame samples the world
once at its capture instant, then each modality becomes available after its
own capture-to-result delay. Each policy tick consumes the newest available
result for every modality. With a 30 fps camera against 15 Hz control, each
consumed result is its pipeline delay plus zero-to-one frame interval old; the
staleness follows a sawtooth as the camera and control clocks beat. Joint
encoders have no analog of this: the bus read is ~2 ms, effectively fresh at a
66.7 ms tick.

CameraSim reproduces the timing skeleton while staying agnostic of what a
"frame" contains: the env records opaque world-state snapshots (`record`),
CameraSim captures frames from that history on a fixed-period schedule —
per-episode random phase, per-episode marker/live/BPS delays sampled from their
ranges, and per-frame capture-time jitter whose random walk stands in for real
clock drift. `observe` returns a separate delivery stream for every modality,
each tagged with the common capture time so the consumer can age it. Each frame
is processed exactly once, at capture time, through the env's `capture_fn`
(detector dropout and measurement noise freeze per frame, like real capture).

The env only snapshots the substeps `needs_state` claims. A snapshot is
expensive (occlusion raycasts against every camera, src/base_env.py
`_capture_camera_state`) while the schedule consumes one per frame: at 30 fps
against a 150 Hz substep rate, snapshotting unconditionally built and threw
away four of every five.
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDeliveries:
    """Frames newly available to each observation modality."""

    marker: list[tuple[float, object]]
    live: list[tuple[float, object]]
    bps: list[tuple[float, object]]


class CameraSim:
    """Frame-schedule bookkeeping for one episode; states/frames are opaque."""

    # Float slack (s) so a frame due exactly at an obs time counts as due:
    # sim time and the frame schedule accumulate through different float sums.
    _EPS = 1e-6

    def __init__(self, frame_s: float,
                 marker_delay_range_s: tuple[float, float],
                 live_delay_range_s: tuple[float, float],
                 bps_delay_range_s: tuple[float, float],
                 jitter_s: float, control_dt: float, substep_s: float):
        frame_s = float(frame_s)
        marker_delay = self._validate_delay_range("marker", marker_delay_range_s)
        live_delay = self._validate_delay_range("live", live_delay_range_s)
        bps_delay = self._validate_delay_range("bps", bps_delay_range_s)
        jitter_s = float(jitter_s)
        substep_s = float(substep_s)
        assert frame_s > 0.0, f"frame_s must be positive, got {frame_s}"
        assert marker_delay[1] <= live_delay[0], \
            f"marker delay {marker_delay} must precede live delay {live_delay}"
        assert live_delay[1] <= bps_delay[0], \
            f"live delay {live_delay} must precede BPS delay {bps_delay}"
        assert 0.0 <= jitter_s < frame_s / 4, \
            f"jitter_s={jitter_s} must be well under frame_s={frame_s}"
        # A substep coarser than half a frame interval would let two captures
        # claim the same snapshot, collapsing the schedule onto the substep grid.
        assert 0.0 < substep_s <= frame_s / 2, \
            f"substep_s={substep_s} must be positive and at most frame_s/2={frame_s / 2}"
        self.frame_s = frame_s
        self.marker_delay_range_s = marker_delay
        self.live_delay_range_s = live_delay
        self.bps_delay_range_s = bps_delay
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
        # Each queue contains (availability time, capture time, frame), ordered
        # by capture/availability time. The frame object is shared across queues.
        self._marker_pending: deque = deque()
        self._live_pending: deque = deque()
        self._bps_pending: deque = deque()
        self._next_capture_t = 0.0
        self._marker_delay = 0.0
        self._live_delay = 0.0
        self._bps_delay = 0.0

    @staticmethod
    def _validate_delay_range(name: str,
                              values: tuple[float, float]) -> tuple[float, float]:
        lo, hi = float(values[0]), float(values[1])
        assert 0.0 <= lo <= hi, \
            f"{name} delay range must satisfy 0 <= lo <= hi, got ({lo}, {hi})"
        return lo, hi

    @property
    def pipeline_delay_s(self) -> float:
        """Sampled marker delay retained as the critic's camera-delay latent."""
        return self._marker_delay

    @property
    def live_delay_s(self) -> float:
        return self._live_delay

    @property
    def bps_delay_s(self) -> float:
        return self._bps_delay

    @classmethod
    def synchronous(cls, control_dt: float, substep_s: float) -> "CameraSim":
        """Zero-latency camera (dr=none / plain env construction): one frame is
        captured exactly at every control tick and available immediately, so
        marker/cube obs equal the current state and reset consumes no RNG."""
        cam = cls(
            frame_s=control_dt,
            marker_delay_range_s=(0.0, 0.0),
            live_delay_range_s=(0.0, 0.0),
            bps_delay_range_s=(0.0, 0.0),
            jitter_s=0.0, control_dt=control_dt, substep_s=substep_s)
        cam._random_phase = False
        return cam

    def reset(self, rng, t: float, state, capture_fn):
        """Start an episode at sim time `t` with the world in `state`. The
        pre-episode world is static, so every frame captured before `t` shows
        exactly the reset state. The schedule is extrapolated backward to
        (a) the newest capture available to each modality at `t`, aging exactly
        like a mid-episode result, and (b) later captures still in each pipeline,
        queued so they land during the first ticks. Each physical frame is
        capture_fn-processed once and shared by the three delivery streams."""
        self._hist.clear()
        self._due.clear()
        self._marker_pending.clear()
        self._live_pending.clear()
        self._bps_pending.clear()
        self._hist.append((t, state))
        if self._random_phase:
            self._marker_delay = rng.uniform(*self.marker_delay_range_s)
            self._live_delay = rng.uniform(*self.live_delay_range_s)
            self._bps_delay = rng.uniform(*self.bps_delay_range_s)
            self._next_capture_t = t + rng.uniform(0.0, self.frame_s)
        else:
            self._marker_delay = 0.0
            self._live_delay = 0.0
            self._bps_delay = 0.0
            self._next_capture_t = t + self.frame_s
        phase = self._next_capture_t - t
        max_delay = max(self._marker_delay, self._live_delay, self._bps_delay)
        n_back = math.ceil((phase + max_delay) / self.frame_s - self._EPS)
        oldest_t = self._next_capture_t - n_back * self.frame_s
        captured = [
            (oldest_t + k * self.frame_s, capture_fn(state))
            for k in range(n_back)
        ]

        def seed_stream(delay: float, pending: deque) -> list[tuple[float, object]]:
            available = [(capture_t, frame) for capture_t, frame in captured
                         if capture_t + delay <= t + self._EPS]
            assert available, f"no reset frame available for delay {delay}"
            newest_t = available[-1][0]
            for capture_t, frame in captured:
                if capture_t > newest_t:
                    pending.append((capture_t + delay, capture_t, frame))
            return [available[-1]]

        return CameraDeliveries(
            marker=seed_stream(self._marker_delay, self._marker_pending),
            live=seed_stream(self._live_delay, self._live_pending),
            bps=seed_stream(self._bps_delay, self._bps_pending),
        )

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
        """Build every newly captured frame once and return per-modality results
        whose availability time has passed, oldest first."""
        while self._due:
            capture_t = self._due.popleft()
            frame = capture_fn(self._state_at(capture_t))
            self._marker_pending.append(
                (capture_t + self._marker_delay, capture_t, frame))
            self._live_pending.append((capture_t + self._live_delay, capture_t, frame))
            self._bps_pending.append((capture_t + self._bps_delay, capture_t, frame))

        def drain(pending: deque) -> list[tuple[float, object]]:
            available = []
            while pending and pending[0][0] <= t + self._EPS:
                _, capture_t, frame = pending.popleft()
                available.append((capture_t, frame))
            return available

        return CameraDeliveries(
            marker=drain(self._marker_pending),
            live=drain(self._live_pending),
            bps=drain(self._bps_pending),
        )
