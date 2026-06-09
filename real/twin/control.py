"""Shared real-arm command shaping for position-control writes.

Both the digital twin's CONTROL mode and the policy rollout drive the servos
with the same two-stage shaping:

1. clamp the per-tick raw-position jump (`clamp_raw_delta`), then
2. stream interpolated sub-targets across the control period
   (`stream_sub_targets`).

The streaming matters because `SERVO_ACCEL` is set deliberately slow: a single
coarse target per control tick leaves the servo's trapezoidal profile crawling
toward a point it never reaches before the next tick, which feels laggy under
manual control. Resampling the move into waypoints a few milliseconds apart
keeps the servo perpetually close to its commanded point without raising
`SERVO_ACCEL` (which would reintroduce the low-Hz shake it was lowered to kill).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def clamp_raw_delta(prev_raw: np.ndarray, target_raw: np.ndarray,
                    max_delta: int) -> np.ndarray:
    """Limit the per-tick raw move to ±max_delta so a large target jump can't
    command a full-speed lunge. Callers derive max_delta from the action scale
    via `src.units.max_raw_delta_per_step`."""
    delta = np.clip(target_raw - prev_raw, -max_delta, max_delta)
    return (prev_raw + delta).astype(np.int64)


def stream_sub_targets(
    prev_raw: np.ndarray,
    target_raw: np.ndarray,
    n_interp: int,
    sub_dt: float,
    write: Callable[[np.ndarray], None],
) -> None:
    """Slide prev_raw -> target_raw as `n_interp` evenly-spaced raw waypoints,
    calling `write(raw)` on each and pacing them `sub_dt` seconds apart.

    `write` encapsulates how each waypoint reaches the bus: the threaded twin
    wraps it in the bus lock, the rollout gates it on `--execute`. Pacing uses a
    monotonic clock anchored at call time, so per-waypoint write jitter doesn't
    accumulate across the tick.
    """
    start = prev_raw.astype(np.float64)
    end = target_raw.astype(np.float64)
    t0 = time.monotonic()
    for sub in range(n_interp):
        alpha = (sub + 1) / n_interp
        write((start + alpha * (end - start)).round().astype(np.int64))
        ahead = t0 + (sub + 1) * sub_dt - time.monotonic()
        if ahead > 0:
            time.sleep(ahead)
