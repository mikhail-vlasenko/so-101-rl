"""sysid.replay_sim.clamp_traj must mirror the real bus's clamp_raw_delta:
identical commanded-target sequences on both sides is the premise of the fit.
"""

import numpy as np
import pytest

from real.twin.control import clamp_raw_delta
from src.units import SERVO_RAW_UNIT_RAD, rad_to_raw_units
from sysid.replay_sim import clamp_traj, max_delta_rad_from_config

MAX_DELTA_RAD = max_delta_rad_from_config()


def test_slow_trajectory_unchanged():
    t = np.linspace(0.0, 1.0, 50)
    traj = 0.5 * np.sin(2 * np.pi * 0.5 * t)[:, None] * np.ones((1, 6))  # ≤0.06 rad/tick
    assert clamp_traj(traj, MAX_DELTA_RAD) == pytest.approx(traj)


def test_fast_jump_is_rate_limited_and_converges():
    traj = np.zeros((30, 6))
    traj[1:, 0] = 1.0  # instant 1 rad step on one joint
    out = clamp_traj(traj, MAX_DELTA_RAD)
    deltas = np.diff(out, axis=0)
    assert np.max(np.abs(deltas)) <= MAX_DELTA_RAD + 1e-12
    assert out[-1] == pytest.approx(traj[-1])  # eventually reaches the target
    assert np.all(out[:, 1:] == 0.0)           # other joints untouched


def test_matches_raw_int_clamp_within_quantization():
    rng = np.random.default_rng(0)
    # random walk with steps up to 4x the clamp, exercising both regimes
    traj = np.cumsum(rng.uniform(-4, 4, size=(200, 6)) * MAX_DELTA_RAD, axis=0)

    rad_cmds = clamp_traj(traj, MAX_DELTA_RAD)

    max_raw = int(round(rad_to_raw_units(MAX_DELTA_RAD)))
    raw = np.round(rad_to_raw_units(traj[0])).astype(np.int64)
    for i in range(1, len(traj)):
        target_raw = np.round(rad_to_raw_units(traj[i])).astype(np.int64)
        raw = clamp_raw_delta(raw, target_raw, max_raw)
        # per-tick agreement within one raw unit of rounding slack
        assert rad_cmds[i] == pytest.approx(raw * SERVO_RAW_UNIT_RAD,
                                            abs=1.5 * SERVO_RAW_UNIT_RAD)
