"""Contract tests for the firmware motion-profile model (src/servo_profile.py).

The profile must behave like the Feetech trapezoidal planner: velocity ramps
at the accel-register rate, is capped at the speed-register rate, decelerates
to arrive at a static goal, and degenerates to plain sub-target interpolation
when the limits are far from binding.
"""

import numpy as np
import pytest

from src.servo_profile import ServoProfile
from src.units import SERVO_ACCEL_UNIT_RAD_S2, SERVO_SPEED_UNIT_RAD_S

DT = 1.0 / 150.0  # so101.xml timestep
N_SUBSTEPS = 10   # 15 Hz control tick


def _profile(n=1, speed=1500, accel=10):
    p = ServoProfile(n, speed=speed, accel=accel)
    p.reset(np.zeros(n))
    return p


def test_accel_limit_respected():
    p = _profile()
    setpoints = [np.zeros(1)]
    rng = np.random.default_rng(0)
    goal = np.zeros(1)
    for _ in range(40):  # random goal jumps, including reversals
        goal = rng.uniform(-1.5, 1.5, size=1)
        for _ in range(N_SUBSTEPS):
            setpoints.append(p.step(goal, DT))
    s = np.concatenate(setpoints)
    accel = np.diff(s, 2) / DT**2
    a_max = 10 * SERVO_ACCEL_UNIT_RAD_S2
    assert np.max(np.abs(accel)) <= a_max * 1.001


def test_speed_limit_respected():
    # Tight speed cap (50 units ≈ 3.8 rad/s) with violent accel: a distant
    # goal must be approached at exactly v_max once the ramp completes.
    p = _profile(speed=50, accel=200)
    v_max = 50 * SERVO_SPEED_UNIT_RAD_S
    s = [p.step(np.array([100.0]), DT) for _ in range(300)]
    vel = np.diff(np.concatenate(s)) / DT
    assert np.max(vel) <= v_max * 1.001
    assert np.max(vel) >= v_max * 0.95  # cap actually reached


def test_converges_to_static_goal_at_rest():
    p = _profile()
    goal = np.array([0.3])
    for _ in range(20 * N_SUBSTEPS):  # 20 control ticks = 1.33 s
        out = p.step(goal, DT)
    assert out == pytest.approx(goal, abs=1e-3)
    assert p.vel == pytest.approx(0.0, abs=0.02)


def test_steady_ramp_lag_matches_trapezoid():
    # Chasing a constant-velocity goal, the decel-to-arrive law trails by the
    # stopping distance v²/(2a) — the lag the old sysid fit faked with damping.
    p = _profile()
    a_max = 10 * SERVO_ACCEL_UNIT_RAD_S2
    v = 0.5  # rad/s, sustainable: needs only 0.083 rad of lag
    lags = []
    for i in range(1, 1500):
        out = p.step(np.array([v * i * DT]), DT)
        lags.append(v * i * DT - out[0])
    expected = v**2 / (2 * a_max)
    assert np.mean(lags[-300:]) == pytest.approx(expected, rel=0.15)


def test_unconstrained_profile_is_passthrough():
    # With huge speed/accel limits the setpoint just tracks the interpolated
    # goal, so tick() reduces to stream_sub_targets-style interpolation.
    p = ServoProfile(2, speed=30000, accel=254)
    start = np.array([0.1, -0.2])
    p.reset(start)
    target = np.array([0.15, -0.25])
    out = p.tick(start, target, N_SUBSTEPS, DT)
    alphas = (np.arange(N_SUBSTEPS) + 1)[:, None] / N_SUBSTEPS
    expected = start + alphas * (target - start)
    assert out == pytest.approx(expected, abs=5e-3)


def test_tick_is_continuous_across_ticks():
    p = _profile(n=2)
    prev = np.zeros(2)
    last = None
    for target in [np.array([0.07, -0.07]), np.array([0.14, 0.0]), np.array([0.1, 0.1])]:
        out = p.tick(prev, target, N_SUBSTEPS, DT)
        assert out.shape == (N_SUBSTEPS, 2)
        if last is not None:
            # adjacent setpoints across the tick boundary differ by at most
            # one substep of travel at the (already accel-limited) velocity
            assert np.abs(out[0] - last) == pytest.approx(0.0, abs=2 * np.max(np.abs(p.vel)) * DT + 1e-9)
        last = out[-1]
        prev = target


def test_reset_zeroes_velocity():
    p = _profile()
    for _ in range(50):
        p.step(np.array([1.0]), DT)
    assert abs(p.vel[0]) > 0.1
    p.reset(np.array([0.5]))
    assert p.pos[0] == 0.5
    assert p.vel[0] == 0.0
