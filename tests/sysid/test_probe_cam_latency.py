"""solve_lag contract: recover a known camera pipeline delay from synthetic
encoder-FK and camera series (sysid/probe_cam_latency.py)."""

import numpy as np

from sysid.probe_cam_latency import MIN_MOTION_STD_M, solve_lag, weighted_lag

TRUE_LAG_S = 0.042
FK_HZ = 50.0
CAM_HZ = 30.0
DURATION_S = 30.0


def _true_pos(t):
    """Ground-truth marker positions (n_tags=2, 3): tag 0 swings on y, tag 1 on
    x with less amplitude; the other axes stay still (no timing signal)."""
    t = np.atleast_1d(t)
    pos = np.zeros((len(t), 2, 3))
    pos[:, 0, 1] = 0.05 * np.sin(2 * np.pi * 0.4 * t)
    pos[:, 0, 0] = 0.20
    pos[:, 1, 0] = 0.02 * np.sin(2 * np.pi * 0.4 * t + 0.7)
    pos[:, 1, 2] = 0.15
    return pos


def _synthetic(rng, cam_noise_m=0.002, static_offset_m=0.004):
    t_fk = np.arange(0.0, DURATION_S, 1.0 / FK_HZ)
    fk = _true_pos(t_fk)
    t_cam = np.arange(0.3, DURATION_S - 0.1, 1.0 / CAM_HZ)
    # The camera frame received at t shows the world as it was at t - lag,
    # plus per-frame measurement noise and a static calibration offset.
    cam = (_true_pos(t_cam - TRUE_LAG_S)
           + rng.normal(0, cam_noise_m, size=(len(t_cam), 2, 3))
           + static_offset_m)
    detected = np.ones((len(t_cam), 2), dtype=bool)
    return t_fk, fk, t_cam, cam, detected


def test_recovers_known_lag():
    rng = np.random.default_rng(0)
    t_fk, fk, t_cam, cam, detected = _synthetic(rng)
    results = solve_lag(t_fk, fk, t_cam, cam, detected)
    assert results, "no lag estimates from a clean synthetic run"
    tau = weighted_lag(results)
    assert abs(tau - TRUE_LAG_S) < 0.002, f"tau={tau * 1e3:.1f} ms vs {TRUE_LAG_S * 1e3} ms"
    # Every individual moving axis lands close too.
    for r in results:
        assert abs(r["lag_s"] - TRUE_LAG_S) < 0.004, r


def test_still_axes_carry_no_estimate():
    """Axes without motion (constant coordinates) must be excluded — they have
    no timing signal and would only dilute the weighted mean."""
    rng = np.random.default_rng(1)
    t_fk, fk, t_cam, cam, detected = _synthetic(rng)
    results = solve_lag(t_fk, fk, t_cam, cam, detected)
    moving = {("marker_finger", "y"), ("marker_wrist", "x")}
    assert {(r["site"], r["axis"]) for r in results} == moving
    for r in results:
        assert r["motion_std_m"] > MIN_MOTION_STD_M


def test_undetected_tag_is_excluded():
    """A tag the camera almost never sees must not produce a lag estimate."""
    rng = np.random.default_rng(2)
    t_fk, fk, t_cam, cam, detected = _synthetic(rng)
    detected[:, 1] = False
    detected[:5, 1] = True  # a handful of detections is below MIN_CAM_SAMPLES
    results = solve_lag(t_fk, fk, t_cam, cam, detected)
    assert {r["site"] for r in results} == {"marker_finger"}


def test_weighting_prefers_strong_motion():
    """The weighted lag must track the strong-motion axis when a weak axis is
    corrupted (e.g. by residual calibration wobble)."""
    rng = np.random.default_rng(3)
    t_fk, fk, t_cam, cam, detected = _synthetic(rng)
    # Corrupt the weak tag with heavy noise: its per-axis estimate degrades,
    # but the 5 cm swing on the finger tag dominates the weighted mean.
    cam[:, 1, :] += rng.normal(0, 0.01, size=(len(t_cam), 3))
    results = solve_lag(t_fk, fk, t_cam, cam, detected)
    tau = weighted_lag(results)
    assert abs(tau - TRUE_LAG_S) < 0.003, f"tau={tau * 1e3:.1f} ms"
