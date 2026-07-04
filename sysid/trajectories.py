"""Trajectory library for SO-101 sim-to-real dynamics characterization.

Each entry in TRAJECTORIES is a (T, 6) array of joint-target positions in rad
at the sysid control rate (15 Hz — same as the policy / so101.xml). Every
trajectory starts and ends at its base pose (HOME or STRETCH) so the set can
run back-to-back; record_real.py ramps the arm from its current pose to
traj[0] before logging begins.

The trajectory set is designed to surface sim-vs-real mismatch:
  - sweep_*: single-joint cosine sweep — isolates per-joint behavior.
  - step_*: step + hold on a gravity-loaded joint — reveals closed-loop
    bandwidth, stiction, and sag.
  - multijoint_reach: coordinated motion across all joints — stresses
    cross-joint coupling and inertia mismatch.
  - gravity_hold: extend arm and hold — steady-state gravity error.
  - chirp_*: per-joint linear chirps, a small-amplitude wide-band segment
    plus a large-amplitude low-band one — probes frequency-dependent damping
    in both the small-signal and the ramp/clamp-saturated regime.
  - stretch_*: the same probes around STRETCH (arm horizontal, fully reached
    out) — maximum gravity torque on shoulder_lift/elbow and maximum pan
    inertia, the regime where holding/lifting failures show up on the real
    arm but the HOME-based data never looks.
  - staircase_*: slow quasi-static steps through the gravity-loaded range —
    stiction/hysteresis signal for the frictionloss fit.

Safety: tests/test_sysid_trajectories.py forward-kinematics every frame and
asserts no arm geom comes within FLOOR_MARGIN of the floor; extend the test
margins rather than eyeballing when adding poses.
"""

import numpy as np

from src.base_env import JOINT_NAMES

SYSID_HZ = 15.0
SYSID_DT = 1.0 / SYSID_HZ

# Hold at traj[0] before logging starts: long enough for the servo's slow
# accel ramp to close the homing lag and for stiction / gravity sag to settle
# — the stretch_* trajectories start gravity-loaded, so an unsettled start
# would leak a startup transient into the first logged ticks. Shared by
# record_real (real bus) and replay_sim (sim) so both sides log from the same
# settled state; a sim that skips it starts exactly AT traj[0] and the fit
# would chase the real arm's sag as if it were dynamics.
SETTLE_S = 1.0

HOME = np.array([0.0, -1.4, 1.2, 0.0, 0.0, 0.5], dtype=np.float64)
# Arm straight out forward, horizontal: EE at (0.39, 0, 0.25) m — peak gravity
# moment on shoulder_lift and elbow_flex, peak inertia about the pan axis.
STRETCH = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)

_SWEEP_AMP = {
    "shoulder_pan":  1.0,
    "shoulder_lift": 0.3,
    "elbow_flex":    0.4,
    "wrist_flex":    0.9,
    "wrist_roll":    1.5,
    "gripper":       0.5,
}

_STEP_AMP = {
    "shoulder_lift": 0.3,
    "elbow_flex":    0.4,
}


def _n_samples(duration_s: float) -> int:
    return int(round(duration_s * SYSID_HZ))


def _times(duration_s: float) -> np.ndarray:
    """Sample times 0..duration_s inclusive, so periodic/windowed trajectories
    built on these return exactly to their base pose on the last sample."""
    return np.linspace(0.0, duration_s, _n_samples(duration_s) + 1)


def _cosine_ramp(n: int) -> np.ndarray:
    """0 → 1 over n samples, smooth at both ends."""
    return (1 - np.cos(np.linspace(0.0, np.pi, n + 1)[1:])) / 2


def _sweep(joint: str, amp: float, duration_s: float,
           base: np.ndarray = HOME) -> np.ndarray:
    """One full cycle: base → -amp → base → +amp → base."""
    t = _times(duration_s)
    idx = JOINT_NAMES.index(joint)
    delta = -amp * np.sin(2 * np.pi * t / duration_s)
    traj = np.tile(base, (len(t), 1))
    traj[:, idx] = base[idx] + delta
    return traj


def _step(joint: str, amp: float, hold_s: float,
          base: np.ndarray = HOME) -> np.ndarray:
    """base-hold, +amp-hold, base-hold, -amp-hold, base-hold."""
    idx = JOINT_NAMES.index(joint)
    n_hold = _n_samples(hold_s)
    pieces = [base[idx], base[idx] + amp, base[idx], base[idx] - amp, base[idx]]
    j_values = np.concatenate([np.full(n_hold, v) for v in pieces])
    traj = np.tile(base, (len(j_values), 1))
    traj[:, idx] = j_values
    return traj


def _multijoint_reach() -> np.ndarray:
    waypoints = [
        HOME,
        np.array([ 0.7, -1.2, 1.0,  0.2,  0.5, 0.5]),
        np.array([-0.7, -1.2, 1.0, -0.2, -0.5, 0.5]),
        np.array([ 0.0, -1.0, 0.8,  0.0,  0.0, 1.0]),
        HOME,
    ]
    seg_s = 2.0
    n_seg = _n_samples(seg_s)
    pieces = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        u = _cosine_ramp(n_seg)
        pieces.append(a[None, :] + u[:, None] * (b - a)[None, :])
    return np.concatenate([HOME[None, :], *pieces], axis=0)


def _ramp_hold_return(pose: np.ndarray, ramp_s: float, hold_s: float) -> np.ndarray:
    """HOME → pose (cosine ramp), hold, ramp back to HOME."""
    n_ramp = _n_samples(ramp_s)
    n_hold = _n_samples(hold_s)
    u = _cosine_ramp(n_ramp)
    out_seg = HOME[None, :] + u[:, None] * (pose - HOME)[None, :]
    hold_seg = np.tile(pose, (n_hold, 1))
    back_seg = pose[None, :] + u[:, None] * (HOME - pose)[None, :]
    return np.concatenate([HOME[None, :], out_seg, hold_seg, back_seg], axis=0)


def _gravity_hold() -> np.ndarray:
    extended = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.5])
    return _ramp_hold_return(extended, ramp_s=1.5, hold_s=4.0)


def _chirp(joint: str, segments: list[tuple[float, float, float, float]],
           base: np.ndarray = HOME) -> np.ndarray:
    """Concatenated linear-chirp segments, each (amp, f0, f1, duration_s).

    Every segment is Hann-windowed, so it starts and ends at the base pose
    exactly and segments join seamlessly."""
    idx = JOINT_NAMES.index(joint)
    parts = []
    for amp, f0, f1, duration_s in segments:
        t = _times(duration_s)
        phi = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t * t)
        window = np.sin(np.pi * t / duration_s) ** 2
        delta = amp * window * np.sin(phi)
        traj = np.tile(base, (len(t), 1))
        traj[:, idx] = base[idx] + delta
        parts.append(traj)
    return np.concatenate(parts, axis=0)


def _staircase(joint: str, levels: list[float], hold_s: float,
               base: np.ndarray) -> np.ndarray:
    """Quasi-static staircase: hold base+level for hold_s at each level.

    Levels are offsets from the base pose; the sequence must begin and end at
    0 so the trajectory starts/ends at the base pose. Long holds let stiction
    settle, so the level-to-level position error isolates dry friction and
    gravity sag from dynamic effects.
    """
    assert levels[0] == 0.0 and levels[-1] == 0.0, "staircase must start/end at base"
    idx = JOINT_NAMES.index(joint)
    n_hold = _n_samples(hold_s)
    j_values = np.concatenate([np.full(n_hold, base[idx] + lv) for lv in levels])
    traj = np.tile(base, (len(j_values), 1))
    traj[:, idx] = j_values
    return traj


# Chirp segments (amp, f0, f1, duration_s): a small-amplitude wide-band
# segment plus a large-amplitude low-band one — the large waves drive the
# servo deep into its accel-ramp and per-tick-clamp regime, where the
# small-signal data carries no information. Large amplitudes are limited by
# the joint range around the base pose (shoulder_lift at HOME has only
# 0.345 rad of headroom) and, around STRETCH, by the floor.
_CHIRP_FINE = (0.2, 0.3, 2.0, 8.0)
_CHIRP_LARGE_F0_F1_DUR = (0.2, 1.0, 6.0)
_CHIRP_LARGE_AMP = {
    "shoulder_pan":  0.6,
    "shoulder_lift": 0.3,
    "elbow_flex":    0.45,
    "wrist_flex":    0.6,
}


def _chirp_segments(joint: str, large_amp: float | None = None) -> list[tuple]:
    amp = _CHIRP_LARGE_AMP[joint] if large_amp is None else large_amp
    return [_CHIRP_FINE, (amp, *_CHIRP_LARGE_F0_F1_DUR)]


# Sweep amplitudes around STRETCH are floor-limited: positive shoulder_lift /
# elbow_flex / wrist_flex all pitch the reached-out gripper toward the table
# (see tests/test_sysid_trajectories.py clearance check).
_STRETCH_SWEEP_AMP = {
    "shoulder_pan":  0.9,
    "shoulder_lift": 0.4,
    "elbow_flex":    0.4,
    "wrist_flex":    0.8,
}

_STAIRCASE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1,
                     0.0, -0.1, -0.2, -0.3, -0.2, -0.1, 0.0]


def _build_trajectories() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for joint in JOINT_NAMES:
        out[f"sweep_{joint}"] = _sweep(joint, _SWEEP_AMP[joint], duration_s=6.0)
    for joint, amp in _STEP_AMP.items():
        out[f"step_{joint}"] = _step(joint, amp, hold_s=1.5)
    out["multijoint_reach"] = _multijoint_reach()
    out["gravity_hold"] = _gravity_hold()
    for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
        out[f"chirp_{joint}"] = _chirp(joint, _chirp_segments(joint))

    # Gravity-loaded probes around the stretched-forward pose.
    for joint, amp in _STRETCH_SWEEP_AMP.items():
        out[f"stretch_sweep_{joint}"] = _sweep(joint, amp, duration_s=6.0, base=STRETCH)
    out["stretch_step_shoulder_lift"] = _step("shoulder_lift", 0.3, hold_s=1.5, base=STRETCH)
    out["stretch_step_elbow_flex"] = _step("elbow_flex", 0.4, hold_s=1.5, base=STRETCH)
    # Large amp 0.4 around STRETCH: floor-limited like the sweep, and the
    # joint-range headroom that binds at HOME doesn't apply at lift=0.
    out["stretch_chirp_shoulder_lift"] = _chirp(
        "shoulder_lift", _chirp_segments("shoulder_lift", large_amp=0.4), base=STRETCH)
    out["stretch_hold"] = _ramp_hold_return(STRETCH, ramp_s=2.5, hold_s=4.0)
    out["staircase_shoulder_lift"] = _staircase(
        "shoulder_lift", _STAIRCASE_LEVELS, hold_s=1.2, base=STRETCH)
    return out


TRAJECTORIES: dict[str, np.ndarray] = _build_trajectories()
