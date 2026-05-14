"""Trajectory library for SO-101 sim-to-real dynamics characterization.

Each entry in TRAJECTORIES is a (T, 6) array of joint-target positions in rad
at the sysid control rate (15 Hz — same as the policy / so101.xml). All
trajectories start at HOME so they can be run back-to-back after a homing
ramp; record_real.py is responsible for ramping the arm from its current
pose to traj[0] before logging begins.

The trajectory set is designed to surface sim-vs-real mismatch:
  - sweep_*: single-joint cosine sweep — isolates per-joint behavior.
  - step_*: step + hold on a gravity-loaded joint — reveals closed-loop
    bandwidth, stiction, and sag.
  - multijoint_reach: coordinated motion across all joints — stresses
    cross-joint coupling and inertia mismatch.
  - gravity_hold: extend arm and hold — steady-state gravity error.
  - chirp_*: per-joint linear chirp — probes frequency-dependent damping.
"""

import numpy as np

from base_env import JOINT_NAMES

SYSID_HZ = 15.0
SYSID_DT = 1.0 / SYSID_HZ

HOME = np.array([0.0, -1.4, 1.2, 0.0, 0.0, 0.5], dtype=np.float64)

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
    return np.arange(_n_samples(duration_s)) * SYSID_DT


def _cosine_ramp(n: int) -> np.ndarray:
    """0 → 1 over n samples, smooth at both ends."""
    return (1 - np.cos(np.linspace(0.0, np.pi, n + 1)[1:])) / 2


def _sweep(joint: str, amp: float, duration_s: float) -> np.ndarray:
    """One full cycle: HOME → -amp → HOME → +amp → HOME."""
    t = _times(duration_s)
    idx = JOINT_NAMES.index(joint)
    delta = -amp * np.sin(2 * np.pi * t / duration_s)
    traj = np.tile(HOME, (len(t), 1))
    traj[:, idx] = HOME[idx] + delta
    return traj


def _step(joint: str, amp: float, hold_s: float) -> np.ndarray:
    """HOME-hold, +amp-hold, HOME-hold, -amp-hold, HOME-hold."""
    idx = JOINT_NAMES.index(joint)
    n_hold = _n_samples(hold_s)
    pieces = [HOME[idx], HOME[idx] + amp, HOME[idx], HOME[idx] - amp, HOME[idx]]
    j_values = np.concatenate([np.full(n_hold, v) for v in pieces])
    traj = np.tile(HOME, (len(j_values), 1))
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


def _gravity_hold() -> np.ndarray:
    extended = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.5])
    ramp_s = 1.5
    hold_s = 4.0
    n_ramp = _n_samples(ramp_s)
    n_hold = _n_samples(hold_s)
    u = _cosine_ramp(n_ramp)
    out_seg = HOME[None, :] + u[:, None] * (extended - HOME)[None, :]
    hold_seg = np.tile(extended, (n_hold, 1))
    back_seg = extended[None, :] + u[:, None] * (HOME - extended)[None, :]
    return np.concatenate([HOME[None, :], out_seg, hold_seg, back_seg], axis=0)


def _chirp(joint: str, amp: float = 0.2, f0: float = 0.3, f1: float = 2.0,
           duration_s: float = 8.0) -> np.ndarray:
    t = _times(duration_s)
    idx = JOINT_NAMES.index(joint)
    phi = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t * t)
    # Hann window so trajectory starts and ends at HOME exactly.
    window = np.sin(np.pi * t / duration_s) ** 2
    delta = amp * window * np.sin(phi)
    traj = np.tile(HOME, (len(t), 1))
    traj[:, idx] = HOME[idx] + delta
    return traj


def _build_trajectories() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for joint in JOINT_NAMES:
        out[f"sweep_{joint}"] = _sweep(joint, _SWEEP_AMP[joint], duration_s=6.0)
    for joint, amp in _STEP_AMP.items():
        out[f"step_{joint}"] = _step(joint, amp, hold_s=1.5)
    out["multijoint_reach"] = _multijoint_reach()
    out["gravity_hold"] = _gravity_hold()
    for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
        out[f"chirp_{joint}"] = _chirp(joint)
    return out


TRAJECTORIES: dict[str, np.ndarray] = _build_trajectories()
