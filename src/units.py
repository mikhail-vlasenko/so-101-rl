"""Single source of truth for the action → joint-delta → servo-raw unit chain.

The full conversion chain, used identically in sim training and real rollouts:

    policy action ∈ [-1, 1]
      × action_scale (conf/config.yaml)        → joint-space delta [rad] per tick
      / SERVO_RAW_UNIT_RAD                     → delta in servo raw units
      deadzone + round (action_to_target)      → executable raw delta
      clamp ±max_raw_delta_per_step()          → real-bus safety bound

All derived quantities (per-tick clamp, joint speed limits) come from the
helpers here — never hand-compute "action_scale in raw units" in a comment.
"""

import numpy as np

# Feetech SMS-STS: 12-bit absolute encoder over 360° → 4096 raw units/rev.
SERVO_RAW_PER_REV = 4096
SERVO_RAW_UNIT_RAD = 2.0 * np.pi / SERVO_RAW_PER_REV

# SyncWritePosEx argument units (Feetech SMS-STS protocol):
# speed argument is in steps of 0.732 RPM, accel argument in steps of
# 8.7 deg/s². Multiply SERVO_SPEED / SERVO_ACCEL (real/twin/constants.py)
# by these to get the firmware profile limits in joint-space SI units.
SERVO_SPEED_UNIT_RAD_S = 0.732 * 2.0 * np.pi / 60.0
SERVO_ACCEL_UNIT_RAD_S2 = 8.7 * np.pi / 180.0

# Commanded position-target deltas below this many raw units get zeroed. Models
# the motor's stiction / min-PWM deadzone: small target errors yield
# below-bring-up current and the joint stalls. Observed on the real arm at
# ~5 commanded raw units under gravity load.
SERVO_DEADZONE_RAW = 4.0

# Headroom above the nominal measured-position-relative action step. The clamp
# instead bounds previous-target-relative changes, so valid actions can still
# bind when tracking lags or the policy reverses direction.
RAW_DELTA_HEADROOM = 2


def rad_to_raw_units(rad: float | np.ndarray) -> float | np.ndarray:
    return rad / SERVO_RAW_UNIT_RAD


def raw_units_to_rad(units: float | np.ndarray) -> float | np.ndarray:
    return units * SERVO_RAW_UNIT_RAD


def max_raw_delta_per_step(action_scale: float) -> int:
    """Per-tick raw-position clamp for real-bus writes: the policy's max
    commanded step (|action|=1) plus RAW_DELTA_HEADROOM."""
    return int(np.ceil(rad_to_raw_units(action_scale))) + RAW_DELTA_HEADROOM


def max_joint_speed_rad_s(action_scale: float, control_hz: float) -> float:
    """Peak joint velocity the policy can command: one full-scale step per tick."""
    return action_scale * control_hz


def clamp_target_delta(previous: np.ndarray, requested: np.ndarray,
                       max_delta: float | int) -> np.ndarray:
    """Bound a target change in the caller's units, relative to the last command.

    Shared by raw real-bus commands, simulated policy commands, and rad-domain
    sysid trajectories. The previous value must be the previously limited target.
    """
    return previous + np.clip(requested - previous, -max_delta, max_delta)


def action_to_target(current: np.ndarray, action: np.ndarray, action_scale: float,
                     joint_low: np.ndarray, joint_high: np.ndarray) -> np.ndarray:
    """Translate a policy action ∈ [-1, 1]^n into the next position target,
    mimicking the real servo's raw-unit quantization + stiction deadzone.
    Without this, the sim policy learns to ease in with tiny actions that the
    real arm cannot execute."""
    raw_delta = rad_to_raw_units(action * action_scale)
    raw_delta = np.where(np.abs(raw_delta) < SERVO_DEADZONE_RAW, 0.0, np.round(raw_delta))
    target = current + raw_units_to_rad(raw_delta)
    return np.clip(target, joint_low, joint_high)
