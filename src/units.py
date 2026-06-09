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

# Commanded position-target deltas below this many raw units get zeroed. Models
# the motor's stiction / min-PWM deadzone: small target errors yield
# below-bring-up current and the joint stalls. Observed on the real arm at
# ~5 commanded raw units under gravity load.
SERVO_DEADZONE_RAW = 4.0

# Headroom (raw units) the per-tick safety clamp allows above the policy's max
# commanded step, so the clamp catches mapping/calibration bugs without
# truncating in-distribution actions.
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
