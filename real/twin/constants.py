"""Shared servo-side safety / control constants for the real-arm tools.

Imported by `digital_twin.py` (CONTROL-mode writes) and `rollout_real.py`
(policy rollout writes). Single source of truth.
"""

# Per-step raw-position delta clamp (in servo units, 0-4095). With the SO-101's
# 0.087 deg/unit, this limits one step to ~2.2° at the joint. Picked so the
# arm can't lurch even if the policy or slider commands a far target.
MAX_RAW_DELTA_PER_STEP = 25

# SyncWritePosEx speed argument. Range 0-32767 (BIT15 = direction), units
# 0.732 RPM. 200 -> ~146 RPM at the servo output.
SERVO_SPEED = 200

# SyncWritePosEx acceleration argument. Range 0-254, units 8.7 deg/s^2.
# 20 -> 174 deg/s^2.
SERVO_ACCEL = 20
