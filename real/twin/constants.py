"""Shared servo-side safety / control constants for the real-arm tools.

Imported by `digital_twin.py` (CONTROL-mode writes) and `rollout_real.py`
(policy rollout writes). Single source of truth.
"""

# Per-step raw-position delta clamp (in servo units, 0-4095). With the SO-101's
# 0.087°/unit, 35 units ≈ 3.05° per step. At 30 Hz that caps real-side joint
# speed at ~1.6 rad/s, which sits just above the policy's max commanded step
# (action_scale=0.05 rad ≈ 33 raw units) — leaving safety headroom without
# silently truncating in-distribution commands.
MAX_RAW_DELTA_PER_STEP = 35

# SyncWritePosEx speed argument. Range 0-32767 (BIT15 = direction), units
# 0.732 RPM. 1500 -> ~1100 RPM ceiling; well above what the policy commands,
# leaves headroom so speed is never the binding limit.
SERVO_SPEED = 1500

# SyncWritePosEx acceleration argument. Range 0-254, units 8.7 deg/s^2.
# 150 -> ~1300 deg/s^2 (~23 rad/s^2). Lets the servo finish ramping inside a
# single 50ms control step at 20Hz; 20 was leaving the servo stuck in the
# acceleration phase.
SERVO_ACCEL = 150

# Position-loop P gain (Feetech SMS-STS register addr 21), per joint in
# JOINT_NAMES order (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
# wrist_roll, gripper). Factory default is 32 uniformly. Lower Kp = softer
# push into obstacles (stall torque ≈ Kp · action_scale), at the cost of
# worse static-hold on gravity-loaded joints (shoulder_lift, elbow_flex) —
# revisit per-joint if they sag. Range 0..254 per value.
SERVO_POSITION_KP = (32, 32, 32, 32, 32, 32)
