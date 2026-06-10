"""Shared servo-side safety / control constants for the real-arm tools.

Imported by `digital_twin.py` (CONTROL-mode writes) and the policy rollouts.
Single source of truth for bus-level settings. Anything derived from
`action_scale` (per-tick raw-delta clamp, speed limits) lives in `src.units`
and is computed at runtime from the composed config — not hard-coded here.
"""

# SyncWritePosEx speed argument. Range 0-32767 (BIT15 = direction), units
# 0.732 RPM. 1500 -> ~1100 RPM ceiling; well above what the policy commands,
# leaves headroom so speed is never the binding limit.
SERVO_SPEED = 1500

# Sub-target interpolation rate (Hz). Between consecutive control ticks the
# clamped raw target is resampled into waypoints at this rate and streamed to
# the bus, so the (deliberately slow) SERVO_ACCEL ramp stays in its
# always-ramping regime instead of restarting from rest each tick. 100 Hz over
# a 15 Hz control rate gives ~7 waypoints/tick. Shared by the twin's CONTROL
# mode and the policy rollout.
INTERP_HZ = 100.0

# SyncWritePosEx acceleration argument. Range 0-254, units 8.7 deg/s^2.
# 10 -> ~87 deg/s^2 (~1.5 rad/s^2). Earlier this was held at 5 to keep the
# servo permanently ramping so the 15Hz square-wave velocity excitation
# (accel→cruise→decel→park→repeat) couldn't shake the arm — but that made
# manual control feel laggy, since each coarse target arrived faster than the
# servo could crawl into it. Now that both the twin and the rollout stream
# interpolated sub-targets at INTERP_HZ, the servo always has a fresh point a
# few ms ahead and never reaches the decel cusp, so the accel can be raised for
# responsiveness without reintroducing the shake.
SERVO_ACCEL = 10

# Position-loop P gain (Feetech SMS-STS register addr 21), per joint in
# JOINT_NAMES order (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
# wrist_roll, gripper). Factory default is 32 uniformly. Lower Kp = softer
# push into obstacles (stall torque ≈ Kp · action_scale), at the cost of
# worse static-hold on gravity-loaded joints (shoulder_lift, elbow_flex) —
# revisit per-joint if they sag. Range 0..254 per value.
#
# shoulder_pan runs at 8: at the factory 32 the P-loop is underdamped against
# the arm's large, gravity-unloaded inertia about the vertical axis and rings
# visibly during motion (worst with the arm extended). Verified with
# scripts/shake_probe sweeps up to 500 raw/s: Kp 32 shakes, 16 is near-clean,
# 8 shows no oscillation at all. Cost: ~4x the tracking error per unit of
# acceleration demand — harmless on an unloaded axis driven closed-loop, but
# note the sim actuator gain (so101.xml sts3215 class) was sysid-fit at 32;
# see TODO.md.
#
# gripper also runs at 8: it needs no positioning stiffness, and since grip
# force scales with Kp · position error, a soft loop squeezes the cube (and
# its own gear train) gently instead of grinding at full P-torque.
SERVO_POSITION_KP = (8, 32, 32, 32, 32, 8)

# Position-correction deadzone (Feetech SMS-STS registers 26/27, CW/CCW),
# per joint in JOINT_NAMES order. Units 0.087°/unit, range 0..16, factory
# default 1. Inside ±deadzone the servo's position loop does not correct,
# so the joint can sit anywhere in a backlash gap without the PID hunting
# across it. shoulder_pan ran at 8 for a while to mask the Kp-32 ringing
# (see SERVO_POSITION_KP above) at the cost of ~0.7° of slop and visible
# stick-slip stepping on slow moves; with pan Kp at 8 the root cause is
# gone and everything is back at factory. If rest-hold chatter returns on
# a joint, bump that joint only.
SERVO_POSITION_DEADZONE = (1, 1, 1, 1, 1, 1)
