# TODO

Long-term tasks and ideas. Not a changelog.

## Sim-to-real fidelity (reach env)

Current symptom: with sim/real kp aligned at 64, real arm has trouble holding/lifting itself even though sim trains fine. Also worth noting: experiments so far have been on the **leader** arm, not the follower the policy will ultimately deploy on. Leader has different mechanical load (no payload, no gripper jaws engaged, possibly different calibration), so some of this gap may collapse once we move to the follower. Independently though, sim has several optimistic assumptions:

- **`forcerange` is stall torque, not continuous.** `so101.xml` actuators use `forcerange="-2.94 2.94"` ≈ Feetech STS3215 stall (~30 kg·cm). Continuous holding torque is roughly 1/3 of stall (~1 N·m). Sim lets the actuator pin 2.94 indefinitely; real will current-limit, heat-throttle, or sag. Lower `forcerange` to ~1.0 and retrain — if the policy still solves the task we've removed a hidden "infinite peak torque" cheat from sim.

- **Link masses / inertias.** Menagerie's `so101.xml` derives inertials from STL meshes at an assumed density (likely plastic). Real arm is metal servos + steel screws. Sum body masses in the XML, compare against the weighed arm. If sim is lighter, real torque demand exceeds what the policy learned.

- **Joint friction / damping.** Check `<joint frictionloss damping>` values in `so101.xml`. Real servos have stiction and gearbox friction that's commonly modeled at zero. Add a reasonable fixed value or domain randomize.

- **Internal servo dynamics — confirmed gap.** Real Feetech has internal PID + current loop + comms latency; sim's `<position>` actuator is instantaneous torque ∝ err. The sysid fit (`sysid_logs/fit.json`) drove damping/armature/frictionloss to the upper bound of its search range, which is the optimizer using joint-side losses as a proxy for the servo's internal speed/accel ramp. Best concrete fix: rate-limit `data.ctrl` in `replay_sim.py` (and the policy env) using `SERVO_SPEED`/`SERVO_ACCEL` from `real/twin/constants.py`, then refit. Until that's in, fast/wide-amplitude motions (e.g. `sweep_wrist_roll`) are visibly over-damped in sim.

## Train a policy in PWM / torque-ish mode

STS3215 servos expose a PWM / "current" mode via register writes. Bypassing the internal position PID lets the policy command motor effort directly — no trapezoidal-profile reset every tick (see `SERVO_ACCEL` comment), no 15 Hz lower bound, and a control interface much closer to what most MuJoCo-trained policies use.

Setup work:
- Swap `<position>` actuators in `so101.xml` for `<motor>`, re-run sysid against PWM step responses (current fit is position-regime only).
- Measure per-joint PWM deadband and add to the sim (analog of `SERVO_DEADZONE_RAW`, in the PWM domain).
- Bump control rate to ≥50 Hz both sides; confirm the Feetech bus sustains it (currently ~33 Hz ceiling without pipelining).
- Add joint-limit clamping in the rollout loop — PWM has no inherent "go to X" semantic.
- Calibrate PWM→torque per joint by hanging known masses and recording the motion-onset PWM.

## Verify on real servos before first policy rollout

Nothing in the rollout layer notices "I've been pushing the same wall for 5 seconds"; the Feetech firmware's overload trip is what saves the hardware. Verify before running:

- **`Maximum Temperature Limit` register (addr 14).** Confirm it's at the factory default 70 °C (or lower) on every joint. Triggers torque-off when the servo overheats.
- **`Overload Protection` register / flag.** Confirm enabled — auto-shuts torque if output stays at max for the configured time. Without this, sustained stall cooks the windings or the H-bridge MOSFETs.
- **`Maximum Torque` register.** Check it's not at the absolute max — capping it firmware-side limits stall current and gives a margin before thermal trip.
- **Per-joint `SERVO_POSITION_KP`.** Currently 64 across the board (`real/twin/constants.py`). For first rollouts consider dropping to the factory default 32 to reduce push-into-obstacle force; raise back per joint if holding is the bottleneck.
- **Power-supply / bus current headroom.** All six joints stalling at once can pop a fuse or trip the bench PSU. Know the supply's trip current and have a hand on the power switch for the first rollouts.
- **Gear-strip risk.** Hobby-grade Feetech servos most commonly fail by stripping the second-to-last plastic gear on hard impact. Slow-and-firm pushing is far less risky than fast contact — start rollouts with the arm well away from the table edge / fixtures.
