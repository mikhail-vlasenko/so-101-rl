# TODO

Long-term tasks and ideas. Not a changelog.

## Sim-to-real fidelity (reach env)

Current symptom: with sim/real kp aligned at 64, real arm has trouble holding/lifting itself even though sim trains fine. Also worth noting: experiments so far have been on the **leader** arm, not the follower the policy will ultimately deploy on. Leader has different mechanical load (no payload, no gripper jaws engaged, possibly different calibration), so some of this gap may collapse once we move to the follower. Independently though, sim has several optimistic assumptions:

- **`forcerange` is stall torque, not continuous.** `so101.xml` actuators use `forcerange="-2.94 2.94"` ≈ Feetech STS3215 stall (~30 kg·cm). Continuous holding torque is roughly 1/3 of stall (~1 N·m). Sim lets the actuator pin 2.94 indefinitely; real will current-limit, heat-throttle, or sag. Lower `forcerange` to ~1.0 and retrain — if the policy still solves the task we've removed a hidden "infinite peak torque" cheat from sim.

- **Link masses / inertias.** Menagerie's `so101.xml` derives inertials from STL meshes at an assumed density (likely plastic). Real arm is metal servos + steel screws. Sum body masses in the XML, compare against the weighed arm. If sim is lighter, real torque demand exceeds what the policy learned.

- **Joint friction / damping.** Check `<joint frictionloss damping>` values in `so101.xml`. Real servos have stiction and gearbox friction that's commonly modeled at zero. Add a reasonable fixed value or domain randomize.

- **Re-record sysid data and refit at the current servo settings.** The sim now models the firmware motion profile (`src/servo_profile.py`: accel/speed-register trapezoid + per-tick sub-target interpolation, shared by training envs and `sysid.replay_sim`), actuator kp is per-joint in `so101.xml` (proportional to `SERVO_POSITION_KP`), and `sysid.fit_params` fits per-Kp-group kp scales, a kv scale, and absolute frictionloss. But the joint/actuator numbers in `so101.xml` are still the pre-profile fit on data recorded 2026-05-14 with `SERVO_ACCEL=150` and uniform Kp 32 — i.e. damping/armature are inflated to fake the missing ramp, now double-counted. Remaining: `python -m sysid.record_real --all --execute` (arm on, current settings), `python -m sysid.fit_params`, bake `fit.json` into `so101.xml`, check `sysid.analyze` plots — fast/wide motions (`sweep_wrist_roll`) were the worst offenders.

- **Marker observations: calibrate sim sites against the real glued tags; wire the camera into rollouts.** The policy obs now carries world poses of `marker_finger` / `marker_wrist` sites (`so101.xml`). Their placement is an eyeballed estimate of where the tags were glued (finger site assumes "bottom of fixed jaw", wrist site assumes "right = -y world when the arm faces the workspace") — measure the real tag poses (e.g. via the digital twin + camera at a known joint pose) and correct the site pos/quat. Separately, `real/rollout_lift.py` currently fills the marker obs from FK on the lockstep sim, not from the camera; until the AprilTag pipeline feeds measured poses there, the marker obs adds no real-world information.

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
- **Per-joint `SERVO_POSITION_KP`.** Now per-joint in `real/twin/constants.py` (pan/gripper 8, shoulder_lift 24, rest factory 32). Raise back per joint only if holding is the bottleneck.
- **Power-supply / bus current headroom.** All six joints stalling at once can pop a fuse or trip the bench PSU. Know the supply's trip current and have a hand on the power switch for the first rollouts.
- **Gear-strip risk.** Hobby-grade Feetech servos most commonly fail by stripping the second-to-last plastic gear on hard impact. Slow-and-firm pushing is far less risky than fast contact — start rollouts with the arm well away from the table edge / fixtures.
