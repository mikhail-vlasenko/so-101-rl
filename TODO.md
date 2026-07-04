# TODO

Long-term tasks and ideas. Not a changelog.

## Sim-to-real fidelity (reach env)

Current symptom: with sim/real kp aligned at 64, real arm has trouble holding/lifting itself even though sim trains fine. Also worth noting: experiments so far have been on the **leader** arm, not the follower the policy will ultimately deploy on. Leader has different mechanical load (no payload, no gripper jaws engaged, possibly different calibration), so some of this gap may collapse once we move to the follower. Independently though, sim has several optimistic assumptions:

- **`forcerange` is stall torque, not continuous.** `so101.xml` actuators use `forcerange="-2.94 2.94"` ≈ Feetech STS3215 stall (~30 kg·cm). Continuous holding torque is roughly 1/3 of stall (~1 N·m). Sim lets the actuator pin 2.94 indefinitely; real will current-limit, heat-throttle, or sag. Lower `forcerange` to ~1.0 and retrain — if the policy still solves the task we've removed a hidden "infinite peak torque" cheat from sim.

- **Link masses / inertias.** Menagerie's `so101.xml` derives inertials from STL meshes at an assumed density (likely plastic). Real arm is metal servos + steel screws. Sum body masses in the XML, compare against the weighed arm. If sim is lighter, real torque demand exceeds what the policy learned.

- **Compliance during contact.** The linear gravity-compliance term `q_true = q_enc − b − c·τ_grav` (lift/elbow/wrist_flex) is now solved jointly with the bias and mounts and applied at deployment (`real/compliance.py`, `real/calibrate_qpos.py` `COMP_JOINTS`, `ArmLoop`); adding it took the committed calibration residual from 4.0 → 2.4 mm RMS (raw-encoder baseline 8.2; cross-validated held-out 3.6 → 2.0 mm mean, 8.2 → 4.7 mm max). Remaining follow-up: deployment reads τ from the arm's gravity `qfrc_bias` only, which is wrong once the gripper loads up during contact — modeling the compliance in sim (springy `*_play` joints) is the principled alternative if contact-regime marker accuracy ever matters. (The `backlash` class is now the measured ±0.2° ceiling from `sysid/probe_backlash.py`.) Separate probe fact: arriving at the same target from opposite directions leaves the *motor* up to 2.1° apart at pan (stiction vs kp=8; enc Δ tracks the per-joint kp ordering) — the encoder sees it so obs stay truthful, but open-loop moves land 1–2° off depending on approach direction.

- **Real control tick overruns: budget the loop to hold 15 Hz.** Instrumented rollouts measure a 72.4 ms start-to-start period against the 66.7 ms the policy trained on: `stream_sub_targets` paces the full wall tick and then the encoder read, predict, lockstep-sim step, and logging add ~5.7 ms on top — a systematic 8.6% time stretch (behaves like a mild `--slow`). Fix in `ArmLoop.tick`: shrink the streaming window by the measured per-tick overhead (or deadline the loop start-to-start) so the realized period matches `control_dt`; re-verify with the `loop_ms` telemetry.

- **Marker obs noise is isotropic in sim; real solvePnP error is not.** Sim draws iid Gaussian xyz noise per tag (`marker_pos_sigma`), but the real error is dominated by the camera's optical axis (scale-based depth, several times worse than the image-plane axes) plus angle-dependent wobble. The camera pose in the base frame is known from the solved extrinsics, so rotate the noise covariance into the camera frame with a larger depth sigma. Same fix should add the per-frame *common-mode* term: the pipeline re-anchors the camera from the table tag every frame (`real/marker_obs.py`), so table-tag noise hits BOTH arm tags identically — sim noise is currently independent per tag.

- **Correlated marker dropout unmodeled.** When the real pipeline loses the *table* tag it stales BOTH arm tags at once (`real/marker_obs.py` can't re-anchor the camera, so both held poses stop updating), while sim dropout (`marker_dropout`) is rolled independently per tag — add a correlated all-tags dropout mode if real logs show it matters. (Camera latency itself is done: `src/camera_sim.py` + `sysid/probe_cam_latency.py`, measured `delay_ms: [42, 52]` in `conf/dr`.)

- **Bursty marker dropout.** Sim dropout is i.i.d. per frame (conditioned on the grazing-angle band), but real detector misses cluster in bursts (a grazing view or partial occlusion persists across frames). Since undetected tags now *hold* their last pose while the age channel grows, burst length directly shapes the held-pose error distribution the policy trains against — consider a two-state Markov dropout (p_enter, p_exit) in `base_env._process_frame`, tuned against measured real dropout runs.

- **Marker observations: tune sim dropout against measured real rates; site rotations still eyeballed.** Mostly done: the camera feeds measured tag poses into rollouts (`real/marker_obs.py`, `--marker-source camera`, holding undetected tags at their last pose with a growing age channel like training), the site *positions* are calibrated (`real/calibrate_qpos.py` writes the freed-mount solve back into `so101.xml`), and `tag_cam`'s sim pose is set from the solved extrinsics. Still open: the site *rotations* are an eyeballed estimate plus the `quarter_turns` in-plane snap — fine while the obs uses positions, revisit if marker_rot ever matters; and the sim visibility model (plane-angle cutoff + the `dr` group's `marker_dropout`, higher in the grazing 65–70° band) has never been checked against measured real-camera dropout rates. There is deliberately no reward term for a hidden tag: losing its pose is penalty enough.

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
