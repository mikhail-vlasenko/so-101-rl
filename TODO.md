# TODO

Long-term tasks and ideas. Not a changelog.

## Sim-to-real fidelity (reach env)

Current symptom: with sim/real kp aligned at 64, real arm has trouble holding/lifting itself even though sim trains fine. Also worth noting: experiments so far have been on the **leader** arm, not the follower the policy will ultimately deploy on. Leader has different mechanical load (no payload, no gripper jaws engaged, possibly different calibration), so some of this gap may collapse once we move to the follower. Independently though, sim has several optimistic assumptions:

- **Link masses / inertias.** Menagerie's `so101.xml` derives inertials from STL meshes at an assumed density (likely plastic). Real arm is metal servos + steel screws. Sum body masses in the XML, compare against the weighed arm. If sim is lighter, real torque demand exceeds what the policy learned.

- **Compliance during contact.** The linear gravity-compliance model (`q_true = q_enc − b − c·τ_grav` for lift/elbow/wrist_flex) is solved and deployed (`real/compliance.py`, `ArmLoop`). Open follow-up: deployment reads τ from the arm's gravity `qfrc_bias` only, which is wrong once the gripper loads up during contact — modeling the compliance in sim (springy `*_play` joints) is the principled alternative if contact-regime marker accuracy ever matters.

- **Elbow (and general) command tracking: real under-executes vs the refit sim.** Diagnosed via `sysid/replay_rollout.py` on the 2026-07-05 floor-pressing lift rollouts (run246, seeds 0/2): replaying the recorded actions open-loop in sim never touches the floor (min ee_z +0.8/+2.0 cm, zero contact force) while the real arm pressed to −0.4/−0.8 cm, and the closed-loop policy from the same start lifts cleanly in sim — so the press is a *plant* gap, not an obs gap (counterfactual FK-marker re-predictions match the recorded actions almost exactly; camera markers didn't change the policy's mind). Per-tick tracking regression (realized qpos delta on commanded delta, two taps): real gains run 10–25 % below sim on every arm joint, worst at elbow_flex (0.34 vs 0.45; per-joint qpos divergence mean 0.16–0.36 rad across the two rollouts). Candidates: the forcerange/continuous-torque and mass items above, per-joint kp mismatch, elbow load in these poses. Fix ideas: refit elbow damping/armature against closed-loop policy-style trajectories (current sysid trajectories may under-weight the loaded elbow regime), or lower sim tracking to match real. Related: the measured qpos reads ~0.5–1 cm below the true pose *while pressing* (contact reverses the compliance load — the known contact-regime compliance gap above), which keeps re-commanded hold targets below the floor and sustains the push.

- **Marker noise anisotropy — remaining pieces.** Position noise is now anisotropic in the camera frame (`src/marker_noise.py`: small lateral, large depth along each tag's ray, sigmas derived from the calibrated intrinsics + per-tag distance/size), plus a per-episode common-mode bias (`obs_bias.marker_common_sigma`) and a small per-frame common-mode residual (`obs_noise.cam_common_sigma`) shared across all tags — the table re-anchor, whose real-side jitter `real/marker_obs.py` now damps by EMAing the static camera (`CAM_EMA_ALPHA`). Open: (1) rotation noise is still isotropic (`marker_rot_sigma`) — the angle-dependent orientation wobble isn't modeled (only the cube-tag rot is in the obs); (2) the per-frame common-mode residual is isotropic, not aligned to the camera optical axis (would need `data.cam_xmat`); (3) the new DR knobs (`tag_px_noise`, `tag_depth_factor`, `cam_common_sigma`, `marker_common_sigma`) and `CAM_EMA_ALPHA` are seeded from geometry/back-of-envelope, not yet tuned against measured real solvePnP logs — ties into the dropout-rate tuning item below.

- **Correlated marker dropout unmodeled.** When the real pipeline loses the *table* tag it stales BOTH arm tags at once (`real/marker_obs.py` can't re-anchor the camera, so both held poses stop updating), while sim dropout (`marker_dropout`) is rolled independently per tag — add a correlated all-tags dropout mode if real logs show it matters. (Camera latency itself is done: `src/camera_sim.py` + `sysid/probe_cam_latency.py`, measured `delay_ms: [42, 52]` in `conf/dr`.)

- **Bursty marker dropout.** Sim dropout is i.i.d. per frame (conditioned on the grazing-angle band), but real detector misses cluster in bursts (a grazing view or partial occlusion persists across frames). Since undetected tags now *hold* their last pose while the age channel grows, burst length directly shapes the held-pose error distribution the policy trains against — consider a two-state Markov dropout (p_enter, p_exit) in `base_env._process_frame`, tuned against measured real dropout runs.

- **Arm-tag occlusion not raycast.** The cube tag's sim detection includes an `mj_ray` occlusion test (`base_env.cube_tag_occluded`), but the arm markers still use the plane-angle + height check only — sim is optimistic when the arm self-shadows a tag from the camera. Extend the ray test to the marker sites (exclude the site's own body like the cube does).

- **Pickplace real rollout script.** `real/rollout_lift.py` now tracks the real sponge via its tag; pickplace needs the equivalent (place-target definition on the real table, ring pose, phase-aware termination) built on `ArmLoop` + `CameraMarkerSource(track_cube=True)`.

- **Marker observations: tune sim dropout against measured real rates; site rotations still eyeballed.** Position calibration and the camera marker source are done (`real/marker_obs.py`, `real/calibrate_qpos.py`). Two open remainders: (1) the site *rotations* are an eyeballed estimate plus the `quarter_turns` in-plane snap — fine while the obs uses positions, revisit if `marker_rot` ever matters; (2) the sim visibility model (plane-angle cutoff + the `dr` group's `marker_dropout`, higher in the grazing 65–70° band) has never been checked against measured real-camera dropout rates. There is deliberately no reward term for a hidden tag: losing its pose is penalty enough.

## Distillation: frame-stacked / history students

The DAgger distillation rig is built (`src/distill.py`, see CLAUDE.md "Distillation"), but `privileged` mode asserts `frame_stack=1`: feeding a history-augmented teacher needs a stacked privileged-obs history. Build alongside the obs-history wrapper (`.claude/plans/obs_history_features.md`). The long-pole consumer — the tag→box obs migration (`.claude/plans/vision_multicam_longterm.md`) — is already designed around this rig existing.

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
