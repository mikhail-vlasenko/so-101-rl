# TODO

Long-term tasks and ideas. Not a changelog.

## Sim-to-real fidelity (reach env)

Current symptom: with sim/real kp aligned at 64, real arm has trouble holding/lifting itself even though sim trains fine. Also worth noting: experiments so far have been on the **leader** arm, not the follower the policy will ultimately deploy on. Leader has different mechanical load (no payload, no gripper jaws engaged, possibly different calibration), so some of this gap may collapse once we move to the follower. Independently though, sim has several optimistic assumptions:

- **Link masses / inertias.** Menagerie's `so101.xml` derives inertials from STL meshes at an assumed density (likely plastic). Real arm is metal servos + steel screws. Sum body masses in the XML, compare against the weighed arm. If sim is lighter, real torque demand exceeds what the policy learned.

- **Compliance during contact.** The linear gravity-compliance model (`q_true = q_enc − b − c·τ_grav` for lift/elbow/wrist_flex) is solved and deployed (`real/compliance.py`, `ArmLoop`). Open follow-up: deployment reads τ from the arm's gravity `qfrc_bias` only, which is wrong once the gripper loads up during contact — modeling the compliance in sim (springy `*_play` joints) is the principled alternative if contact-regime marker accuracy ever matters.

- **Elbow (and general) command tracking: real under-executes vs the refit sim.** Diagnosed via `sysid/replay_rollout.py` on the 2026-07-05 floor-pressing lift rollouts (run246, seeds 0/2): replaying the recorded actions open-loop in sim never touches the floor (min ee_z +0.8/+2.0 cm, zero contact force) while the real arm pressed to −0.4/−0.8 cm, and the closed-loop policy from the same start lifts cleanly in sim — so the press is a *plant* gap, not an obs gap (counterfactual FK-marker re-predictions match the recorded actions almost exactly; camera markers didn't change the policy's mind). Per-tick tracking regression (realized qpos delta on commanded delta, two taps): real gains run 10–25 % below sim on every arm joint, worst at elbow_flex (0.34 vs 0.45; per-joint qpos divergence mean 0.16–0.36 rad across the two rollouts). Candidates: the forcerange/continuous-torque and mass items above, per-joint kp mismatch, elbow load in these poses. Fix ideas: refit elbow damping/armature against closed-loop policy-style trajectories (current sysid trajectories may under-weight the loaded elbow regime), or lower sim tracking to match real. Related: the measured qpos reads ~0.5–1 cm below the true pose *while pressing* (contact reverses the compliance load — the known contact-regime compliance gap above), which keeps re-commanded hold targets below the floor and sustains the push.

- **Marker noise anisotropy — remaining pieces.** Position noise is now anisotropic in the camera frame (`src/marker_noise.py`: small lateral, large depth along each tag's ray, sigmas derived from the calibrated intrinsics + per-tag distance/size), plus a per-episode common-mode bias (`obs_bias.marker_common_sigma`) — the table re-anchor, whose real-side jitter `real/marker_obs.py` damps by EMAing the static camera (`CAM_EMA_ALPHA`); a per-frame common-mode residual knob existed briefly and was removed as negligible post-EMA. Open: (1) rotation noise is still isotropic (`marker_rot_sigma`) — the angle-dependent orientation wobble isn't modeled (only the cube-tag rot is in the obs); (2) the new DR knobs (`tag_px_noise`, `tag_depth_factor`, `marker_common_sigma`) and `CAM_EMA_ALPHA` are seeded from geometry/back-of-envelope, not yet tuned against measured real solvePnP logs — ties into the dropout-rate tuning item below.

- **Pose-locked (semi-static) tag noise — cube first.** Sim's per-tag noise is i.i.d. per frame, but a static tag under static lighting feeds the detector nearly the same pixels every frame, so the real error is mostly a *repeatable, pose-dependent bias* (corner quantization, calibration residual, print/glue geometry, planar-pose depth ambiguity) plus a small temporal jitter from sensor noise — it neither re-draws every frame the way `_process_frame` samples it, nor averages away over frames on the real rig. Risk: the sim policy learns temporal filtering that transfers as a persistent offset instead. The cube tag is the worst case (it sits still most of the episode; real reads a near-constant wrong pose, sim dances along the ray) and the natural first implementation target; the constantly-moving finger/wrist tags decorrelate over ~1 px of image motion anyway, so i.i.d. is a fair approximation there. Fix sketch: split `tag_px_noise` into a pose-locked component (per-tag draw in the same camera-frame ellipsoid, re-drawn/OU-updated as a function of tag *displacement and view-angle change*, not time) + a small truly-per-frame jitter component. First measure the split on the rig: log a static tag for ~30 s (`real.marker_view`) — frame-to-frame pose std = temporal jitter (10 ms exposure at gain 255 may make this non-trivial); offset from FK = the pose-locked part. Until this lands, the px knobs are held at the un-inflated calibration residual (arm markers `tag_px_noise` 0.2 full / 0.1 light; cube `cube_px_noise` 0.2 in both — ~old-iso overall magnitude, ~4 mm vec on the cube) so the i.i.d. approximation doesn't make the DR stages harder than the model they replaced; once the split exists, the pose-locked share can carry the honest deploy-conditions inflation (~0.4 px total) without adding temporal churn.

- **Correlated marker dropout unmodeled.** When the real pipeline loses the *table* tag it stales BOTH arm tags at once (`real/marker_obs.py` can't re-anchor the camera, so both held poses stop updating), while sim dropout (`marker_dropout`) is rolled independently per tag — add a correlated all-tags dropout mode if real logs show it matters. (Camera latency itself is done: `src/camera_sim.py` + `sysid/probe_cam_latency.py`, measured `delay_ms: [42, 52]` in `conf/dr`.)

- **Bursty marker dropout.** Sim dropout is i.i.d. per frame (conditioned on the grazing-angle band), but real detector misses cluster in bursts (a grazing view or partial occlusion persists across frames). Since undetected tags now *hold* their last pose while the age channel grows, burst length directly shapes the held-pose error distribution the policy trains against — consider a two-state Markov dropout (p_enter, p_exit) in `base_env._process_frame`, tuned against measured real dropout runs.

- **Arm-tag occlusion not raycast.** The cube tag's sim detection includes an `mj_ray` occlusion test (`base_env.cube_tag_occluded`), but the arm markers still use the plane-angle + height check only — sim is optimistic when the arm self-shadows a tag from the camera. Extend the ray test to the marker sites (exclude the site's own body like the cube does).

- **Pickplace real rollout script.** `real/rollout_lift.py` now tracks the real sponge via its tag; pickplace needs the equivalent (place-target definition on the real table, ring pose, phase-aware termination) built on `ArmLoop` + `CameraMarkerSource(track_cube=True)`.

- **Marker observations: tune sim dropout against measured real rates; site rotations still eyeballed.** Position calibration and the camera marker source are done (`real/marker_obs.py`, `real/calibrate_qpos.py`). Two open remainders: (1) the site *rotations* are an eyeballed estimate plus the `quarter_turns` in-plane snap — fine while the obs uses positions, revisit if `marker_rot` ever matters; (2) the sim visibility model (plane-angle cutoff + the `dr` group's `marker_dropout`, higher in the grazing 65–70° band) has never been checked against measured real-camera dropout rates. There is deliberately no reward term for a hidden tag: losing its pose is penalty enough.

## Asymmetric critic: follow-ups

The `[actor block | privileged tail]` layout is live (see CLAUDE.md, `base_env.priv_dim_for`), with `logs/ppo_lift/asym_critic_run264.zip` the first fine-tuned artifact (0.96 lift success, dr=light). Open:

- **Tail utilization is shallow after a short warm fine-tune.** After 10M steps from the migrated run262, the critic's privileged input columns carry ~14x less weight than the actor columns (top consumers: true cube velocity, jaw contact flags, cube-tag bias). The feature should pay most where value noise actually dominates — from-scratch curriculum stages and dr=full refines — not a near-converged dr=light polish; measure there before judging its value.

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
