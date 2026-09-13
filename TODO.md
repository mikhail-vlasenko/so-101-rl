# TODO

Unresolved, actionable work only. Not a changelog or project-status document.

## Tag-free object tracking (dual C922 + SAM), follow-ups

- **Pickplace dense-stereo rollout.** Add the equivalent camera object source to
  pickplace with a real-table target, ring pose, and phase-aware termination on
  `ArmLoop`.
- **Delete the legacy tag remnants when the legacy teacher retires.** The
  `cube_tag` site in the scene XMLs and the GT tag-pose helper survive only
  for `distill.teacher_obs=legacy_tag` (migrating the last tag-obs
  checkpoint); delete both, plus the mode, once no tag-obs teacher matters.

## Sim-to-real fidelity (reach env)

- **Re-test the plant gap on the follower arm.** Repeat the holding and lift
  probes on the deployment arm before changing the simulator, because the
  existing measurements came from the mechanically different leader arm.
- **Correct link masses and inertias.** Weigh the real arm or its separable
  links, compare them with `so101.xml`, and update the model where the assumed
  mesh density is wrong.
- **Measure and model contact compliance.** Record tag-versus-encoder residuals
  while the gripper is loaded; if the contact error is material, add compliant
  `*_play` joints in sim and cover the sim/real behavior with twin tests.
- **Match loaded command tracking.** Capture policy-style trajectories that
  load the elbow, refit damping/armature and per-joint tracking, then verify the
  floor-contact replay no longer diverges between sim and real.
- **Calibrate arm-marker noise and dropout from logs.** Record solvePnP error,
  table-anchor jitter, grazing-angle misses, and dropout burst lengths; tune
  `tag_px_noise`, `tag_depth_factor`, `marker_common_sigma`, `CAM_EMA_ALPHA`, and
  the visibility/dropout model. Add a two-state dropout model only if the logs
  show that burst duration matters.
- **Raycast arm-marker occlusion.** Extend the marker visibility test beyond
  plane angle and FOV so arm links can occlude their marker sites, excluding
  each site's own body from its ray test.

## Asymmetric critic: follow-ups

- **Evaluate the privileged tail where it can affect learning.** Compare value
  quality and policy success with and without the tail during from-scratch
  curriculum training and `dr=full` refinement, then decide whether to retain
  it.

## Lift policy deployment

- **Compare refined and reference policies on hardware.** Test first-attempt
  success and object disturbance with unchanged perception; use
  `bps_ee_clamp_refine_20260913/final_model.zip` versus
  `bps_proper_grasp_light3/final_model.zip` under `logs/ppo_lift/`.
- **Improve clearance and retention.** Train wider jaw/depth margins, require
  sustained unsupported lift, and prevent repeated height credit after drops.
  Resolve the lift cases in `tests/env/test_review_regressions.py`.
- **Check object-perception parity.** Compare real/sim centroids and static gates
  during approach/occlusion; measure stereo timing errors before adding more noise.
- **Revisit policy memory if failures persist.** Measure clamp binding and compare
  last-target observations or short history against the corrected baseline;
  migrate checkpoints if the observation layout changes.

## Object generalization and stacking

- **Support more objects.** Add boxes/cylinders with matching surface observations
  and physical properties; replace sponge-specific grasp checks.
- **Build a two-sponge placement primitive.** Require released, stable placement
  in pickplace (resolve its review regression), then add a separately observed
  fixed base sponge and evaluate stacking after hand withdrawal.

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
