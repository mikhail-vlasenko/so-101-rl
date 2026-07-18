# Real-Rig Calibration Plan

Bias DR (per-episode constant offsets on qpos / EE / cube) is **disabled** in
training. Instead, we calibrate the real arm and vision pipeline once and
subtract the measured offsets before feeding observations to the policy. The
policy code is identical sim→real; only a thin wrapper applies the calibration.

## What we measure

- `qpos_bias[6]` — per-joint encoder zero-offset, rad.
- `compliance[6]` — per-joint elastic deflection under gravity load, rad per N·m
  (lift/elbow/wrist_flex; 0 on the rigid joints). Together:
  `θ_true = θ_encoder − b_i − compliance_i · τ_grav,i`.
- `cube_pos_bias[3]` — vision pipeline systematic offset in base frame, m.

EE bias is *not* measured separately: EE position is computed via forward
kinematics from calibrated qpos, so once qpos is right, EE is right.

(Optional, beyond this plan: the wrist tag below can also feed the EE
observation *live* instead of via FK, turning open-loop encoder trust into
closed-loop visual feedback that self-corrects residual calibration error. This
plan covers the offline-bias approach; live EE is an extension.)

## Fiducial markers

We print both ArUco and AprilTag families and choose later — the pipeline is
marker-agnostic. Notes for the choice:
- AprilTag (`tag36h11`) is more robust to motion blur and small apparent size,
  but needs its native detector (`pupil-apriltags`) for that robustness;
  OpenCV's `aruco` module can *decode* AprilTag dictionaries yet still detects
  them with the ArUco pipeline, so it doesn't buy the blur robustness.
- ArUco is simpler and native to OpenCV, and its **ChArUco** board is the
  better target for the camera-intrinsics pass.

Tags serve two distinct jobs — world frame vs. end-effector:

| Tag | Frame | Role |
|-----|-------|------|
| table (≥1, known layout) | world / base | camera extrinsics; re-localize if the camera is moved |
| arm base | robot base | ties the base frame into the world without trusting the bolt position |
| wrist (rigid gripper body) | EE | primary end-effector pose source / Step-1 marker |
| static finger | EE | bonus EE detection; expect occlusion during grasp and peel risk |

Mounting rules:
- **Unique ID per tag, recorded** (e.g. 0=wrist, 1=finger, 2=base, 10–13=table).
  The detector tells tags apart only by ID.
- **Per-ID physical size in config**: table/base tags large (far-range
  accuracy), EE tags small to fit. The solver needs the true printed size per ID.
- **Flat and rigid.** Double-sided tape is fine, but on the *moving* tags it can
  creep — re-verify the EE tags after collisions. A tag taped to a curved finger
  won't be planar and will bias the pose.
- **Table tags need a known layout** to localize against: either a
  known-geometry board (ChArUco / AprilGrid) or measure scattered singles into
  the table frame once (pick one tag as the origin).

## Procedure (fiducial-based)

Hardware:
- Calibrated camera (intrinsics from a ChArUco or checkerboard pass). **Done — see
  Step 0.**
- The fiducials above: wrist tag (qpos / EE), a cube tag or marker block (cube
  channel), table tags (base frame), arm-base tag (base cross-check).
- The arm bolted to a known frame; base frame defined by the table tags and
  cross-checked by the arm-base tag.

### Step 0 — camera intrinsics (DONE)

Calibrated the Logitech C922 from a plain checkerboard (8×10 squares = 7×9 inner
corners, 20 mm). Result: **RMS ~0.21 px** over ~29 views, stable across repeat
runs (`fx≈fy≈968`, `cx≈648`, `cy≈335`). Saved to `real/vision/camera_intrinsics.yaml`
(camera_matrix, dist_coeffs, focus_absolute, pattern, square size).

Tools (all in `real/`):
- `calibrate_camera.py` — live checkerboard capture + `cv2.calibrateCamera`.
  Auto-detects the inner-corner count, captures only when the board is *still*
  (no motion blur), and auto-prunes outlier views before the final fit.
- `focus_picker.py` — sharpness sweep (variance of Laplacian) to choose a focus.
- `camera.py` — `open_camera` / `v4l2_set`, the single source of camera settings.

What we learned (C922 / UVC quirks — don't relitigate):
- **What actually fixed the high RMS (1.67 px → ~0.2):** two script changes that
  landed together — capturing only when the board is *still* (kills motion blur)
  and freezing focus (no autofocus drift between views). They changed in the same
  run, so we can't attribute the gain to one over the other; both matter. The
  sheet was unchanged between those runs, so board flatness was *not* the cause
  here — though a rigid flat backing is still good practice (a bowed sheet biases
  pose) and worth doing for the pose steps below.
- **Hold the board, don't lay it flat.** Intrinsics need many *tilted* views at
  varied distance with corners pushed into the frame edges. A single coplanar
  shot is degenerate.
- **Focus must be pinned and recorded.** Focus is locked at `focus_absolute=30`
  (`FOCUS_ABSOLUTE` in `calibrate_camera.py`, also written to the YAML). The
  intrinsics are valid *only* at that focus — **the rig must open the camera with
  `focus=30`** or the focal length differs and the calibration is void.
- **C922 `focus_absolute` is a manual target only.** With continuous autofocus
  on, firmware drives the lens internally and never reports the position back (it
  reads 0), so you can't "let AF settle and read the value" — pick focus by
  sharpness sweep instead.
- **OpenCV `CAP_PROP_FOCUS` silently no-ops** on this camera; set focus via
  `v4l2-ctl` (`focus_automatic_continuous=0` then `focus_absolute=N`), which is
  what `open_camera(focus=N)` now does.
- Autofocus drifting between views was a real error source (`fx`/`fy` split apart
  until we froze focus); blur from capturing mid-motion was another.

### Step 1 — qpos offsets

**Implemented** by `real/calib/calibrate_qpos.py` (self-driven, position-only). The arm
drives *itself* to a spread of sim-generated poses (collision-free, in-limits,
arm tag facing the camera), captures `(encoder qpos, arm-tag tvec)`, and solves
`qpos_bias` *jointly* with `T_base_cam` so FK(θ_enc − b) lands the tags where the
camera sees them. One run writes both `calibration.yaml` (`qpos_bias`, `compliance`)
and `extrinsics.yaml` (`t_base_cam_fixed`, `t_base_table`, `quarter_turns`).

Deviations from the orientation-based sketch below, deliberate for a first cut:
- **Position-only** (tag centres), so it inherits immunity to solvePnP rvec flips
  on the small arm tags — at the cost of weaker observability on axial joints.
- **Tag mounts freed (position-only).** Each arm tag's 3D centre offset in its
  parent body frame is solved jointly with the bias, under a strong XML prior, so a
  hand-taped tag a few mm off CAD isn't absorbed as a fake encoder bias. The mount
  *rotation* is still trusted (position-only — a centre has no orientation; the glue
  rotation is handled separately by `quarter_turns`). The solved site centres are
  written back into `so101/so101.xml`: deployment reads camera-measured tags, but
  *training's* marker observations are FK of the sites, so a site that doesn't match
  the physical tag is a systematic sim-vs-real offset in the marker channels. The
  finger offset is an exact gauge pair with the wrist_roll bias (roll is the last
  joint before the finger's body and no other tag sees roll), which the prior resolves
  in favour of the bias.
- **pan & gripper biases pinned to 0.** A pan bias is a base-z rotation that a
  yaw of `T_base_cam` reproduces exactly (gauge-degenerate, unobservable); the
  camera absorbs it and deployment stays self-consistent. The gripper joint moves
  neither tag. The other four joints (lift/elbow/wrist_flex/wrist_roll) are solved.
- **Gravity compliance solved too.** On top of the constant bias, the lift/elbow/
  wrist_flex links flex elastically under gravity, so a per-joint coefficient
  `compliance_i` (rad per N·m) is solved jointly with the bias and mounts and applied
  as `θ_true = θ_enc − b − compliance·τ_grav(θ_enc − b)` at both solve and deployment
  time (`real/calib/compliance.py`). It roughly halves the residual (committed run 8.2 → 2.4
  mm RMS; cross-validated held-out 3.6 → 2.0 mm mean, 8.2 → 4.7 mm max). The backlash
  probe (`sysid/probe_backlash.py`) confirmed the mechanism is elastic flex, not gear
  play (link-vs-motor hysteresis ≤0.2°).

The orientation-based optimisation below is the fuller version (adds the axial
observability and the freed mount), kept as the upgrade path if position-only
residuals plateau above target.

1. Mount the wrist tag rigidly on the gripper body and fix its pose in the
   gripper frame, `T_marker_in_gripper`. With a printed bracket this is known
   from CAD; with double-sided tape it is unknown, so add it as free variables
   in the Step-3 optimization (a joint hand-eye + encoder-bias solve).
2. Drive the arm to ~10 well-spread poses (cover the workspace, avoid
   singularities). At each pose, record:
   - Encoder reading `θ_enc[6]`.
   - Marker pose in base frame from the camera.
3. Solve

       b* = argmin_b  Σ_k || FK( θ_enc,k − b ) ⊕ T_marker_in_gripper  −  T_marker_observed,k ||²

   where the loss is on translation (mm) plus orientation (axis-angle, rad)
   with a sensible weighting (e.g. 1 m ↔ 1 rad).
4. Validation: command a fresh known pose, apply `b`, measure marker error.
   Target: < 3 mm translation and < 1° orientation. If worse, fix the
   calibration — don't paper over it by re-enabling bias DR.

### Step 2 — cube vision offset

1. Place the cube (or a marker block of identical size) at ~10 known points
   on the table, measured against the same base-frame fiducial.
2. For each, record the cube-pos estimate from the production vision
   pipeline (the same code path the policy will consume on the real rig).
3. Fit a constant 3-vector `cube_pos_bias` minimizing
   `||vision_estimate − ground_truth − cube_pos_bias||²`. If the residual
   is structured (e.g. scales with distance), the pipeline has a non-constant
   error and a single bias vector is the wrong model — revisit before deploy.

## Storage and deploy-time wrapper

Save results to `calibration.yaml`:

```yaml
qpos_bias:     [b1, b2, b3, b4, b5, b6]   # rad
compliance:    [c1, c2, c3, c4, c5, c6]   # rad per N·m (0 on rigid joints)
cube_pos_bias: [x, y, z]                   # m, base frame
```

The real-robot wrapper (`ArmLoop`, `real/rollout/rollout_common.py`), on every encoder read,
maps raw → true joint angle through the bias *and* the gravity compliance:

```python
q_bc   = raw_to_rad(raw) - qpos_bias
qpos   = q_bc - compliance * τ_grav(q_bc)      # real/calib/compliance.py
obs[cube_idx:cube_idx + 3] -= cube_pos_bias
```

and inverts it on every write (fixed-point, so a hold command doesn't creep). `qvel`,
`ee_pos`, and `cube_to_target` are derived: `qvel` from differentiating calibrated
qpos, `ee_pos` from FK on calibrated qpos, `cube_to_target` from
`(cube_pos − cube_pos_bias) − target`. No extra subtractions needed there.

## When to recalibrate

- After any collision hard enough to flex the mount or move the camera.
- After re-seating the arm or the camera.
- If on-table behavior visibly degrades vs. the previous calibration's
  validation error.

A drift > ~0.5° per joint or > 5 mm cube offset is the signal to redo Step 1
or Step 2 respectively.
