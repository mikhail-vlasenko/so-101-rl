# Real-Rig Calibration Plan

Bias DR (per-episode constant offsets on qpos / EE / cube) is **disabled** in
training. Instead, we calibrate the real arm and vision pipeline once and
subtract the measured offsets before feeding observations to the policy. The
policy code is identical sim→real; only a thin wrapper applies the calibration.

## What we measure

- `qpos_bias[6]` — per-joint encoder zero-offset, rad. `θ_true = θ_encoder − b_i`.
- `cube_pos_bias[3]` — vision pipeline systematic offset in base frame, m.

EE bias is *not* measured separately: EE position is computed via forward
kinematics from calibrated qpos, so once qpos is right, EE is right.

## Procedure (ArUco-based)

Hardware:
- Calibrated camera (intrinsics from a checkerboard pass).
- Two ArUco markers: one rigidly mounted on the gripper, one on the cube
  (or its placeholder for cube-channel calibration).
- The arm bolted to a known frame (define base frame from a fixed fiducial on
  the table).

### Step 1 — qpos offsets

1. Mount marker rigidly on the gripper; record its pose in the gripper frame
   (mechanical CAD or a one-time hand-eye solve).
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
cube_pos_bias: [x, y, z]                   # m, base frame
```

The real-robot wrapper, before each `policy.predict(obs)`:

```python
obs[:6]                          -= qpos_bias
obs[cube_idx:cube_idx + 3]       -= cube_pos_bias
```

`qvel`, `ee_pos`, and `cube_to_target` are derived: `qvel` from differentiating
calibrated qpos, `ee_pos` from FK on calibrated qpos, `cube_to_target` from
`(cube_pos − cube_pos_bias) − target`. No extra subtractions needed there.

## When to recalibrate

- After any collision hard enough to flex the mount or move the camera.
- After re-seating the arm or the camera.
- If on-table behavior visibly degrades vs. the previous calibration's
  validation error.

A drift > ~0.5° per joint or > 5 mm cube offset is the signal to redo Step 1
or Step 2 respectively.
