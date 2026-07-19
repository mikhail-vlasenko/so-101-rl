# Tag-free sponge tracking: dual-channel SAM stereo obs with shape tensor √M

## Context

The cube obs is currently the raw pose of AprilTag id 1 on the sponge's largest
face (`.claude/plans/real_cube_tracking.md`). Goal: replace it with a tag-free,
object-agnostic pipeline — text-prompted SAM masks from both C922s — so the
approach extends to arbitrary objects (per-object cost = one text prompt, no
markers, no per-object models). A box is symmetric under 180° flips, so raw
rotation representations are ill-posed; the policy needs orientation only up to
the object's symmetry — "which way is it shorter."

Full retrain accepted (obs-dim change). The real sponge keeps its tag(s) for
**ground truth in evaluation only** — never in the rollout obs path.

## Decisions

1. **Obs = two cube channels**, both under the existing hold-last-pose + age
   convention (`MARKER_AGE_CAP_S`):
   - **Live** (fast, imprecise, current): triangulated 3D centroid of the two
     visible-mask centroids (3) + age (1). Biased (visible-surface point,
     partial occlusion shifts it) but tells the policy "it moved / I'm holding
     it / it dropped". Requires **both** views (a single view gives only a ray;
     mono fallback is out of scope for v1 — revisit with dataset numbers,
     TODO.md). One view lost → live goes stale, age grows.
   - **Precise** (slow, accurate, refreshed only when static): estimated body
     center (3) + **√M** (6, upper triangle) + age (1). Refresh gate: object
     static (live-centroid speed below threshold for a dwell) AND visible
     fraction high in both views; multi-frame averaged over the static window.
     Grasped/occluded → held with growing age; by then orientation did its job.
2. **√M, not M or angles**: M = R·diag(hx²,hy²,hz²)/3·Rᵀ is the volume's
   second-moment (covariance) matrix; feed its principal square root
   √M = R·diag(hx,hy,hz)/√3·Rᵀ — units of meters (conditions sanely in
   obs_norm), symmetric 3×3 → 6 numbers, invariant under all box symmetries,
   continuous over SO(3), defined for any future object shape. dᵀ(√M)²d =
   squared spread along d. Encodes the episode's actual size (couples with
   `cube_size_jitter`).
3. **Estimator v1 = visual hull** from the two silhouette cones ∩ workspace
   box ∩ z ≥ table: voxel sample → centroid + covariance → √M. No learned
   parts, no per-object knowledge. Escalation (only if eval fails the
   acceptance criterion): learned-stereo fusion or a synthetic-trained
   masks→√M net — both stay per-object-free; choose from measured failure
   modes.
4. **Eval harness before estimator**: a recorded dataset with tag GT decouples
   estimator work from the rig and makes every candidate an offline
   experiment. **Acceptance criterion (fixed now):** on held-out static
   windows, per-component errors must sit within the `dr=full` noise envelope
   (2σ of the corresponding knobs, read from the yaml at eval time — no
   duplicated numbers); occlusion-sliced live-centroid error within its knob
   likewise. Ship when green, escalate when not.
5. **Single source of truth for the twin logic**: the dual-channel hold-last /
   age / static-gate state machine becomes one shared module imported by both
   the sim env and the real rollout (`src/shape_obs.py`), replacing today's
   duplicated-and-contract-tested convention for the cube. (Arm markers keep
   their existing twin pair.)
6. **Sim models both cameras**: add `tag_cam_aux` to the scene; live/precise
   detection requires both views, mirroring the real gates. Sim's live
   centroid = per-camera visible-surface-point centroid (face-normal facing
   test + `mj_ray` occlusion on sampled cube-surface points), averaged across
   the two cameras — reproducing the visible-surface bias instead of
   pretending the live channel sees the true center.
7. **Spawns broadened**: no tag to face a camera → all resting faces (flat on
   the largest face included) + free yaw. New curriculum crutch flag
   `cube_no_flat_spawns` (analogous to `cube_smallest_face_only`, no obs-dim
   change, safe to flip across resume). Spawn rejection-samples on the new
   visibility (live channel detectable, arm already placed).
8. **Training = distill first, scratch as fallback**: new
   `distill.teacher_obs: legacy_tag` mode assembles the teacher's old 37-dim
   frame from sim GT (tag-site pose, always fresh — privileged-style) while
   the student trains on the new obs. The `cube_tag` scene site and a minimal
   GT pose helper survive **only** for this mode (note in TODO.md: delete both
   when the legacy teacher is retired). Fallback: full curriculum
   (dr=none → light → full, shaping none → light → full, crutch flags staged).
9. **Termination on the real rig = live channel**: dwell on live centroid z ≥
   target_height while live age < `CUBE_FRESH_DWELL_S`. `tag_center_z` and the
   tag-pose back-derivation are deleted, not ported.
10. **Aux camera repositioned — done** (~45° angular separation from the scene
    center; angular diversity is the cheapest accuracy win for both
    triangulation and the hull). Extrinsics need no rework (per-session
    table-tag anchoring). Residual checks before Stage 2 recording: re-run
    `real.tracking.stereo_check` at the new geometry (require the validated
    sub-mm repeatability; focus re-check if the mount distance changed) and
    snapshot aux `T_base_cam` for the sim camera pose. If Stage 3 eval shows
    the hull still starved along the mean viewing direction, widening beyond
    45° is the first lever to revisit — before any estimator escalation.

New actor block: `qpos(6), qvel(6), markers(2·3), marker_age(2), live(3),
live_age(1), center(3), sqrtM(6), precise_age(1), extra(4), prev_actions(N·6)`
— cube block 7 → 14 dims, base 37 → 44 at `prev_actions_n=1`. Invalidates all
checkpoints (accepted). `_obs_extra`'s `cube_to_target` (pickplace) switches to
the **live** centroid — the carrying phase needs "where is it now".

## Stage 1 — sim obs contract (~2 d)

- **`src/shape_obs.py`** (new; importable from both `src/` and `real/`):
  `box_sqrtm(R, half_extents)`, `sqrtm_from_cov(cov)` (eigh, clamps tiny
  negatives), `sqrtm_upper(S) -> (6,)` (order xx,yy,zz,xy,xz,yz — pinned
  here), `ObjectObsState` (pure dual-channel hold-last/age state machine:
  `ingest_live(t, centroid|None)`, `ingest_precise(t, center, sqrtM)`,
  `serve(t) -> (live, live_age, center, sqrtM, precise_age)`; never-seen =
  zeros + age at cap), `is_static(times, centroids)` + the gate constants
  (`STATIC_SPEED_MAX_M_S`, `STATIC_DWELL_S`, `VISIBLE_FRACTION_MIN`).
- **Scene XMLs**: add `tag_cam_aux` (pose from the decision-10 `T_base_cam`
  snapshot) to `so101.xml`; delete
  the `cube_tag` site **usage** from the obs path but keep the site itself for
  decision 8.
- **`src/base_env.py`**:
  - `obs_dim_for`/`priv_dim_for` + docstrings: new layout; priv tail adds true
    √M (6) and swaps the cube-tag biases for: live bias (3), precise center
    bias (3), precise rot-perturb (3).
  - Cube-surface sample points (fixed body-frame grid on the box faces, scaled
    by the episode's half extents); per-camera visibility = outward-normal
    facing test + `mj_ray` toward the camera with `bodyexclude=cube_body_id`
    (self-occlusion handled by the facing test, arm/ring by the ray). Delete
    `cube_tag_sample_points/_occluded/_visible` from the obs path (keep the
    minimal GT tag-pose helper for `legacy_tag` distillation).
  - `CamState`/`CamFrame`: replace cube-tag fields with per-camera visible
    fraction + per-camera visible-point centroid; `_process_frame` adds live
    noise (`obs_noise.live_sigma`) and rolls `marker_dropout` for the live
    detection; `_ingest_frame` drives an `ObjectObsState` + the static gate on
    the sim-measured live history, refreshing the precise channel with GT
    center + `box_sqrtm` perturbed by `obs_noise.sqrtm_rot_sigma` (small
    random rotation applied to R before building √M) + per-episode biases.
  - `reset`: broadened `sample_cube_orientation` (all faces unless
    `cube_no_flat_spawns`/`cube_smallest_face_only`); rejection-sample on live
    detectability from both sim cameras. `step`: `info["live_hidden_ratio"]`,
    `info["precise_age_mean"]`.
- **`src/obs_norm.py`**: cube block → live (POS_CENTER/POS_SCALE), ages
  (cap/2), center (POS), √M (diag center = nominal `h/√3`, off-diag 0;
  scale ~0.02/0.01); priv-tail additions with scales from the dr=full knobs.
- **`conf/`**: `dr/*.yaml` — replace `cube_px_noise`/cube-tag comments with
  `live_sigma`, `precise_sigma`, `sqrtm_rot_sigma` + bias keys (placeholder
  values, re-seeded from Stage 3 measurements); `config.yaml` —
  `cube_no_flat_spawns` flag; env yamls reference it.
- **Tests**: new `tests/env/test_shape_obs.py` (√M closed form vs eigh, state
  machine, static gate), `test_cube_channels.py` (visible-surface bias sign,
  occlusion → stale, refresh only when static+visible, both-view requirement),
  rewrite `test_cube_spawn.py` (flat spawns present, crutch flags exclude);
  shift indices in the existing obs-layout tests + `test_obs_norm.py`.

Sim training can start after this stage (crutch flags on) while real-side work
proceeds.

## Stage 2 — dataset + eval harness (~1–1.5 d)

- **`real/marker_spec.py`**: eval-only tag ids for the sponge's other faces
  (20 mm, ROLES-commented as dataset-only; peel after recording).
- **`real/tracking/tag_body_calib.py`** (new): solve each sponge tag's
  in-plane offset + yaw on its (manually declared) face from frames where ≥2
  tags are co-visible, anchored to the box geometry; writes
  `sponge_tags.yaml`. GT body pose then follows from any single visible tag.
- **`real/tracking/record_shapes.py`** (new): both cameras → per-frame raw
  frames + all tag detections + per-camera `T_base_cam` (EMA'd) + timestamps
  into `datasets/sponge_<date>/` with an index file. Masks are **not**
  recorded — computed offline so the dataset stays estimator-agnostic.
  Coverage checklist printed live: 3 resting-face classes × yaw spread ×
  workspace positions; arm occluding ~0/25/50/75% per camera; moving segments
  (hand-carried + in-gripper); settle events. Static/moving auto-labeled from
  tag speed.
- **`real/tracking/eval_estimator.py`** (new): offline SAM pass (sam_seg) →
  candidate estimator → errors vs tag GT: centroid, √M Frobenius,
  principal-axis angle, eigen-spread — sliced by occlusion fraction and
  static/moving, judged against the dr yaml envelope (decision 4).

## Stage 3 — hull estimator (~1 d)

**`real/tracking/hull_shape.py`** (new): masks + intrinsics + `T_base_cam`
pair → silhouette-cone membership test on a voxel grid (workspace box ∩
z ≥ table, ~4 mm pitch) → inside-both voxels → centroid + covariance →
`sqrtm_from_cov`. Multi-frame averaging over the static window happens on M
(linear domain) before the sqrt. Score it with `eval_estimator`; tune nothing
by eye. If acceptance fails, pick the escalation (decision 3) from the
occlusion/component slices — separate plan.

## Stage 4 — real rollout integration (~2 d)

- **`real/rollout/frame_bus.py`** (new): per-camera capture thread with
  fan-out to consumers (one camera can't be opened twice — the marker detector
  and SAM need the same main-cam frames). `CameraMarkerSource` becomes a
  frame-bus consumer for arm tags + table anchor only: `track_cube`,
  `cube_pose()` and their state are **deleted**.
- **`real/rollout/object_obs.py`** (new): `ObjectSource` — SAM3 prompt at
  start (+ re-prompt on K empty frames = re-acquisition, closing that TODO),
  SAM2 tracker per camera on the frame bus, per-frame mask centroid + area →
  triangulated live centroid (`real/vision/stereo.py`), area-vs-static-baseline
  occlusion gate, shared `ObjectObsState` + `is_static`, hull refresh on
  static windows in a worker thread (never blocks the tick), serving
  `object_obs()`. `--gui` overlay: mask tint, live-centroid dot, projected
  √M ellipse per view — the debug view that makes failures visible.
- **`real/rollout/rollout_lift.py`**: camera mode consumes `ObjectSource`;
  fk mode drives the same sim-side channel helpers off the lockstep sim (twin
  contract preserved without a camera). `build_actor_frame` → new layout;
  termination per decision 9; CSV/plot/log columns for both channels + ages;
  viewer draws the live centroid and M-ellipsoid via a `panel/sim_stream`
  helper. `panel/registry.py` updated for changed args/scripts.
- **Contract test**: one recorded measurement sequence pushed through the sim
  helper path and the real `ObjectSource` state path must serve identical
  channel values (the shared state machine makes this near-tautological — the
  test pins that nobody forks it later).

## Stage 5 — retraining (~½ d hands-on + compute)

Attempt A: `python -m src.distill env=lift distill.teacher=<best tag ckpt>
distill.teacher_obs=legacy_tag` → mandatory short PPO fine-tune under
`dr=full` at reduced LR (~3e-5). Attempt B (fallback, or if flat spawns tank
the distilled policy): scratch curriculum — stage 1 `dr=none shaping=none`
with `cube_smallest_face_only=true cube_no_flat_spawns=true
marker_always_visible=true`, then stage the flags off with
shaping light → full, `dr=full` last. User launches long runs; agent-launched
runs use `conda run -n mujoco_env ... > run.log 2>&1` in background.

## Verification

1. `conda run -n mujoco_env pytest tests/env tests/training -v` green at every
   stage boundary.
2. `python -m src.show_starts` — flat + upright spawns, all yaws, none outside
   both-camera visibility.
3. Smoke train (`env=lift train.total_timesteps=20000 wandb.enabled=false`)
   under `dr=none` and `dr=full`: obs dim matches the new `obs_dim_for`,
   `live_hidden_ratio` near 0 early / rising near grasp, precise age growing
   during approach.
4. `eval_estimator` on the dataset: acceptance criterion green on static
   windows before any real rollout.
5. Dry-run `python -m real.rollout.rollout_lift --marker-source camera` (no
   `--execute`): hand-move the sponge → live tracks within ~1 cm; cover it →
   both ages grow, precise holds; let it settle → precise refreshes; `--gui`
   ellipse hugs the sponge in both views.

## Docs & follow-ups (same change set)

- `CLAUDE.md`: obs layout line, twins list (cube logic now single-sourced in
  `src/shape_obs.py`; arm markers unchanged), sponge tracked via SAM stereo,
  tag id 1 = eval-only.
- `TODO.md`: add — mono live fallback (with dataset both-view availability
  number); estimator escalation criteria; pickplace `ObjectSource` rollout;
  delete `cube_tag` site + GT helper + `legacy_tag` when the legacy teacher
  retires; camera-sync note downgraded to footnote (static-regime design);
  revisit pose-locked cube noise item (largely superseded by the two-regime
  obs — the precise channel *is* pose-locked now).
