# Dense-stereo BPS observation for the sponge

## Status and motivation

The tag-free v1 estimator (`.claude/plans/shape_tensor_tracking.md`) uses the
intersection of two SAM silhouette cones to produce a center and shape tensor
√M. The full recording `datasets/sponge_20260808_145110` rejected that
estimator:

- 82 usable static windows;
- center error 12.1 mm p95 (inside the old 15.1 mm envelope);
- √M eigenvalue error 10.3 mm p95 versus 2.9 mm allowed;
- mean principal-axis disagreement 51°;
- live static jitter remained good at 0.9 mm p95.

The live MuJoCo comparison confirms that this is a real estimator failure, not
just an awkward scalar metric: the visual hull invents volume along directions
the two silhouettes do not constrain. Even the ideal green √M ellipsoid is an
indirect and sometimes confusing representation of the physical sponge.

Replace the v1 **precise** center/√M channel with a dense-stereo visible-surface
point cloud reduced to a fixed Basis Point Set (BPS) observation. Keep the fast
SAM mask-centroid channel for motion. The observation layout changes, so old
checkpoints are not expected to load directly.

The AprilTags on the sponge remain evaluation-only ground truth. Neither the
dense stereo pipeline, BPS observation nor deployed policy may depend on them.

## Decisions

1. **Keep two temporal regimes.** SAM2 mask-centroid triangulation remains the
   fast live channel. Dense stereo runs only after the shared static gate opens;
   its BPS observation holds and ages while the sponge moves, is grasped or
   is occluded. Free-running C922 skew is therefore charged to the live error
   budget and does not corrupt the precise cloud.

2. **Use one fixed calibration for the rigid stereo mount.** A stationary shared
   checkerboard determines `T_aux_main` and the rectification maps. The existing
   table-tag anchors still place measurements in the arm base frame, but never
   modify the relative stereo calibration. If the startup anchor comparison
   indicates that the rig moved, withhold dense measurements and require a new
   checkerboard calibration.

3. **Use the accepted OpenCV StereoSGBM configuration.** Its parameters are
   frozen perception, never part of PPO. Run matching on the full rectified
   frames and apply SAM masks afterward rather than blacking out correspondence
   context.

4. **The precise measurement is the visible surface, not a fabricated full
   object.** Dense disparity is converted to metric XYZ, filtered by both SAM
   masks and stereo consistency, transformed into the base frame, then reduced
   by voxel downsampling before BPS reduction. Hidden surfaces remain unknown.
   Training sees the same partial-surface convention and learns what is
   task-relevant for the one physical sponge family.

5. **Use a fixed 64-point base-aligned BPS observation.** Center the filtered
   cloud on its measured centroid. Form the basis as the Cartesian product of
   `[-0.04, -0.01, 0.01, 0.04] m` on the base x/y/z axes, and encode the
   distance from each ordered basis point to the nearest visible point. The
   denser samples near the center resolve the sponge surface while the outer
   samples cover centroid bias and unusual poses.

6. **The BPS block is not repeated through history taps.** The observation
   contains the existing low-dimensional state history, one current/held BPS
   block with metadata, and the privileged tail. The asymmetric critic still
   receives privileged true object state.

7. **Evaluation must remove the tags' visual shortcut.** The 20 mm black/white
    tags add excellent stereo texture and would make a quantitative tagged run
    over-optimistic relative to a tag-free sponge. Report raw and digitally
    tag-inpainted stereo results separately; acceptance uses the inpainted
    frames. Also record a short tag-free qualitative run after quantitative
    validation and compare valid-point/noise distributions.

## Target real perception pipeline

For each scheduled static capture:

1. Read the newest main/aux C922 frames.
2. Undistort and rectify both frames with the current accepted stereo calibration.
3. Rectify each SAM mask with nearest-neighbor interpolation.
4. Run the frozen StereoSGBM configuration on the **full rectified pair**,
   producing left-image disparity.
5. Convert disparity to metric rectified-left-camera XYZ through OpenCV's `Q`
   matrix; do not hand-derive signs/principal-point corrections.
6. Keep points whose left pixel is in the left SAM mask and whose reprojection
   lands in the right SAM mask.
7. Reject invalid/negative disparity, left-right-inconsistent or low-confidence
   points, points outside the workspace, points below the table and statistical
   depth outliers.
8. Transform retained points from rectified-left coordinates into the arm base
   using the current EMA'd rig anchor.
9. Voxel-downsample the retained cloud, compute its centroid, and reduce all
   retained points through the frozen BPS transform.
10. Compute BPS metadata, update the held precise observation and reset its age.
    A miss holds the previous observation and lets age grow.

The initial model input target is 640×384: half-width processing with vertical
crop/padding to a multiple of 32. Final dimensions and `max_disp` are derived
from the calibrated rectified focal length, baseline, valid ROI and nearest
workspace depth. For the target geometry, `f≈480 px`, `B=0.12 m`, `Z=0.4 m`
implies approximately 144 px disparity and 2.8 mm depth change per one-pixel
disparity change. Do not copy those estimates into runtime constants.

### BPS contract

Let `p_j` be each retained base-frame point minus the measured cloud centroid.
Define the 64 basis points once:

```text
basis_axis_m = [-0.04, -0.01, 0.01, 0.04]
basis = CartesianProduct(basis_axis_m, basis_axis_m, basis_axis_m)
bps_distance[i] = clip(min_j ||basis[i] - p_j|| / 0.08, 0, 1)
```

Order the Cartesian product lexicographically by `(x, y, z)` and preserve that
order as a 64-value observation vector. `conf/config.yaml` owns the axis values
and 0.08 m distance cap. The policy checkpoint stores a fingerprint of those
resolved BPS values; a mismatch is a load error. The transform uses every
retained voxel-downsampled point and has no stochastic sampling step. Across
the 265 accepted cached clouds, the largest raw basis-to-cloud distance was
58.4 mm, so the 80 mm cap leaves margin without compressing normal measurements.

Metadata supplied outside the BPS distance vector:

```text
cloud_center_base (3)
cloud_age_s       (1)
valid_fraction    (1)
live_center_base  (3)
live_age_s        (1)
```

Centering separates global location from visible shape. `valid_fraction` is the
fraction of left-mask pixels that survive correspondence and cloud filtering.
The never-measured state is an all-zero BPS block with zero valid fraction and
age at the cap. Exact observation dimensions are computed from the observation
specification, never duplicated as literals.

## Stage 1 — stereo calibration

### Calibration tooling

- Add an annotated rectification viewer with horizontal guide lines and
  measured vertical correspondence residual. Refuse dense inference when the
  calibration image size/focus does not match.
- Compare the two cameras' startup anchor estimates with the calibration's
  known-good reference. If the movement threshold is exceeded, fail loudly and
  instruct the operator to re-run the stationary checkerboard calibration.

### Calibration acceptance

- Shared checkerboard reprojection RMS below 1 px per camera.
- Rectified vertical correspondence p95 below 1 px throughout the workspace.
- The two-tag board's per-camera rig-pose estimates agree within calibrated
  gates; log disagreement rather than letting it perturb rectification.
- No workspace pose falls outside either valid rectified ROI.

## Stage 2 — offline dense-stereo feasibility gate

Do not modify the policy yet.

1. Record a short 1–2 minute tagged dataset at the new camera geometry:
   representative resting faces/yaws/regions, long static holds, slow movement,
   arm occlusion and full disappearance/reappearance.
2. Cache rectified raw RGB, tag-inpainted RGB and masks under a
   calibration-specific dataset path. Cache disparities and clouds beneath
   StereoSGBM-configuration-specific keys. A partial run resumes and reports
   progress/rate/ETA.
3. Split by physical placement/session into a parameter-development set and a
   held-out acceptance set. No matcher or filter tuning may use held-out frames.
4. Run OpenCV StereoSGBM on the tag-inpainted development frames. Derive
   `minDisparity` and `numDisparities` from the calibrated valid ROI and
   workspace depth bounds, then search a small, recorded grid over block size,
   smoothness penalties, uniqueness threshold, left-right consistency and
   speckle filtering. Freeze the chosen parameters before held-out evaluation.
5. Convert disparity through the filtering and base-frame point pipeline, then
   save the per-frame clouds. Derive confidence/validity deterministically from
   uniqueness and left-right residual, with calibration fitted on development
   frames only.
6. Add a synchronized viewer: camera pair + disparity/confidence + MuJoCo GT
   box + actual point cloud. Do not summarize the cloud as an ellipsoid in the
   primary diagnostic.
7. Evaluate the frozen configuration on held-out tag-inpainted frames; this is
   the quantitative acceptance path. Also run it on raw-tag frames and report
   the difference to measure the visual shortcut.
8. Benchmark end-to-end StereoSGBM CPU latency on the deployment machine.

### Dense-depth metrics

- Point-to-tag-GT cuboid surface signed error: median, RMS and p95.
- Bias and scatter decomposed along the rectified camera depth axis.
- Valid points and covered surface area by face/yaw/workspace/occlusion slice.
- Static temporal point-to-surface jitter.
- Fraction of left-mask pixels rejected by right-mask/correspondence checks.
- Catastrophic outlier rate outside the GT cuboid by >2 cm.
- Full latency and CPU load: rectification, stereo, filtering, voxel
  downsampling and BPS reduction.
- Disappearance/reacquisition behavior and held-cloud age.

Initial feasibility gates, fixed before BPS tuning:

- tag-inpainted point-to-surface error <=3 mm median and <=8 mm p95;
- catastrophic outliers <1% after filtering;
- at least 128 valid points on >=95% of static, unoccluded pairs;
- no systematic face/yaw slice with <64 valid points;
- StereoSGBM static refresh <=100 ms on the deployment machine;
- raw-tag performance is reported but never used to pass a gate.

StereoSGBM must pass every geometric slice before its cloud reaches the BPS
transform; the fixed observation cannot recover trustworthy geometry from a
failing stereo source.

### Stage 2 result — complete

The original 3,522-pair dataset
`datasets/sponge_20260808_203620` was relabelled from its saved raw tag poses
after suppressing pose-estimator jitter with a 150 ms causal mean. Sixteen of
its 26 static placement windows are inside the policy workspace; 12 were used
for development and four whole placements were held out.

Freeze OpenCV StereoSGBM with block size 7, uniqueness ratio 15, 150-pixel
speckle window, speckle range 2 and left/right maximum difference 2. On 144
held-out tag-inpainted frames it produced 1.35 mm median / 5.12 mm p95 cuboid
surface error, zero catastrophic outliers and at least 627 filtered points in
every frame. Median camera-depth bias/scatter were 0.09/1.40 mm, median static
point-to-surface jitter was 0.25 mm, and the median visible surface coverage
was 23.0 cm². Cached rectified-pair through filtered-voxel-cloud time was
17.8 ms median / 20.3 ms p95. Raw tags improve the numbers only slightly. The
tag-inpainted acceptance path passes every fixed gate, so this StereoSGBM
configuration is frozen for deployment.

`real.tracking.view_dense_stereo` provides the synchronized rectified pair,
disparity, left/right confidence, orthographic cloud/GT views and optional 3D
MuJoCo cloud viewer. The current recording has no disappearance after initial
acquisition, so it cannot measure true reacquisition delay. A broader
supplementary capture and its disappearance/reacquisition check are deferred.
The full capture-to-cloud latency benchmark is also intentionally deferred:
precise refresh is static-gated, and the measured matcher/filter path is far
inside the 100 ms budget.

References:

- OpenCV StereoSGBM:
  <https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html>
- OpenCV stereo calibration/rectification:
  <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- Basis Point Sets: <https://arxiv.org/abs/1908.09186>

## Stage 3 — BPS contract validation

### Synthetic cloud generator

Build on the existing simulated camera visibility/surface sampling rather than
rendering RGB/depth for every vectorized environment:

- sample points on the randomized sponge's physical faces;
- retain only points visible to both calibrated sim cameras;
- raycast arm/ring occlusion with the existing geom-group convention;
- apply configurable point noise and random point dropout;
- simulate whole-view loss and held-cloud age;
- voxel-downsample and encode through the exact real BPS transform.

This generator becomes the single source used by the sim env and offline BPS
validation. Contract tests push the same synthetic measurement through both
paths and require byte-equal BPS observations and metadata.

### Fixed-basis validation

Run the exact 64-point transform over synthetic measurements and every accepted
cached real cloud. Confirm that outputs are finite, ordinary measurements do
not clip, and sub-millimetre point jitter causes small output changes.

Keep the StereoSGBM, filter and BPS parameters frozen during policy training.
Store and validate the resolved BPS fingerprint with each policy checkpoint.

### BPS acceptance

- the generated basis is exactly the lexicographically ordered Cartesian
  product of the four configured axis values and produces 64 distances;
- point-order invariance tests pass exactly;
- sim and real transforms produce byte-equal output for the same retained cloud;
- every valid input produces finite distances in `[0, 1]`;
- no accepted static, unoccluded real cloud reaches the 80 mm clipping cap;
- the never-measured state is an all-zero distance vector with zero valid
  fraction and age at the cap;
- distances change smoothly under sub-millimetre point jitter and degrade
  predictably as valid points are removed.

### Stage 3 result — complete

The fixed transform lives in `src/bps.py`: the configured axis values expand
to the lexicographically ordered 64-point basis, clouds are canonically sorted
before centroid and nearest-distance reduction, and the normalized float32
result is therefore byte-identical under point permutations. The same module
owns voxel selection, the never-measured/held-age state, and the SHA-256
fingerprint over the resolved ordered basis and 80 mm cap. The fingerprint is
`3257e6752899edceb76df6d70ed39bbda7488961d2e0d9b0755edcbc6002963c`.

`src/sim_bps.py` samples the randomized physical box faces, intersects
visibility through both calibrated MuJoCo cameras, uses the existing geom-group
raycast convention for arm/ring occlusion, applies configurable point noise,
dropout and whole-view loss, and then uses the shared real voxel/BPS path.
Contract tests pass the same retained cloud through the sim generator and
`real.tracking.dense_stereo.cloud_to_bps` and require byte-equal distances,
center and metadata.

`python -m real.tracking.eval_bps --dataset
datasets/sponge_20260808_203620` validated all 265 accepted tag-inpainted Stage
2 clouds (627–1,303 points per cloud). Every output was finite and in range,
point-order invariance was exact, and the largest raw basis distance was 58.384
mm, leaving 21.616 mm below the 80 mm clipping cap. Bounded 0.5 mm point jitter
changed a normalized distance by at most 0.006215, inside the analytic 0.0125
centering-aware Lipschitz bound. Every Stage 3 gate passes; the machine-readable
report is cached beside the Stage 2 report as `bps_report.yaml`.

## Stage 4 — observation and policy integration

- Extend the existing flat observation specification to
  `[state history | current BPS block | privileged tail]`:
  - `state history`: qpos/qvel, arm markers/ages, live channel/age, task extras
    and previous actions through the existing history convention;
  - `current BPS block`: ordered distances, measured cloud center, precise age
    and valid fraction, included exactly once;
  - `privileged tail`: true object state kept inaccessible to the actor.
- Feed the actor-visible state history and current BPS block directly to the
  configured policy MLP.
- Update `obs_dim_for`, normalization and checkpoint metadata so the complete
  layout has one source of truth.
- Extend distillation and rollout loading explicitly; no checkpoint surgery.
- Delete old √M actor normalization/bias knobs only after no supported policy
  consumes them. Preserve migration support only where an actual teacher needs
  it.

Tests:

- exact 64-point basis coordinates, lexicographic ordering and normalization;
- point-order invariance of the BPS transform;
- distance clipping at the configured 80 mm cap;
- all-invalid never-seen behavior;
- sim/real BPS twin contract;
- actor cannot read privileged tail;
- BPS block included once, not copied per history tap;
- checkpoint save/load preserves basis fingerprint and normalization;
- distillation updates the actor under the new observation layout;
- vectorized env throughput benchmark catches accidental per-step rendering.

### Stage 4 result — complete

The cube-policy layout is now `[state history | current BPS block | privileged
tail]`. `src/base_env.py` computes the state, BPS and privileged widths from
one layout contract; `ObsHistory` receives only the low-dimensional state, so
the 64 distances, cloud center, age and valid fraction occur exactly once.
The sim captures partial surface clouds only on scheduled camera snapshots,
passes them through the shared voxel/BPS transform and publishes them through
`BPSObsState` under the shared live/static gate.

Normalization covers the bounded BPS values directly. PPO checkpoints persist
the resolved BPS fingerprint and training resume, eval, distillation and real
rollout loading reject a missing or mismatched fingerprint. Distillation
supports BPS-identical/current/privileged teachers plus the one intentional
pre-BPS bridge, `legacy_tag`; no rejected precise-observation checkpoint is a
supported teacher.

The rejected shape-tensor actor, critic-tail fields, bias configuration,
normalization, online source, offline evaluator and visualization code were
deleted. Generic dataset/mask helpers used by accepted dense stereo moved to
`real.tracking.shape_dataset`. Until Stage 5 supplies the complete asynchronous
dense worker, real camera mode fails explicitly; the FK rollout already builds
the new state-history-plus-BPS observation and validates its checkpoint.

## Stage 5 — real rollout integration

- Add one dense-stereo worker behind the existing `FrameBus`; cameras remain
  single-owner resources.
- SAM2/live tracking stays at camera rate. Static-gated stereo work consumes the
  newest eligible pair and never blocks `ArmLoop`.
- The worker publishes immutable BPS measurements to the shared held/age state
  machine and retains the filtered cloud only for diagnostics and visualization.
- Track inference time, cloud age, valid count and rejected fractions in rollout
  logs.
- Compare complete two-tag camera poses with the calibration's known-good
  anchor reference at startup. If either movement threshold is exceeded, print
  the checkerboard recalibration instruction and withhold dense measurements.
- A dense-worker failure is loud and aborts camera-mode startup; transient
  empty/invalid clouds are measurements misses and use hold/age normally.
- Register every new script/argument/resource in `panel/registry.py`.

GPU scheduling:

- SAM2 runs every camera frame.
- SAM3 remains GPU-resident for the entire rollout. Do not offload it, reload it
  between acquisitions or fall back to CPU.
- Dense stereo runs only for static refreshes using the frozen CPU StereoSGBM
  implementation evaluated in Stage 2.
- The PPO policy remains on CPU unless end-to-end profiling proves a shared GPU
  path is simpler and faster.

### Stage 5 implementation result — complete and live-validated

`real.rollout.object_obs.ObjectSource` now consumes both camera feeds without
opening either device. One background loop keeps SAM3 resident, runs the two
SAM2 trackers, triangulates the live centroid, denoises only the static-gate
point and replaces the pending dense job with the newest eligible immutable
pair. A second CPU loop runs the frozen sixth StereoSGBM candidate, shared
rectification/filter/voxel/BPS path and publishes through `BPSObsState`; misses
hold and age, while worker failures re-raise on the rollout thread. The latest
filtered cloud remains available for diagnostics.

Camera mode in `real.rollout.rollout_lift` now combines the existing arm-marker
source with this object source, mirrors measured channels in the viewer, and
logs cloud age, valid count/fraction, correspondence/overall rejection,
refresh/miss counts, SAM/dense latency and startup rig movement. The panel
registers both the mode's arguments/resources and a camera-only
`python -m real.rollout.object_obs --seconds 5` smoke test. The software and
sim/real contract suite passes 494 tests.

After refreshing the stereo calibration, table-anchor reference and both sim
camera snapshots, the five-second live smoke passed the rig gate at 0.09 mm /
0.009 degrees and found the sponge in both views at SAM3 score 0.92. It
published 86 dense refreshes with zero misses: clouds retained 1,016–1,088
voxel points, validity was 90.7–95.0 percent, and dense latency was 24.0–30.5
ms. The live centroid remained near `(0.141, -0.054, 0.032)` m.

The available paper checkerboard was visibly bowed, so this placement snapshot
was accepted under temporary 1.25 px reprojection / 3 px rectified-p95 gates;
its measured values were 1.088/0.926 px and 2.665 px. `TODO.md` retains the
action to restore both 1 px Stage 1 gates after mounting the target to a rigid
planar backing.

## Stage 6 — policy training

1. Distill from the best available privileged/tag teacher using synthetic
   partial clouds. The teacher receives its old/GT-compatible observation;
   the student receives the frozen BPS observation.
2. PPO fine-tune under `dr=none` until control succeeds.
3. Continue curriculum through `dr=light` and `dr=full`, enabling point noise,
   dropout and whole-cloud loss in stages.

Training acceptance:

- synthetic success matches or exceeds the current lift baseline across all
  resting faces;
- performance degrades gracefully under point dropout and full-cloud holds;
- replacing the BPS distances with the never-measured value materially reduces
  performance, demonstrating that the policy uses the precise geometry;
- short real dry-runs show sensible actions while cloud age grows;
- real lift evaluation starts slow and non-executing, then follows the existing
  hardware safety checklist.

## Failure handling

- If rectification is unstable, fix the shared stereo calibration before
  changing models.
- If correspondence fails on the textureless/tag-inpainted sponge, test
  lighting and non-semantic removable texture first; permanent texture is only
  acceptable if it is part of the deployed object specification.
- If StereoSGBM no longer provides valid shared-surface depth after a camera or
  calibration change, fix the geometry/texture regression or move to an
  active-depth camera before training on bad geometry.
- If BPS contract validation fails, fix the transform, normalization or
  point-noise/dropout model before policy training.

## Definition of done

- The fixed stereo calibration passes its reprojection, rectification and
  workspace-coverage gates, and startup detects rig movement.
- The tag-inpainted StereoSGBM cloud passes every held-out offline geometry gate
  and pose/coverage slice.
- The frozen StereoSGBM configuration meets the static refresh budget.
- The fixed 64-distance BPS contract passes clipping and jitter tests.
- Sim and real serve the identical BPS contract.
- Distilled + PPO-fine-tuned policy succeeds under the configured point-cloud
  DR.
- Real diagnostics show tag GT only in evaluation mode.
- The real rollout consumes no sponge tags and handles cloud loss through the
  shared hold/age convention.
- Old visual-hull actor code/config is removed or explicitly retained only for
  reproducible legacy evaluation, with `TODO.md` updated accordingly.
