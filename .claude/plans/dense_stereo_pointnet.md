# Dense-stereo point-cloud observation for the sponge

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
point cloud encoded by a small, supervised-pretrained PointNet. Keep the fast
SAM mask-centroid channel for motion. The policy is allowed a full network and
observation redesign; old checkpoints are not expected to load directly.

The AprilTags on the sponge remain evaluation-only ground truth. Neither the
dense stereo model, PointNet nor deployed policy may depend on them.

## Decisions

1. **Keep two temporal regimes.** SAM2 mask-centroid triangulation remains the
   fast live channel. Dense stereo runs only after the shared static gate opens;
   its point cloud/latent holds and ages while the sponge moves, is grasped or
   is occluded. Free-running C922 skew is therefore charged to the live error
   budget and does not corrupt the precise cloud.

2. **Calibrate the independently mounted cameras at their current placement.**
   Individual intrinsics stay fixed, but either stand moving invalidates
   `T_aux_main` and its rectification maps. A stationary shared checkerboard
   snapshots that relative geometry. During episodes, AprilTags 10 and 11 keep
   one EMA'd base pose per camera; only simultaneous valid board pairs may
   update relative geometry, and an incomplete/rejected observation holds it.
   Rectification updates happen only while static and must pass the vertical
   correspondence gate before dense inference resumes.

3. **Select the simplest stereo matcher that passes geometric gates.** Start
   with OpenCV StereoSGBM as the classical baseline and Fast-FoundationStereo
   as the primary neural candidate. Run original FoundationStereo only when
   neither primary candidate passes or a targeted accuracy-ceiling diagnosis
   is needed.
   Prefer StereoSGBM if it passes; otherwise deploy Fast-FoundationStereo,
   initially in PyTorch and only then through fixed-shape ONNX/TensorRT if
   profiling justifies it. Matcher parameters/weights are frozen perception,
   never part of PPO. Run matching on the full rectified frames and apply SAM
   masks afterward rather than blacking out correspondence context.

4. **The precise measurement is the visible surface, not a fabricated full
   object.** Dense disparity is converted to metric XYZ, filtered by both SAM
   masks and stereo consistency, transformed into the base frame, then reduced
   to a fixed 256-point sample. Hidden surfaces remain unknown. Training sees
   the same partial-surface convention and learns what is task-relevant for the
   one physical sponge family.

5. **Do not feed thousands of pixels or a flat point vector to PPO.** A small
   PointNet-style encoder maps the unordered 256-point set to a 32-dimensional
   feature. The encoder is pretrained with supervised synthetic geometry and
   fine-tuned against tag GT before RL; it is frozen for distillation and the
   first PPO stage. PPO learns control from stable geometric features, not
   correspondence or permutation invariance from sparse reward.

6. **Do not use a PointNet spatial transformer.** Point coordinates are in the
   arm base frame and their orientation is task information. Learned
   canonicalization would erase exactly the signal the policy needs.

7. **The cloud is not repeated through history taps.** Redesign the policy
   input as structured branches: the existing low-dimensional state history,
   one current/held cloud, and cloud metadata. The cloud branch is encoded once
   per policy tick, then concatenated with the vector branch. The asymmetric
   critic still receives privileged true object state.

8. **Try explicit known-sponge fitting before committing the policy to a
   latent-only interface.** Dense visible points plus both silhouettes may fit
   the known 6×4×2.5 cm cuboid accurately enough to retain a compact explicit
   center/symmetry observation. This is a cheap branch of the same feasibility
   work. If it passes held-out tag GT, prefer its inspectability; otherwise use
   the PointNet latent described here.

9. **Evaluation must remove the tags' visual shortcut.** The 20 mm black/white
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
4. Run the selected StereoSGBM/Fast-FoundationStereo backend on the **full
   rectified pair**, producing left-image disparity.
5. Convert disparity to metric rectified-left-camera XYZ through OpenCV's `Q`
   matrix; do not hand-derive signs/principal-point corrections.
6. Keep points whose left pixel is in the left SAM mask and whose reprojection
   lands in the right SAM mask.
7. Reject invalid/negative disparity, left-right-inconsistent or low-confidence
   points, points outside the workspace, points below the table and statistical
   depth outliers.
8. Transform retained points from rectified-left coordinates into the arm base
   using the current EMA'd rig anchor.
9. Voxel-downsample, then choose/pad exactly 256 points. Preserve a validity
   bit and confidence per row.
10. Compute cloud metadata, update the held precise observation and reset its
    age. A miss holds the previous cloud and lets age grow.

The initial model input target is 640×384: half-width processing with vertical
crop/padding to a multiple of 32. Final dimensions and `max_disp` are derived
from the calibrated rectified focal length, baseline, valid ROI and nearest
workspace depth. For the target geometry, `f≈480 px`, `B=0.12 m`, `Z=0.4 m`
implies approximately 144 px disparity and 2.8 mm depth change per one-pixel
disparity change. Do not copy those estimates into runtime constants.

### Point tensor contract

One held cloud is a `(256, 5)` float tensor:

```text
[dx, dy, dz, confidence, valid]
```

- `dx/dy/dz`: base-frame point minus the measured cloud centroid, metres;
- `confidence`: normalized stereo validity/confidence in `[0, 1]`;
- `valid`: exactly 0 or 1; padded rows are all zero.

Metadata supplied outside the point set:

```text
cloud_center_base (3)
cloud_age_s       (1)
valid_fraction    (1)
live_center_base  (3)
live_age_s        (1)
```

Centering separates global location from visible shape. Points use a fixed
physical normalization scale based on the workspace/sponge dimensions, saved
with the policy exactly like the current `ObsNorm` constants. Sampling is
deterministic during evaluation and randomized during synthetic pretraining.

### PointNet v1

```text
valid point [dx,dy,dz,confidence]
    shared MLP 4 -> 32 -> 64 (LayerNorm + GELU)
masked max pool (64) || masked mean pool (64)
    projection 128 -> 64 -> 32 (LayerNorm + GELU)
cloud feature (32)
```

Invalid rows must be excluded from both pools, not merely represented by zero.
The all-invalid case is the explicit never-measured state and serves a zero
feature plus age at the cap. Unit tests pin permutation invariance, padding
invariance and finite gradients.

The actor concatenates the 32 cloud features and 5 metadata values with the
existing vector/history branch. The critic additionally sees true sponge
center, orientation/symmetry, velocity, contact state and physical dimensions
through the privileged tail. Exact final dimensions are computed from the
structured observation specification, never duplicated as literals.

## Stage 1 — stereo calibration

### Calibration tooling

- Add an annotated rectification viewer with horizontal guide lines and
  measured vertical correspondence residual. Refuse dense inference when the
  calibration image size/focus does not match.
- Extend the two-tag camera EMAs to regenerate relative rectification after a
  stand moves, accepting updates only from synchronized complete-board poses
  that pass the same vertical-residual gate. Until then, re-run the stationary
  checkerboard snapshot after moving either camera.

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
   backend/configuration-specific keys. A partial run resumes and reports
   progress/rate/ETA.
3. Split by physical placement/session into a parameter-development set and a
   held-out acceptance set. No matcher or filter tuning may use held-out frames.
4. Run OpenCV StereoSGBM first on the tag-inpainted development frames. Derive
   `minDisparity` and `numDisparities` from the calibrated valid ROI and
   workspace depth bounds, then search a small, recorded grid over block size,
   smoothness penalties, uniqueness threshold, left-right consistency and
   speckle filtering. Freeze the chosen parameters before held-out evaluation.
5. Run Fast-FoundationStereo on the same tag-inpainted development frames.
   Compare a small, recorded set of official checkpoint/refinement trade-offs;
   freeze the choice before held-out evaluation. Start with its official
   environment in an isolated process because its pinned PyTorch stack may
   conflict with `mujoco_env`.
6. Convert every backend's disparity through the identical filtering and
   base-frame point pipeline, then save per-frame clouds under backend-specific
   cache keys. Define confidence/validity explicitly per backend: use native
   confidence only when exposed, otherwise derive it deterministically from
   uniqueness and left-right residual, with calibration fitted on development
   frames only.
7. Add a synchronized viewer: camera pair + disparity/confidence + MuJoCo GT
   box + actual point cloud. Do not summarize the cloud as an ellipsoid in the
   primary diagnostic.
8. Evaluate the frozen configurations on held-out tag-inpainted frames; this
   is the quantitative acceptance path. Also run the frozen configurations on
   raw-tag frames and report the difference to measure the visual shortcut.
9. If neither primary candidate passes every gate, run original
   FoundationStereo on roughly 100 representative failing and passing pairs as
   a diagnostic accuracy ceiling. It is not a prerequisite when either primary
   candidate already passes cleanly.
10. Select the simplest backend that passes every held-out slice. Benchmark
    end-to-end CPU latency for StereoSGBM or PyTorch GPU latency for Fast. Only
    benchmark fixed-input ONNX/TensorRT when Fast is selected and PyTorch does
    not already satisfy the refresh budget or deployment profiling warrants it.

### Dense-depth metrics

- Point-to-tag-GT cuboid surface signed error: median, RMS and p95.
- Bias and scatter decomposed along the rectified camera depth axis.
- Valid points and covered surface area by face/yaw/workspace/occlusion slice.
- Static temporal point-to-surface jitter.
- Fraction of left-mask pixels rejected by right-mask/correspondence checks.
- Catastrophic outlier rate outside the GT cuboid by >2 cm.
- Full latency, CPU load and GPU memory as applicable: rectification, stereo,
  filtering and sampling.
- Disappearance/reacquisition behavior and held-cloud age.

Initial feasibility gates, fixed before model tuning:

- tag-inpainted point-to-surface error <=3 mm median and <=8 mm p95;
- catastrophic outliers <1% after filtering;
- at least 128 valid points on >=95% of static, unoccluded pairs;
- no systematic face/yaw slice with <64 valid points;
- selected backend static refresh <=100 ms on the deployment machine;
- raw-tag performance is reported but never used to pass a gate.

Backend selection is gate-based: prefer StereoSGBM when it passes all geometric
slices, otherwise use Fast-FoundationStereo when it passes. If both fail and
original FoundationStereo also fails as the accuracy ceiling, stop: PointNet
cannot recover trustworthy geometry. Revisit baseline, rectification,
lighting/texture, or an active-depth camera. If original passes while Fast
fails, treat that as a matcher compression/runtime gap and decide explicitly
whether the slower model, a different Fast checkpoint or another optimized
matcher is acceptable.

References:

- FoundationStereo: <https://github.com/NVlabs/FoundationStereo>
- Fast-FoundationStereo: <https://github.com/NVlabs/Fast-FoundationStereo>
- OpenCV StereoSGBM:
  <https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html>
- OpenCV stereo calibration/rectification:
  <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- PointNet: <https://arxiv.org/abs/1612.00593>

## Stage 3 — explicit fitting branch

Before changing the policy observation, fit the known sponge cuboid to each
filtered cloud plus both SAM silhouettes:

- optimize body center and symmetry-aware orientation;
- keep physical dimensions fixed initially, then test bounded dimension
  fitting only if needed;
- use robust point-to-box-surface loss plus silhouette reprojection loss;
- enforce table support while static;
- initialize from the cloud centroid and visible plane normals;
- evaluate against held-out tag GT with the same pose/coverage slices.

If this produces <=10 mm center p95 and <=15° long-axis symmetry error p95
without pose-specific failures, retain an explicit low-dimensional observation
and use the dense cloud only inside perception. Still train with measured fit
noise/holds. If it fails or throws away useful contact geometry, continue with
the PointNet observation.

## Stage 4 — PointNet supervised pretraining

### Synthetic cloud generator

Build on the existing simulated camera visibility/surface sampling rather than
rendering RGB/depth for every vectorized environment:

- sample points on the randomized sponge's physical faces;
- retain only points visible to both calibrated sim cameras;
- raycast arm/ring occlusion with the existing geom-group convention;
- model rectified depth noise, quantization, holes, foreground-edge outliers,
  mask erosion/dilation, confidence and valid-count distributions measured in
  Stage 2;
- simulate whole-view loss and held-cloud age;
- voxel/sample/pad through the exact real cloud reducer.

This generator becomes the single source used by PointNet pretraining and the
sim env. Contract tests push the same synthetic measurement through both paths
and require byte-equal point tensors/metadata.

### Supervision

Pretrain the encoder with lightweight heads for:

- visible-cloud-centroid to true-body-center residual (3);
- symmetry-aware true √M (6), used as supervision rather than the deployed
  observation contract;
- resting-face class (3-way);
- long-axis direction under sign symmetry;
- optional fixed-size full-surface completion with Chamfer loss if the above
  heads leave the latent insensitive to missing backside geometry.

Train broadly in simulation, then fine-tune these heads on the new tagged real
cloud dataset, using the tag-inpainted stereo input. Acceptance is held-out by
physical placement/session, not random frames from the same static window.

Freeze the selected stereo backend permanently: neural weights or classical
parameters must not change during policy training. Freeze the PointNet encoder
after real fine-tuning for policy distillation and initial PPO. Save its
weights, normalization, point contract, stereo-backend configuration and
calibration fingerprint inside the policy checkpoint or one atomic deployment
artifact; never load mismatched pieces silently.

### Encoder acceptance

- permutation and padding invariance tests pass exactly;
- held-out real center error <=10 mm p95;
- held-out resting-face accuracy >=95%;
- held-out long-axis symmetry error <=15° p95;
- no face/yaw/workspace slice is missing from the report;
- latent changes smoothly under sub-millimetre point jitter and degrades
  predictably as valid points are removed.

## Stage 5 — structured observation and network integration

- Replace the flat-only actor observation with a `gymnasium.spaces.Dict` (or an
  equivalently explicit typed structure) containing:
  - `state_history`: qpos/qvel, arm markers/ages, live channel/age, task extras
    and previous actions through the existing history convention;
  - `cloud`: one held `(256,5)` precise point tensor;
  - `cloud_meta`: center, precise age and valid fraction;
  - critic-only privileged object state kept structurally inaccessible to the
    actor.
- Add a custom SB3 feature extractor with separate vector and PointNet
  branches. Concatenate their features before the actor MLP.
- Keep LayerNorm + GELU. Do not use BatchNorm: online batches and changing
  visibility regimes make running statistics an avoidable train/deploy state.
- Update `obs_dim_for`/normalization/checkpoint metadata so the structured
  contract has one source of truth.
- Extend distillation and rollout loading explicitly; no checkpoint surgery.
- Delete old √M actor normalization/bias knobs only after no supported policy
  consumes them. Preserve migration support only where an actual teacher needs
  it.

Tests:

- point permutation and padded-row invariance;
- all-invalid never-seen behavior;
- sim/real point reducer twin contract;
- actor cannot read privileged tail;
- cloud encoded once, not copied per history tap;
- checkpoint save/load preserves encoder and normalization;
- distillation updates the actor while a frozen PointNet remains unchanged;
- vectorized env throughput benchmark catches accidental per-step rendering.

## Stage 6 — real rollout integration

- Add one dense-stereo worker behind the existing `FrameBus`; cameras remain
  single-owner resources.
- SAM2/live tracking stays at camera rate. Static-gated stereo work consumes the
  newest eligible pair and never blocks `ArmLoop`.
- The worker publishes immutable sampled-cloud measurements to the shared
  held/age state machine.
- Track inference time, cloud age, valid count, rejected fractions and stereo
  calibration fingerprint in rollout logs.
- Compare complete two-tag camera poses with the calibration's known-good
  anchor reference at startup. If either movement threshold is exceeded, print
  the checkerboard recalibration instruction and withhold dense measurements.
- Replace the orange √M overlay with actual colored 3D points in MuJoCo and
  projected depth/confidence in the camera viewer. Keep the tag-derived green
  box only in explicit evaluation mode.
- A dense-worker failure is loud and aborts camera-mode startup; transient
  empty/invalid clouds are measurements misses and use hold/age normally.
- Register every new script/argument/resource in `panel/registry.py`.

GPU scheduling:

- SAM2 runs every camera frame.
- SAM3 remains resident only if memory permits; otherwise load/reacquire through
  an explicit managed path, never an implicit CPU fallback.
- Dense stereo runs only for static refreshes. Fast-FoundationStereo may use a
  fixed TensorRT engine after parity with PyTorch is tested; StereoSGBM remains
  the frozen CPU implementation evaluated in Stage 2.
- The PPO policy remains on CPU unless end-to-end profiling proves a shared GPU
  path is simpler and faster.

## Stage 7 — policy training

1. Pretrain/fine-tune/freeze PointNet as Stage 4 specifies.
2. Distill from the best available privileged/tag teacher using synthetic
   partial clouds. The teacher receives its old/GT-compatible observation;
   the student receives the new structured observation.
3. PPO fine-tune under `dr=none` with PointNet frozen until control succeeds.
4. Continue curriculum through `dr=light` and `dr=full`, enabling measured
   point noise, holes, outliers and occlusion in stages.
5. Only after a stable full-DR policy exists, experiment with unfreezing the
   final PointNet projection at a much lower learning rate. Keep the supervised
   auxiliary losses active to prevent RL feature collapse.
6. Compare against a policy using the explicit fit from Stage 3. Prefer the
   simpler interface unless PointNet materially improves real success or
   robustness.

Training acceptance:

- synthetic success matches or exceeds the current lift baseline across all
  resting faces;
- performance degrades gracefully under point dropout and full-cloud holds;
- the policy uses cloud features (ablation to zero/permute meaningful points
  reduces performance) without depending on padding/order artifacts;
- short real dry-runs show sensible actions while cloud age grows;
- real lift evaluation starts slow and non-executing, then follows the existing
  hardware safety checklist.

## Rollback and alternatives

- If rectification is unstable, fix the shared stereo calibration before
  changing models.
- If correspondence fails on the textureless/tag-inpainted sponge, test
  lighting and non-semantic removable texture first; permanent texture is only
  acceptable if it is part of the deployed object specification.
- If StereoSGBM and Fast-FoundationStereo fail and original FoundationStereo
  also lacks valid shared-surface depth at the closer baseline, prefer an
  active-depth camera over teaching PointNet from bad geometry.
- If dense geometry is good but explicit fitting fails, proceed with PointNet.
- If synthetic-to-real PointNet transfer fails, expand supervised real tagged
  clouds and measured corruption models before unfreezing it inside PPO.
- A DUSt3R/MASt3R-style unconstrained reconstruction is a later alternative,
  not the first implementation: it relaxes rectified-stereo assumptions but is
  heavier and complicates known metric calibration/runtime.

## Definition of done

- The shared stereo calibration passes its reprojection, rectification and
  workspace-coverage gates.
- A tag-inpainted StereoSGBM or Fast-FoundationStereo cloud passes every
  held-out offline geometry gate and pose/coverage slice.
- The selected frozen backend meets the static refresh budget; any optimized
  export matches its accepted reference output closely enough.
- Either explicit fitting passes, or the pretrained PointNet passes held-out
  real geometry tests.
- Sim and real serve the identical structured cloud contract.
- Distilled + PPO-fine-tuned policy succeeds under full measured point-cloud DR.
- Real viewer shows tag GT only in evaluation mode and actual points otherwise.
- The real rollout consumes no sponge tags and handles cloud loss through the
  shared hold/age convention.
- Old visual-hull actor code/config is removed or explicitly retained only for
  reproducible legacy evaluation, with `TODO.md` updated accordingly.
