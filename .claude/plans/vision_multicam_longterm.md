# Longer term — complex tasks, box representation, two cameras

Decision-level plan for scaling past "lift this sponge": tower building, semi-novel
objects, the vision→representation→policy pipeline, and binocular depth. This is the
long-pole track; the distillation rig, history features, and asymmetric critic plans are
its prerequisites.

## Task scaling: reuse pickplace's interface

- **The observation representation is the fork in the road.** Tags scale to a tower of
  *tagged blocks* (with caveats: a block inside a stack or in the gripper has its tag
  occluded — the held-pose+age machinery carries far more load, and reset rejection
  sampling gets combinatorially awkward with mutual occlusion). "Semi-novel" within the
  tag paradigm means **within a randomized training family** (shape-DR'd boxes/cylinders,
  all tagged) with grasp feedback + memory doing implicit shape ID. Truly novel objects
  (no tag, arbitrary geometry) require the vision pipeline below.
- **Stacking = pickplace with the place target defined by another block's top-face pose**,
  tighter tolerance, gentle release, topple penalty. **Bucket = pickplace with a large
  forgiving target** — it isolates the semi-novel *grasping* problem from placement
  precision. Both are increments on the existing phase machine, not new tasks.
- **Never train "build a tower" as one RL episode.** Train the primitive "stack the next
  block onto the current stack top" with short episodes; the randomized initial stack
  height (0/1/2 blocks at reset) *is* the curriculum; sequence the primitive outside RL.
  This keeps 300-step episodes and γ=0.99 viable and sidesteps long-horizon credit
  assignment entirely.
- **Role slots, not set encoders**, for multi-object obs: "object in hand" / "current
  target" slots assigned by the phase machine. Fixed obs dim, permutation problem
  dissolved, obs_norm boxes stay hand-settable.

## New physics problems (sysid so far is arm-only)

- Placement needs ~5 mm accuracy vs lift's 10 cm threshold — the known elbow
  under-tracking plant gap and the calibration residual start costing successes, not
  style points.
- **The sponge is the wrong object for a tower**: it deforms; the sim box doesn't. Benign
  for lifting, central for stack stability. Use rigid tagged blocks (tags on multiple
  faces so one is visible inside a stack) and do an object-contact sysid pass (friction,
  restitution, mass) + DR over those parameters.
- **Toppling is irreversible within the episode** — unlike a failed grasp there is no
  retry. Extend the careful-behavior shaping (tower-contact force term, analog of the
  floor/poke terms) and note this raises the value of good belief/memory: knocking things
  you can't currently see is precisely the memory failure mode.
- **Hardware ceilings to measure before designing the object family:** gripper max
  aperture, servo torque/payload, camera FOV + height ceiling vs a 3-block tower's top
  tag, tag size vs detectability (smaller objects → smaller tags → earlier, noisier
  dropout → wider `marker_dropout` DR).

## Vision → compact representation → policy

Same pattern as the tag pipeline — swap the estimator, widen the representation. Sim
trains on GT + modeled errors; SAM 2 never runs in the training loop.

- **Representation: oriented box** (position, yaw, dimensions), **canonicalized** — sorted
  dims, yaw modulo the box symmetry — otherwise real estimator flip-flops become ±90° obs
  jumps. Everything else transfers: held-pose+age, dropout convention, `camera_sim` frame
  schedule, obs_norm, sim↔real contract tests. Arm markers stay on the tag pipeline;
  don't couple the migrations.
- **The whole sim2real bet moves into error-model fidelity**, and a segmentation→box
  pipeline has *structured, state-dependent* errors Gaussian DR doesn't cover:
  - partial occlusion **shrinks the mask** → box center shifts toward the visible part
    and dims shrink, systematically (a bias mode, not a dropout mode);
  - symmetry flips;
  - SAM 2 is a tracker: errors are **autocorrelated** (drift, then snap on re-detection),
    unlike per-frame tag noise.
- **Order of work: characterize the real estimator FIRST**, then derive the DR from
  measured errors (the `probe_cam_latency` methodology). The failure mode to avoid is
  training 40M steps against hand-guessed Gaussians. Cheap validation along the way:
  run the real estimator on **sim renders** offline (error model without rig time — SAM 2
  tolerates non-photoreal input well enough); **replay recorded real vision outputs**
  through the policy before any `--execute`.
- Latency/compute: re-measure `delay_ms`/`frame_ms` with SAM 2 in the loop; deploy needs
  a GPU next to the rig (the policy stays CPU).
- Payoff: dimensions *in* the obs means shape is told, not inferred — generalization over
  the family becomes interpolation, and "semi-novel" gets a crisp definition: **objects
  whose grasp is adequately predicted by their bounding box.** Grasp-feedback memory
  remains the safety net for when the box lies (aperture at contact ≠ advertised dim).
- Migration path: the distillation rig's dual-obs env emits tag-pose and box obs from the
  same state → existing checkpoints distill into box-obs students, no curriculum restart.

## Two cameras

- **Sparse object-level triangulation, NOT dense stereo.** Unsynced webcams + textureless
  objects are exactly SGBM's failure modes, and dense depth isn't needed — triangulate
  the matched mask/box across two views. Depth sensitivity is Z²/(f·B) per pixel of
  matching error ≈ **0.9 mm/px** at Z=0.5 m, f≈1350 px (C922 @1080p), B=0.2 m — well
  inside the 5 mm stacking budget even with a few px of centroid slop.
- Kills both monocular weaknesses: no table-plane assumption (object stays localizable
  after lift-off while both cameras see it) and metric dimensions.
- **Extrinsics via independent table-tag anchoring per camera** — triangulate in world
  frame; no stereo calibration, no rigid bar, a bumped camera self-heals at the next
  anchor. Cost: cross-camera consistency inherits tag-extrinsics error (few mm). Upgrade
  path if placement precision demands it: rigidly mounted pair + proper stereo cal.
- **Sync is the honest gotcha:** free-running cameras → up to ~half-frame offset → ~3 mm
  error at 0.2 m/s along the motion direction. Fix classically: v4l2 timestamps +
  interpolate one camera's (smooth) track to the other's capture time. The error vanishes
  during slow, careful placement — exactly when precision matters.
- Compute: SAM 2 per view doubles GPU cost; alternative is full segmentation on one view
  + light verification around the epipolar-projected box on the other. Decide after
  measuring single-view latency.
- Bonuses: **occlusion decorrelation** (the gripper rarely hides the object from both
  views — the held-pose channels go stale far less often, which the tower task badly
  needs) and **cross-view consistency as fault detection**: if the two rays don't nearly
  intersect, one tracker has drifted — demote the view inconsistent with the recent track
  to "lost" instead of fusing garbage. A single camera can never detect this.

## Degradation ladder (what the policy sees)

- **Both views:** triangulated box; two per-camera age channels at the pipeline-delay
  floor. Per-camera ages replace the single age — the pair encodes the regime for free,
  and their slow EMAs are per-camera reliability estimates.
- **One view:** fresh **bearing** from the live camera's mask centroid + **held depth
  along that ray** (recorded at the last triangulation) + held dims (constant per
  episode) + held rotation (or mask-derived yaw — deferred choice; yaw needs no depth).
  This is the held-pose convention applied per-coordinate: hold only the unobservable
  number, keep measuring the other two.
  - **Rejected: table-plane depth** (breaks the moment the object lifts, needs the plane,
    creates an on-table/in-air regime discontinuity).
  - **Rejected as primary: apparent-size depth.** Its input is mask *extent*, and partial
    occlusion — the main reason to be in single-view mode — clips the extent and biases
    depth away from the camera exactly when it's relied on. The centroid (bearing) is far
    more robust to clipping. Keep apparent-size as a refinement only if depth-hold
    measurably falls short.
  - Fixed rig ⇒ the policy learns "camera k stale ⇒ uncertainty lies along the other
    camera's ray" as a constant geometric fact; no explicit covariance channel needed.
  - In-hand depth staleness is acceptable: a grasped object is kinematically tied to the
    gripper, and the policy has markers + qpos; growing ages are its cue to weight
    proprioception.
- **No view:** hold everything, both ages grow to the cap (existing convention).
- **Accept the jumps at regime switches; no filtering in the estimator.** A filter adds
  latency and hidden estimator state that would have to be replicated exactly in sim; the
  pipeline philosophy is raw measurements + honest conventions.
- **Sim runs the identical fallback mechanism code on GT quantities** (per-camera
  `mj_ray` visibility → project GT into the live sim camera for the bearing + held depth
  from the last sim triangulation). Mechanism simulation, not noise-distribution
  imitation: the anisotropy, drift, and re-triangulation jump emerge correct by
  construction — same philosophy as `camera_sim` simulating the frame schedule. This is
  what keeps the sim↔real contract test writable.
- **Reset protocol:** object placed visible in *both* cameras (two-camera version of the
  current rejection sampling, matching how it'd be placed on the real rig) so a valid
  triangulation always precedes any fallback. Never-seen stays zeros + capped ages.

## Sequencing

1. Shape-DR grasp family + bucket place (semi-novel grasping, forgiving target) — pure
   extension of lift/pickplace; needs the history features.
2. Stack-one-block on rigid tagged blocks — isolates precision, release, occlusion; needs
   the object-contact sysid pass.
3. Tower = curriculum over initial stack height, sequenced outside RL.
4. Vision/two-camera track runs in parallel, **starting with estimator characterization**,
   and merges via the distillation rig once the box representation's error model is
   trustworthy.

Buy-vs-build note kept for honesty: an active-IR depth camera (~$300) is the trivial path
to synced, textureless-robust depth, at the cost of new noise quirks and another device
class. The dual-C922 design wins on reuse: known camera, existing calibration and
anchoring protocol, and the staleness conventions carry over unchanged.
