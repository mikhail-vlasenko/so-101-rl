# Observation history — lag taps + multi-timescale EMA traces

Decision-level plan. Fixed history features instead of a recurrent policy; the reasoning
for that choice is recorded here because it will be questioned later.

**Status (2026-07-09): lag taps implemented** — `history_taps` in `conf/config.yaml`
(default `[0]`), shared ring buffer `src/obs_history.py`, layout
`[actor block per tap | single priv tail]`, `frame_stack` deleted (subsumed; the A/B
showed plain stacking doesn't help). Warm start via `src/distill.py
distill.teacher_obs=current`. EMAs deliberately not built yet; see TODO.md
"Obs history: follow-ups".

## Motivation

The env is a POMDP by design: held-last-pose tags with age channels, and DR that
deliberately creates **per-episode constant latents** (marker/cube bias ~5 mm, camera
delay 42–52 ms, cube dims ±1 cm, dropout rates). A memoryless policy must play
robust-to-worst-case across those latents; a policy with history can identify them online
and adapt (implicit sysid). Motivating behavior: remembering a failed grasp — gripper
closed on nothing — long enough to inform the retry.

## Why fixed history features, not RecurrentPPO

- Plain PPO stays: no BPTT, no sequence batching, no 2–5× throughput hit, no
  hidden-state debugging (a corrupted belief poisons the rest of an episode and can't be
  inspected like an obs vector).
- Bounded sim-overfitting surface: a memory policy *hunts* for temporal regularities
  (30 fps sawtooth, servo-profile lag shape); a fixed window caps what it can exploit.
- Warm start is trivial (below); LSTM warm starts are not.
- **Escalation criterion:** if taps/EMA plateau, the residual gap is specifically
  *event-anchored* memory (fixed lags sample clock time; an event drifts through the taps
  and ages out past the max lag). That — and only that — is the justification for
  RecurrentPPO later, not "recurrence is more powerful in general".

## Key decisions

1. **Cheap A/B first: `frame_stack=3`.** Already fully plumbed (VecFrameStack + obs_norm
   tiling). Config-only signal for "does temporal context help at all" before building
   anything.
2. **Lag taps: coverage first, spacing second.** Requirement: the failure signature must
   remain visible in *some* tap from the failed close until the retry completes → max lag
   ≥ retry-cycle time. At 15 Hz that's likely 32–64 ticks (~2–4 s) — do not guess:
   **measure the close→re-close time distribution** from eval episodes / rollout CSVs
   before fixing numbers. Geometric spacing (~×3–4, e.g. [0, 4, 16, 48] ≈
   0/0.27/1.07/3.2 s) spends taps efficiently: dense where dynamics live, sparse where
   events live. Linear spacing ([0,4,8,16]) wastes taps on the redundant mid-range and
   slides off the event exactly when the retry starts.
3. **EMA traces with 2–3 time constants**, τ ≈ 0.3 / 1.5 / 5 s
   (alpha = 1 − exp(−dt/τ) ≈ 0.2 / 0.044 / 0.013). Division of labor: snapshots keep
   events crisp; EMAs estimate slow latents. Free insight: the **EMA of the age channels**
   is an online per-episode detection-rate estimate — a dropout-latent sysid feature by
   accident.
4. **obs_norm comes for free.** An EMA is a convex combination of past obs → each traced
   dim keeps its source dim's center/scale (the workspace-box bounds tests stay valid);
   taps tile exactly like the existing frame_stack tiling.
5. **One shared history class** (ring buffer + EMA update + reset convention) used by both
   the sim wrapper and the real rollout scripts, contract-tested like `ArmLoop`. A silent
   train/real mismatch in tap indexing or EMA update is undetectable at deploy and deadly.
   Reset convention: pad taps with the reset obs / init EMAs to the first obs — matches
   the real boot condition (no history exists).
6. **Warm start:** expand the first Linear layer — copy the trained FF checkpoint's
   weights into the current-frame columns, zero-init all history columns. The network
   starts computing exactly the current policy and learns to use history on top
   (same idea as `widen_checkpoint`). Alternatively via the distillation rig
   (`distillation_rig.md`), which subsumes this.

## Insights / limits worth remembering

- **Time-anchored vs event-anchored memory** is the fundamental limitation of fixed lags.
  Partial mitigation if needed: hand-latched event features (e.g. "min gripper aperture
  since the gripper was last fully open" — a crisp *the-last-close-found-nothing* bit).
  Same design family as held-pose+age, composes with the taps.
- What history does **not** need to solve: per-episode marker bias is single-frame
  observable in principle (marker pose vs FK from qpos); history only averages noise on
  top of that.
- Held-pose dims are near-duplicates across taps during a long occlusion — harmless; the
  differing age values still encode the sighting history.
- If obs bloat ever matters, taps over a *subset* of dims (gripper aperture is the one
  that provably needs lookback) is a legitimate cut; the failed-grasp *location* is
  already carried by the held cube pose and current qpos.

## Sequencing

frame_stack=3 A/B → measure retry timescale → taps + EMA with warm start → (independent
axis) privileged critic, see `asymmetric_critic.md`. Obs dim changes invalidate
checkpoints — batch this with other obs-layout changes to pay that cost once.
