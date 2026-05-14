# Autoresearch Report — Lift (May 14, 2026)

## Summary

24 experiments over ~8 hours on branch `autoresearch-lift-may14`. Final
`rollout/lift/mean_max_cube_height` reached **0.039 m** (vs ~0.027 re-baseline,
target 0.10 m). Three changes survived; the rest reverted. The policy still
does not reliably *complete* the lift (mean episode length stayed near
truncation, ~280–296 of 300 steps), so this is partial progress, not a solve.

## Kept Changes (in order)

| # | Commit | Last-min metric | Description |
|---|--------|-----------------|-------------|
| 1 | 9d52a21 | 0.031 | `frame_stack` 1 → 4 |
| 2 | 30aef0c | 0.035 | `GRASP_HOLD_REWARD` 0.05 → 0.5 (10×) |
| 3 | 6872e42 | 0.039 | PPO `gae_lambda` 0.95 → 0.97 |

## Key Findings

**Temporal context unlocked progress.** Frame stack 1 → 4 was the first
non-noise improvement after 8 single-knob failures. With `obs_latency=2` the
policy sees stale observations; multiple stacked frames let it estimate cube
and arm velocities implicitly. `frame_stack=8` was worse, so the sweet spot
sits at 4. This was the discovery that broke the cluster of "every change is
0.024 ± noise" results.

**The grasp signal was an order of magnitude too small.** With
`GRASP_HOLD_REWARD=0.05/step`, a 300-step episode of perfect grasping was worth
+15 — barely offsetting the −15 from `TIME_PENALTY`. Bumping to 0.5/step gave
the policy a clear gradient to actively maintain grasp rather than treat it as
incidental. Pushing to 1.0 did not help further (within noise of 0.5).

**Longer-horizon credit assignment helped, slightly.** GAE λ 0.95 → 0.97
propagates terminal/late rewards further back. The lift task has a long
sequence (approach → grasp → lift, ~100+ steps) where credit for the lift
should flow back into the approach. λ=0.98 over-shot and regressed on every
percentile.

**Reward shaping mostly failed.** Of seven reward-shaping attempts (held-up
bonus at two thresholds, height-scaled grasp, grasp-gated lift, pre-grasp
pose, terminal `SUCCESS_BONUS` at two combinations, 5× `HEIGHT_PROGRESS_COEFF`),
*none* survived. The interpretation that fits all of them: the policy never
gets reliably into the rare "grasping + lifting" regime where these bonuses
fire, so they act as noise on top of the existing dense shaping.

**Variance is large; trust distributions not single numbers.** The first
"baseline" returned 0.034 last-minute, suggesting many later experiments were
regressions; a re-baseline returned 0.027 with the same code. The
last-minute average has ~0.005–0.010 noise. After that, judging by full-run
distribution (`q75`, `max`) was more reliable than the headline number.

**Domain randomization is not the current bottleneck.** Per a user-requested
diagnostic, zeroing `obs_noise` and `obs_latency` (with all other kept
changes in place) returned **0.032** last-min — *worse* than 0.039 with full
DR. Removing the noise did not unlock learning, so the gap to the target is
not "DR is too hard" but "the policy is stuck in a drag-then-give-up
behavior". `cube_drag_ratio` was 0.43 in the diagnostic run vs ~0.27–0.33 in
the noised runs.

## Discarded Changes

| Last-min | Commit | Description | Why it failed |
|----------|--------|-------------|---------------|
| 0.025 | cc6b13e | `ent_coef` 0.001 → 0.01 | Over-explored, settled on drag |
| 0.027 | ed91d60 | Held-up bonus 2.0·max(0, z−0.02) | Bonus rarely triggered |
| 0.022 | 17b9457 | `gamma` 0.95 → 0.99 | Shaping is tuned for short horizon |
| 0.025 | 6019e86 | Grasp-gated lift bonus 5.0·max(0, z−0.05) | Grasp itself never reached |
| 0.016 | 0766cbb | Switch PPO → SAC | Only 0.31M steps in 20 min; didn't learn |
| 0.024 | 48b0d34 | `HEIGHT_PROGRESS_COEFF` 200 → 1000 | Bigger signal, still unreachable |
| 0.024 | c7ae5ab | Pre-grasp pose bonus (xy<4 cm + z-above) | No effect on metric |
| 0.024 | 2348a07 | `input_batchnorm` true → false | BN was not the bottleneck |
| 0.024 | 1b694f5 | Net 3×256 → 3×512 | Not capacity-limited |
| 0.024 | bd147b7 | `frame_stack` 4 → 8 | Diminishing returns |
| 0.029 | 68869dd | `n_envs` 32 → 64 (with fs=4) | Distribution worse (max 0.032 vs 0.045) |
| 0.032 | 1239505 | fs=4 + held-up bonus | Held-up still doesn't help |
| 0.024 | 5fb4091 | fs=4 + constant LR | Linear decay was needed |
| 0.023 | 3fc8438 | PPO `n_epochs` 5 → 10 | Overfit stale rollout |
| 0.024 | ad6723b | LR 1e-3 → 2e-3 | Destabilized |
| 0.024 | cbd1f83 | `clip_range` 0.2 → 0.3 | No effect |
| 0.033 | 019e947 | `GRASP_HOLD_REWARD` 0.5 → 1.0 | No further gain over 0.5 |
| 0.030 | 050f924 | + height-scaled grasp (5.0·cube_z) | Slight regression |
| 0.035 | d6871c2 | + `SUCCESS_BONUS=50` (alone) | Fires too rarely (ep_len ~280) |
| 0.037 | c3ec818 | `gae_lambda` 0.97 → 0.98 | Over-shot 0.97 peak |
| 0.033 | b836962 | `SUCCESS_BONUS=50` (with gae=0.97) | Same: bonus rarely fires |
| 0.032 | 4db30c8 | DIAGNOSTIC: zero `obs_noise` + `obs_latency` | DR is not the bottleneck |

## Final State

Branch `autoresearch-lift-may14` at commit **6872e42**. Diffs from `master`:

- `conf/config.yaml`: `frame_stack: 4`, `ppo.gae_lambda: 0.97`
- `lift_env.py`: `GRASP_HOLD_REWARD = 0.5` (was 0.05)

All sim-to-real defenses (`obs_noise`, `obs_latency=2`, `obs_bias` scaffolding)
unchanged.

## Open Questions / Next Steps

1. **Why does the policy drag instead of lift, even with no noise?** The
   diagnostic run shows ~43% of timesteps are "cube on floor + lateral motion".
   The policy reliably reaches the cube but doesn't transition to a lifting
   action. Worth instrumenting *grasp rate* (how often `_detect_grasp` is true
   per episode) and *post-grasp height delta* to see whether the bottleneck is
   discovering grasp or following through after grasping.

2. **The 20-min budget may simply be too tight for this task.** The fact that
   PPO is still climbing at end-of-training in every kept run (`q90 > q75 > med`)
   suggests more wallclock would help. If lifting the budget is on the table,
   it would calibrate whether the current setup *can* solve the task at all.

3. **Curriculum on cube spawn range** was not tried. Starting with a narrow
   spawn (cube at ~`[0.20, 0.0]`) and widening over training is a standard
   sim-to-real warm-start; it doesn't violate the DR rules if the *final*
   policy is trained on the full distribution.

4. **The "drag local optimum"** appears in `cube_drag_ratio` (0.27–0.43 across
   runs) and in `mean_max_cube_height` plateauing near the cube's resting
   height. Adding a per-step *anti-drag* penalty was previously catastrophic in
   the pickplace session (−0.02 made the policy avoid the cube entirely), so a
   weaker version (e.g., −0.002) might be worth trying. Pairing it with the
   stronger grasp reward could be the right balance now.

5. **Gripper/cube physics.** The cube is 3 cm with `friction=1.0` and
   `solref=0.001 1` (very stiff). It's worth a non-RL check: can a scripted
   trajectory grasp and lift this cube in the current sim? If not, the policy
   isn't the issue.
