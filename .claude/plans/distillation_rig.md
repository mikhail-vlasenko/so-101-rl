# Distillation rig — migrate checkpoints across architectures and obs spaces

Decision-level plan. Captures the reasoning and gotchas; implementation details TBD when picked up.

## Motivation

Every architecture or obs-layout change (net width, history features, obs_norm constants,
tag→box representation) invalidates checkpoints, and the only recovery today is a 40M-step
curriculum restart (`widen_checkpoint` covers exactly one case: same-obs widening).
Distillation turns "retrain from scratch" into a supervised problem — fast, stable, cheap —
and is the enabling tool for everything else planned: history features, asymmetric-critic
obs layout, wider nets, a possible LSTM student, and eventually the box-obs vision migration.

## Key decisions

1. **DAgger, not plain behavior cloning.** Collect states by rolling out the *student*
   (or annealing teacher→student control), query the teacher's action on those states,
   regress, aggregate. Plain BC on teacher rollouts fails from covariate shift precisely
   in the states where the student is still bad — which is where it needs supervision.
2. **Dual-obs env is the reusable core.** The env computes both the teacher's and the
   student's observation from the same `MjData` every step. For pure architecture changes
   the two views are identical; for obs-space migrations this is the crux (and the reason
   the rig, once built, de-risks the tag→box move for free). DAgger needs it too: the
   teacher must be queryable on student-visited states.
3. **Losses.** MSE on the Gaussian mean (use the teacher's deterministic action). SB3 PPO's
   `log_std` is a state-independent parameter — copy it, don't regress it. **Also distill
   the value function** (regress teacher V onto the student critic): a warm actor with a
   fresh critic is a classic trap on PPO resume — the noisy critic produces garbage
   advantages that destroy the distilled actor in the first updates.
4. **Always finish with a short PPO fine-tune** via the existing `resume` machinery.
   Distillation gets competence; fine-tuning adapts it (and is mandatory in the regimes
   below where imitation alone is provably insufficient).

## Two regimes — know which one you're running

- **Same-obs teacher → new architecture** (current checkpoint → wider net, or → net with
  history features). Transfers competence only. New memory/history inputs are *useless for
  predicting a memoryless teacher* — distillation trains them toward zero weight. That's
  fine: it's a generalized Net2Net warm start, and the RL fine-tune is where the new
  capacity gets used.
- **Privileged teacher → deployable student** (teacher trained with
  `marker_always_visible=true` or GT cube pose; student on camera obs + history features).
  Here imitation *actively trains* the student's memory: predicting the privileged action
  forces latent inference from history (the RMA / Learning-by-Cheating mechanism). This is
  the regime that teaches memory features for real.

**Fundamental limit:** a student can only imitate what its own obs make inferable. Where
the latent is genuinely unobservable, pure imitation yields the marginalized (averaged)
action — e.g. the average of "reach left" and "reach right", which is worse than either.
Expect a residual gap in exactly those states; the fine-tune closes it, and the student
should end up with information-gathering behavior there that no amount of imitation can
teach (the teacher never needed to gather information).

## Gotchas

- **Collect under the student's deployment DR** (`dr=full`): that's the distribution the
  student must master. The teacher either handles it or is privileged past it.
- **Eval before fine-tuning:** run teacher and student through the same sim eval + stats
  callbacks. Distillation bugs don't crash — they look like "slightly worse everywhere".
- **Data budget is a non-issue:** a few million transitions from the SubprocVecEnv farm is
  cheap next to 40M RL steps; don't economize on DAgger iterations.
- obs_norm constants live in the student's network and are re-derived for the student's
  layout; distillation is supervised, so obs_norm changes stop being scary — that's part
  of the point.

## Sequencing

Build once. First consumer: warm starts for the obs-history plan (`obs_history_features.md`).
Long-pole consumer: the tag→box obs migration (`vision_multicam_longterm.md`), which is
designed around this rig existing.
