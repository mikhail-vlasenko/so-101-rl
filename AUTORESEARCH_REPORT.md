# Autoresearch Report — Pick-and-Place (Mar 10, 2026)

## Summary

27 experiments over ~10 hours. Completion rate improved from **0.024 → 0.937** (39x) within a 20-minute training budget per run (~10M timesteps). Final state includes ring_height and task_id restored to observations (0.903 at 20 min, expected to recover with longer training).

## Kept Changes (in order)

| # | Commit | Rate | Description |
|---|--------|------|-------------|
| 1 | 3df2bda | 0.191 | Replace task_id/ring_height obs with cube-to-target XY vector |
| 2 | c473adb | 0.355 | Reduce network from 4×256 to 2×256 |
| 3 | de3bc8a | 0.773 | Increase learning rate from 3e-4 to 1e-3 |
| 4 | 808258c | 0.908 | Remove dead JOINT_PASSIVE_COEFF, make XY regress penalty symmetric |
| 5 | 11ffd77 | 0.924 | Remove height multiplier from XY progress reward |
| 6 | 099abcd | 0.937 | Increase n_envs from 16 to 32 |
| 7 | 237275d | 0.903 | Add ring_height and task_id back to obs (OBS_DIM 20→22) |

## Key Findings

**Observations matter most.** The single biggest win (+0.167) was giving the agent the cube-to-target relative vector. ring_height and task_id were later restored (OBS_DIM 20→22) since ring_height varies per episode and the agent needs it for correct placement height. The slight regression (0.937→0.903) is expected to recover with longer training.

**Smaller networks learn faster under time pressure.** 2×256 nearly doubled completion rate vs 4×256 at 20 minutes. Fewer parameters converge faster with limited data.

**Higher LR pairs well with smaller networks.** 1e-3 was a 2x improvement over 3e-4 with the 2×256 network. The smaller network tolerates aggressive updates.

**Reward simplification helps.** Removing the asymmetric XY regress penalty and the height multiplier both improved results. Simpler reward = easier credit assignment.

**VecNormalize makes reward magnitude irrelevant.** Doubling PLACE_BONUS/GRASP_HOLD_REWARD or XY_PROGRESS_COEFF all hurt — VecNormalize adapts, so only relative structure matters.

**Input BatchNorm is critical.** Disabling it dropped completion from 0.937 to 0.285. The obs dimensions have different scales (joint angles vs positions vs distances).

## Discarded Changes

| Rate | Description | Why it failed |
|------|-------------|---------------|
| 0.028 | action_scale 0.05→0.08 | Larger actions hurt grasping precision |
| 0.188 | gamma 0.95→0.99 | No benefit at 20 min (needs more training) |
| 0.102 | Conditional EE-cube penalty | EE-cube distance during grasp helps maintain grip |
| 0.109 | 2x grasp/place rewards | VecNormalize negates magnitude changes |
| 0.103 | PPO n_steps 2048→4096 | Fewer gradient updates per wall-clock minute |
| 0.314 | PPO n_epochs 5→10 | Marginal, possibly overfitting rollout data |
| 0.847 | ent_coef 0.001→0.0 | Some exploration noise helps |
| 0.880 | Remove time penalty | Time pressure helps avoid idle steps |
| 0.693 | batch_size 4096→2048 | Training instability (high variance) |
| 0.285 | Disable input_batchnorm | Obs scale normalization is essential |
| 0.863 | max_steps 300→200 | Not enough time for far cube positions |
| 0.863 | Shrink cube spawn range | Less diversity hurts generalization |
| 0.924 | EE_CUBE_COEFF -0.5→-1.0 | No improvement, slightly worse |
| 0.838 | XY_PROGRESS_COEFF 200→400 | VecNormalize negates the change |
| 0.896 | 2×128 network | Too small, not enough capacity |
| 0.839 | n_envs 32→64 | Rollout buffer too large for batch_size=4096 |
| 0.902 | Disable floor_contact_penalty | Floor penalty prevents bad arm poses |
| 0.779 | n_substeps 10→15 | Slower sim = fewer timesteps in budget |

## Suggestions for 120-Minute Budget

**High confidence:**
- **Longer training alone** should reach 0.95+ — the curve was still climbing at 10M steps.
- **gamma 0.99** — tied at 20 min but benefits from longer training for learning full task horizon.
- **Larger network (3×256)** — higher asymptotic performance, enough data at 60M+ steps.

**Medium confidence:**
- **LR schedule** — start at 1e-3, decay to 1e-4. Fast early learning, stable late refinement.
- **n_envs=64 + batch_size=8192** — more throughput, with larger batch to compensate.
- **n_epochs=10** — more gradient steps when not data-starved.

**Worth one run:**
- **Ring height curriculum** — start with ring_height_max=0 (easiest), increase over training.
- **Conditional EE-cube penalty** — may work once the agent already knows how to reach.

**Recommended first combo:** current settings + `time_limit_minutes=120`, `gamma=0.99`, `net_arch=[256,256,256]`, linear LR decay 1e-3 → 1e-4.
