# Asymmetric critic — privileged information for the value function

Decision-level plan. Orthogonal to the history features and cheap relative to its
expected value; the critic is discarded at deployment, so this changes nothing on the
real rig.

## Motivation

Partial observability hurts training twice: the actor can't see, and the critic can't
evaluate. The second cost is removable for free. Under PPO the critic never ships — so it
may consume ground truth the actor is forbidden: noisy value estimates under partial obs
make advantage estimation noisy *regardless of actor architecture*, and cleaning them up
speeds and stabilizes learning without touching what deploys. The value of this grows
with the DR latent zoo — today bias/latency/cube-size/dropout, later shape, mass,
friction (see `vision_multicam_longterm.md`).

## Key decision: flat concat, not Dict obs

**Obs vector = `[actor dims | privileged dims]`.** The actor slices `[:actor_dim]` before
its first layer; the critic consumes the full vector.

- A Dict obs space (`{"actor": ..., "privileged": ...}`) is conceptually cleaner but
  ripples through everything: VecFrameStack, the obs_norm tiling, every rollout script,
  eval, the panel. Rejected.
- SB3's rollout buffer stores only `obs` — privileged info has to ride inside the obs
  vector *either way*. The Dict version buys ceremony, not capability.
- The actor **structurally cannot** see the privileged dims (the slice happens before its
  first layer), so the "policy sees no GT" contract holds in substance, not just in
  spirit.

## Theory note (why flat concat is also the *correct* design)

A critic conditioned **only** on privileged state is formally biased in a POMDP: it
estimates V(state) while the actor lives in belief space (Baisero & Amato 2022,
"Unbiased Asymmetric Reinforcement Learning under Partial Observability"). The unbiased
form conditions the critic on *both* the actor's observation and the privileged state —
which `[actor obs | privileged]` does naturally. The pragmatic layout and the
theoretically right one coincide; don't "simplify" to a privileged-only critic.

## Privileged content

All of it already exists inside the env:

- true cube pose + velocity (not the held/noised tag pose),
- a real grasp-contact flag (cube–fingertip contacts),
- the episode's sampled DR latents: obs biases, camera pipeline delay, cube dimensions,
  dropout rates.

Each privileged dim needs its own physically-derived center/scale in obs_norm, same as
everything else.

## Deployment & contract

- Rollout scripts pad the privileged block with zeros that are never read — or export an
  actor-only artifact. Either way `load_policy`'s obs-dim check must account for the
  layout explicitly (fail loud on mismatch, as today).
- The eval env is sim, so it emits real privileged values — eval stays meaningful.

## Gotchas

- **History features apply to the actor block only.** Taps/EMAs of fresh GT are pointless
  dims; keep the privileged block un-stacked. This constrains how the history wrapper and
  frame_stack interact with the layout — decide the layout once, with both features on
  the table (they should land together anyway, since each invalidates checkpoints).
- Critic capacity: the privileged block widens the critic input; the existing net_arch is
  probably fine, but watch the value-loss curves rather than assuming.
- With the distillation rig (`distillation_rig.md`), a privileged-critic checkpoint
  distills into any future layout like any other teacher — value distillation included.

## Sequencing

Land together with the obs-history layout change to pay the checkpoint invalidation once.
No real-rig work involved beyond the zero-padding shim in the rollout scripts.
