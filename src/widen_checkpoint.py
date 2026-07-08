"""Function-preserving width expansion (Net2WiderNet) for trained checkpoints.

Widen a [W]*L policy into a [k*W]*L one whose actor, critic and value/Q heads
compute nearly the same function at initialization (identical up to O(noise^2),
see "Symmetry breaking" below), so `resume=`-training the bigger network starts
from the original checkpoint's performance instead of from scratch -- no
distillation phase, no regression, deterministic.

The only architecture-specific subtlety is the LayerNorm inside each hidden block
(Linear -> LayerNorm -> GELU, see src/networks.build_mlp). LayerNorm normalizes
across the feature dimension, so widening is exactly function-preserving only when
it *uniformly* replicates units: widening a layer by an integer factor k copies
every old unit exactly k times, which leaves the per-layer mean and variance
untouched (e.g. 256 -> 512, k=2). Non-integer ratios are rejected.

Symmetry breaking. Exact replicas are a trap: identical units get identical
gradients forever (a degenerate wide net that never uses its extra capacity), and
even *near*-identical units barely diverge under Adam -- a weight row's gradient
is always a scalar times the shared layer input, so Adam's per-parameter
normalization makes two tied copies move in lockstep (measured; see the
test-file docstring). So distinctness must be *seeded* at init: each old unit's
copies get small Gaussian jitter on their incoming weights, drawn zero-sum across
the copy group so the first-order effect on the layer's output cancels (function
preserved to O(noise^2)) while the copies become genuinely distinct, higher-rank
units. `noise=0` reproduces the exact-but-degenerate replication.

Usage:
    python -m src.widen_checkpoint in.zip out.zip                    # double every layer
    python -m src.widen_checkpoint in.zip out.zip --net-arch 512,512,512,512
    python -m src.widen_checkpoint in.zip out.zip --algorithm sac
then resume with:
    python -m src.train env=lift resume=out.zip
The new width rides in out.zip's policy_kwargs, which resume restores directly;
train.net_arch (used only for from-scratch construction) is inert on resume.
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv

from src.networks import ObsNorm, TakeFirst
from src.train import ALGORITHM_REGISTRY


class _SpacesEnv(gym.Env):
    """Minimal env exposing given spaces so an SB3 model can be constructed to
    copy weights into. reset/step are never called -- model construction reads
    only the spaces."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def _replication_maps(old_width, new_width):
    """Return (out_map, out_factors) for widening one layer old_width -> new_width.

    out_map[i] is the old unit that new unit i copies (each old unit copied
    exactly reps = new_width // old_width times -> uniform, so LayerNorm stats are
    preserved). out_factors[i] = 1/reps is the fraction of that old unit's
    *outgoing* weight new unit i carries; they sum to 1 per old unit so the
    downstream pre-activation is unchanged. Copies of unit c sit at strided rows
    c, c+old_width, ... which _incoming_noise relies on."""
    if new_width % old_width != 0 or new_width < old_width:
        raise ValueError(
            f"cannot widen {old_width} -> {new_width}: LayerNorm-exact widening "
            "needs an integer replication factor (new width a multiple of old)")
    reps = new_width // old_width
    out_map = [c for _ in range(reps) for c in range(old_width)]
    out_factors = [1.0 / reps] * new_width
    return out_map, out_factors


def _incoming_noise(new_w, reps, old_out, noise, rng):
    """Add zero-sum Gaussian jitter across each copy group's incoming weights, so
    the replicas become distinct but the group's summed output is unchanged to
    first order. Rows are laid out [copy0 of all units, copy1 of all units, ...],
    i.e. copy r of unit c is row r*old_out + c."""
    if noise == 0 or reps == 1:
        return
    jitter = rng.standard_normal((reps, old_out, new_w.shape[1])) * (noise * float(new_w.std()))
    jitter -= jitter.mean(axis=0, keepdims=True)          # cancels at first order (factors 1/reps)
    new_w += torch.as_tensor(jitter.reshape(new_w.shape), dtype=new_w.dtype)


def _copy_linear(old_lin, new_lin, out_map, in_cols, in_facs):
    """Copy old_lin -> new_lin, replicating output rows by out_map and input
    columns by (in_cols, in_facs) with per-column weight fractions."""
    w = old_lin.weight.data
    new_lin.weight.data.copy_(w[out_map][:, in_cols] * torch.tensor(in_facs, dtype=w.dtype))
    new_lin.bias.data.copy_(old_lin.bias.data[out_map])


def _widen_sequential(old_seq, new_seq, rng, noise):
    """Widen an MLP chain [ObsNorm?, (Linear, LayerNorm, GELU)*, Linear?] in
    lockstep. Returns (out_map, out_factors) of the last *hidden* Linear so
    standalone heads can consume it. A trailing Linear whose output stays the
    same width (a Q head projecting to 1-D) is widened here as an input-only
    expansion and does not become the returned "last hidden" layer."""
    in_cols, in_facs = None, None          # None -> this layer's input isn't widened
    last_map, last_facs = None, None
    for old_m, new_m in zip(old_seq, new_seq):
        if isinstance(old_m, TakeFirst):
            # Asymmetric-critic actor slice: width-only widening never changes
            # the obs layout, so the slice must carry over unchanged.
            assert old_m.n == new_m.n, (old_m.n, new_m.n)
        elif isinstance(old_m, ObsNorm):
            new_m.center.copy_(old_m.center)
            new_m.scale.copy_(old_m.scale)
        elif isinstance(old_m, nn.Linear):
            old_out, old_in = old_m.weight.shape
            new_out, new_in = new_m.weight.shape
            if new_in == old_in:
                cols, facs = list(range(old_in)), [1.0] * old_in
            else:
                assert in_cols is not None and len(in_cols) == new_in, (new_in, in_cols)
                cols, facs = in_cols, in_facs
            if new_out == old_out:                       # terminal head: output unchanged
                out_map, out_facs = list(range(old_out)), None
            else:
                out_map, out_facs = _replication_maps(old_out, new_out)
                last_map, last_facs = out_map, out_facs
            _copy_linear(old_m, new_m, out_map, cols, facs)
            if out_facs is not None:                     # seed distinctness among the copies
                _incoming_noise(new_m.weight.data, new_out // old_out, old_out, noise, rng)
            in_cols, in_facs = out_map, out_facs         # feeds the next Linear / its LayerNorm
        elif isinstance(old_m, nn.LayerNorm):
            new_m.weight.data.copy_(old_m.weight.data[in_cols])
            new_m.bias.data.copy_(old_m.bias.data[in_cols])
        elif isinstance(old_m, nn.GELU):
            pass
        else:
            raise TypeError(f"unexpected module in MLP: {type(old_m).__name__}")
    return last_map, last_facs


def _widen_head(old_lin, new_lin, in_cols, in_facs):
    """Widen a terminal Linear whose input was widened but output dim is fixed."""
    assert in_facs is not None
    _copy_linear(old_lin, new_lin, list(range(old_lin.weight.shape[0])), in_cols, in_facs)


def _widen_ppo(old_p, new_p, rng, noise):
    ex_old, ex_new = old_p.mlp_extractor, new_p.mlp_extractor
    pi_map, pi_facs = _widen_sequential(ex_old.policy_net, ex_new.policy_net, rng, noise)
    vf_map, vf_facs = _widen_sequential(ex_old.value_net, ex_new.value_net, rng, noise)
    _widen_head(old_p.action_net, new_p.action_net, pi_map, pi_facs)
    _widen_head(old_p.value_net, new_p.value_net, vf_map, vf_facs)
    new_p.log_std.data.copy_(old_p.log_std.data)         # state-independent, width-invariant


def _widen_sac(old_p, new_p, rng, noise):
    a_map, a_facs = _widen_sequential(old_p.actor.latent_pi, new_p.actor.latent_pi, rng, noise)
    _widen_head(old_p.actor.mu, new_p.actor.mu, a_map, a_facs)
    _widen_head(old_p.actor.log_std, new_p.actor.log_std, a_map, a_facs)
    for old_c, new_c in ((old_p.critic, new_p.critic),
                         (old_p.critic_target, new_p.critic_target)):
        for qf_old, qf_new in zip(old_c.q_networks, new_c.q_networks):
            _widen_sequential(qf_old, qf_new, rng, noise)  # each ends in its own Q head


_WIDEN = {"ppo": _widen_ppo, "sac": _widen_sac}


def _max_output_diff(algorithm, old_p, new_p, obs_t):
    """Largest absolute difference between the two policies' outputs on obs_t."""
    if algorithm == "ppo":
        do, dn = old_p.get_distribution(obs_t), new_p.get_distribution(obs_t)
        diffs = [(do.distribution.mean - dn.distribution.mean).abs().max(),
                 (do.distribution.stddev - dn.distribution.stddev).abs().max(),
                 (old_p.predict_values(obs_t) - new_p.predict_values(obs_t)).abs().max()]
    else:
        act = old_p.actor(obs_t, deterministic=True)
        diffs = [(act - new_p.actor(obs_t, deterministic=True)).abs().max(),
                 (torch.cat(old_p.critic(obs_t, act), dim=1)
                  - torch.cat(new_p.critic(obs_t, act), dim=1)).abs().max()]
    return float(max(d.item() for d in diffs))


def widen_model(old_model, algorithm, new_arch=None, *, noise=3e-2, seed=0, tol=0.1,
                device="cpu", n_check=256):
    """Build a wider copy of old_model computing (nearly) the same function,
    verifying the max output difference is below tol before returning it. The
    copies are seeded distinct (noise) so the wider net isn't degenerate; the
    optimizer starts fresh (moment shapes changed)."""
    algo_cls, _, policy_cls = ALGORITHM_REGISTRY[algorithm]
    assert isinstance(old_model, algo_cls), (type(old_model), algorithm)
    old_model.policy.to(device)          # align source with the copy target for the diff check
    old_arch = list(old_model.policy_kwargs["net_arch"])
    assert all(isinstance(w, int) for w in old_arch), \
        f"width-only widening needs a flat net_arch, got {old_arch}"
    if new_arch is None:
        new_arch = [2 * w for w in old_arch]
    if len(new_arch) != len(old_arch):
        raise ValueError(f"depth must stay {len(old_arch)} layers, got net_arch {new_arch}")

    rng = np.random.default_rng(seed)
    new_pk = dict(old_model.policy_kwargs)
    new_pk["net_arch"] = list(new_arch)
    env = DummyVecEnv([lambda: _SpacesEnv(old_model.observation_space, old_model.action_space)])
    construct_kwargs = {"buffer_size": 1, "learning_starts": 0} if algorithm == "sac" else {}
    new_model = algo_cls(policy_cls, env, policy_kwargs=new_pk, device=device, **construct_kwargs)

    old_model.policy.set_training_mode(False)
    new_model.policy.set_training_mode(False)
    with torch.no_grad():
        _WIDEN[algorithm](old_model.policy, new_model.policy, rng, noise)
        obs_dim = old_model.observation_space.shape[0]
        obs_t = torch.as_tensor(
            (rng.standard_normal((n_check, obs_dim)) * 2.0).astype(np.float32), device=device)
        diff = _max_output_diff(algorithm, old_model.policy, new_model.policy, obs_t)
    if diff > tol:
        raise RuntimeError(
            f"widening not function-preserving: max output diff {diff:.2e} > tol {tol:.0e}")
    print(f"widened {old_arch} -> {list(new_arch)}  |  max output diff {diff:.2e}")
    return new_model


def widen_checkpoint(in_path, out_path, algorithm, new_arch, *, noise, seed, tol, device):
    algo_cls, _, _ = ALGORITHM_REGISTRY[algorithm]
    old_model = algo_cls.load(in_path, device=device)
    new_model = widen_model(old_model, algorithm, new_arch,
                            noise=noise, seed=seed, tol=tol, device=device)
    new_model.save(out_path)
    print(f"saved widened checkpoint -> {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="trained .zip checkpoint to widen")
    p.add_argument("output", help="path for the widened .zip")
    p.add_argument("--algorithm", choices=list(ALGORITHM_REGISTRY), default="ppo")
    p.add_argument("--net-arch", default=None,
                   help="target widths, comma/space separated (default: double every layer)")
    p.add_argument("--noise", type=float, default=3e-2,
                   help="relative incoming-weight jitter that seeds copy distinctness "
                        "(0 = exact but degenerate replicas)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.1, help="abort if max output diff exceeds this")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    new_arch = ([int(x) for x in args.net_arch.replace(",", " ").split()]
                if args.net_arch is not None else None)
    widen_checkpoint(args.input, args.output, args.algorithm, new_arch,
                     noise=args.noise, seed=args.seed, tol=args.tol, device=args.device)


if __name__ == "__main__":
    main()
