"""Function-preserving migration of a symmetric checkpoint onto the asymmetric
[actor block | privileged tail] obs layout (base_env.priv_dim_for).

The asymmetric critic changed the env observation from the bare actor block to
[actor block | privileged tail], invalidating every earlier checkpoint. But the
*actor's* input is unchanged — the new actor slices the first actor_obs_dim
dims (src/networks.TakeFirst), which is exactly the old observation — so unlike
a genuine obs change, no distillation is needed: the old actor copies verbatim,
and the old critic copies with zero weights on the first Linear's new
privileged input columns. Both heads then compute bit-for-bit the old function
(verified before saving), and the zero columns receive gradient immediately on
fine-tune — a warm actor with its own equally-warm critic, so the classic
fresh-critic PPO-resume trap does not apply.

Output is a `resume`-compatible checkpoint carrying the new observation space,
obs_norm constants, and policy_kwargs.actor_obs_dim.

Usage:
    python -m src.asymmetrize_checkpoint old.zip new.zip env=lift
then fine-tune:
    python -m src.train env=lift resume=new.zip dr=light
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, spaces
from hydra import compose, initialize
from stable_baselines3.common.vec_env import DummyVecEnv

from src.base_env import priv_dim_for
from src.networks import ObsNorm, TakeFirst
from src.train import (
    ALGORITHM_REGISTRY, actor_obs_dim_for, build_fresh_model, obs_norm_for,
)


class _SpacesEnv(Env):
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


def _copy_same_graph(old_mods: list, new_mods: list) -> None:
    """Copy an identically-shaped MLP chain module-by-module. ObsNorm constants
    are asserted equal, not copied — a drift in src/obs_norm.py would silently
    break function preservation, so it must abort here."""
    assert len(old_mods) == len(new_mods), (len(old_mods), len(new_mods))
    for old_m, new_m in zip(old_mods, new_mods):
        assert type(old_m) is type(new_m), (type(old_m), type(new_m))
        if isinstance(old_m, ObsNorm):
            assert torch.equal(new_m.center, old_m.center) and \
                torch.equal(new_m.scale, old_m.scale), \
                "obs_norm constants drifted from the checkpoint's; migration " \
                "would not be function-preserving"
        elif isinstance(old_m, nn.Linear):
            assert new_m.weight.shape == old_m.weight.shape, \
                (new_m.weight.shape, old_m.weight.shape)
            new_m.weight.data.copy_(old_m.weight.data)
            new_m.bias.data.copy_(old_m.bias.data)
        elif isinstance(old_m, nn.LayerNorm):
            new_m.weight.data.copy_(old_m.weight.data)
            new_m.bias.data.copy_(old_m.bias.data)
        elif isinstance(old_m, nn.GELU):
            pass
        else:
            raise TypeError(f"unexpected module in MLP: {type(old_m).__name__}")


def _copy_actor(old_seq: nn.Sequential, new_seq: nn.Sequential, actor_dim: int) -> None:
    """New actor = [TakeFirst(actor_dim), <old graph>]: the slice output is
    exactly the old input, so everything after it copies verbatim."""
    new_mods = list(new_seq)
    assert isinstance(new_mods[0], TakeFirst) and new_mods[0].n == actor_dim, \
        "new model's actor must start with TakeFirst(actor_dim) — was it built " \
        "with actor_obs_dim?"
    _copy_same_graph(list(old_seq), new_mods[1:])


def _copy_critic(old_seq: nn.Sequential, new_seq: nn.Sequential, actor_dim: int) -> None:
    """New critic = old graph with a wider first Linear: old input columns
    verbatim, privileged columns zero. Zero columns keep the value function
    bit-identical while still receiving gradient from step one (input-column
    weights have no duplicated-unit symmetry to break)."""
    old_mods, new_mods = list(old_seq), list(new_seq)
    assert len(old_mods) == len(new_mods), (len(old_mods), len(new_mods))
    old_norm, new_norm = old_mods[0], new_mods[0]
    assert isinstance(old_norm, ObsNorm) and isinstance(new_norm, ObsNorm)
    assert torch.equal(new_norm.center[:actor_dim], old_norm.center) and \
        torch.equal(new_norm.scale[:actor_dim], old_norm.scale), \
        "obs_norm actor-block constants drifted from the checkpoint's"
    old_lin, new_lin = old_mods[1], new_mods[1]
    assert isinstance(old_lin, nn.Linear) and isinstance(new_lin, nn.Linear)
    assert old_lin.weight.shape[1] == actor_dim, \
        (old_lin.weight.shape, actor_dim)
    assert new_lin.weight.shape[0] == old_lin.weight.shape[0] and \
        new_lin.weight.shape[1] > actor_dim, \
        (new_lin.weight.shape, old_lin.weight.shape)
    new_lin.weight.data.zero_()
    new_lin.weight.data[:, :actor_dim].copy_(old_lin.weight.data)
    new_lin.bias.data.copy_(old_lin.bias.data)
    _copy_same_graph(old_mods[2:], new_mods[2:])


def asymmetrize_model(old_model, cfg, tol: float = 1e-5, n_check: int = 256):
    """Build a new-layout model computing the same function as old_model,
    verifying max output difference on random inputs before returning it."""
    assert cfg.algorithm == "ppo", "asymmetric migration is PPO-only"
    algo_cls, _, _ = ALGORITHM_REGISTRY[cfg.algorithm]
    assert isinstance(old_model, algo_cls), (type(old_model), cfg.algorithm)

    actor_dim = actor_obs_dim_for(cfg)
    assert actor_dim is not None, f"env {cfg.env_name} has no privileged tail"
    old_obs_dim = old_model.observation_space.shape[0]
    assert old_obs_dim == actor_dim, (
        f"checkpoint obs dim {old_obs_dim} must equal the target actor block "
        f"{actor_dim} — this migration only appends the privileged tail")
    full_dim = actor_dim + priv_dim_for(bool(cfg.marker_include_rot))

    assert cfg.env_name in ("lift", "pickplace"), \
        "single-task migration only (multitask shares the layout; migrate via " \
        "either task)"
    n_substeps = int(cfg[f"{cfg.env_name}_env"].n_substeps)
    obs_norm = obs_norm_for(cfg, n_substeps)
    assert len(obs_norm[0]) == full_dim, (len(obs_norm[0]), full_dim)

    obs_high = np.full(full_dim, np.inf, dtype=np.float32)
    obs_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
    env = DummyVecEnv([lambda: _SpacesEnv(obs_space, old_model.action_space)])
    net_arch = list(old_model.policy_kwargs["net_arch"])
    new_model = build_fresh_model(cfg, env, obs_norm, net_arch, cfg.seed,
                                  actor_obs_dim=actor_dim, verbose=0)

    old_p, new_p = old_model.policy, new_model.policy
    old_p.to("cpu")
    new_p.to("cpu")
    old_p.set_training_mode(False)
    new_p.set_training_mode(False)
    with torch.no_grad():
        _copy_actor(old_p.mlp_extractor.policy_net, new_p.mlp_extractor.policy_net,
                    actor_dim)
        _copy_critic(old_p.mlp_extractor.value_net, new_p.mlp_extractor.value_net,
                     actor_dim)
        _copy_same_graph([old_p.action_net], [new_p.action_net])
        _copy_same_graph([old_p.value_net], [new_p.value_net])
        new_p.log_std.data.copy_(old_p.log_std.data)

        rng = np.random.default_rng(0)
        obs_t = torch.as_tensor(
            (rng.standard_normal((n_check, full_dim)) * 2.0).astype(np.float32))
        old_obs_t = obs_t[:, :actor_dim]
        d_old, d_new = old_p.get_distribution(old_obs_t), new_p.get_distribution(obs_t)
        diff = max(
            float((d_old.distribution.mean - d_new.distribution.mean).abs().max()),
            float((d_old.distribution.stddev - d_new.distribution.stddev).abs().max()),
            float((old_p.predict_values(old_obs_t)
                   - new_p.predict_values(obs_t)).abs().max()))
    if diff > tol:
        raise RuntimeError(
            f"migration not function-preserving: max output diff {diff:.2e} > tol {tol:.0e}")
    # The surgery ran on CPU; put the policy back where the model expects it.
    new_p.to(new_model.device)
    print(f"asymmetrized obs {old_obs_dim} -> {full_dim} "
          f"(actor slice {actor_dim})  |  max output diff {diff:.2e}")
    return new_model


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="trained symmetric .zip checkpoint (old obs layout)")
    p.add_argument("output", help="path for the asymmetric .zip")
    p.add_argument("overrides", nargs="*",
                   help="Hydra overrides selecting the layout, e.g. env=lift")
    args = p.parse_args()

    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=args.overrides)

    algo_cls, _, _ = ALGORITHM_REGISTRY[cfg.algorithm]
    old_model = algo_cls.load(args.input, device="cpu")
    new_model = asymmetrize_model(old_model, cfg)
    new_model.save(args.output)
    print(f"saved asymmetric checkpoint -> {args.output}\n"
          f"fine-tune with:  python -m src.train env={cfg.env_name} resume={args.output}")


if __name__ == "__main__":
    main()
