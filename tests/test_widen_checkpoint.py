"""Contract for src.widen_checkpoint: a Net2WiderNet widening must (a) preserve
the policy's function (nearly) at init so it's a real headstart, and (b) produce
a *non-degenerate* wider net -- the replicated units must be distinct, so the
extra width is usable rather than a redundant copy of the original network.

The non-degeneracy point has teeth because of Adam: a weight row's gradient is
always a scalar times the shared layer input, so two *tied* copies receive
near-identical Adam updates and never separate on their own (verified: exact
replicas stay bitwise-identical through training). Distinctness therefore has to
be seeded at init by `noise`; `noise=0` reproduces the degenerate exact copy.
"""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.networks import LayerNormActorCriticPolicy, LayerNormSACPolicy
from src.widen_checkpoint import (
    _SpacesEnv, _max_output_diff, widen_checkpoint, widen_model,
)


def _spaces(obs_dim, act_dim):
    return (Box(-np.inf, np.inf, (obs_dim,), np.float32),
            Box(-1.0, 1.0, (act_dim,), np.float32))


def _ppo(obs_dim, act_dim, arch):
    env = DummyVecEnv([lambda: _SpacesEnv(*_spaces(obs_dim, act_dim))])
    return PPO(LayerNormActorCriticPolicy, env, n_steps=8, batch_size=8, seed=0, device="cpu",
               policy_kwargs={"net_arch": list(arch),
                              "obs_norm": ([0.0] * obs_dim, [1.0] * obs_dim)})


def _sac(obs_dim, act_dim, arch):
    env = DummyVecEnv([lambda: _SpacesEnv(*_spaces(obs_dim, act_dim))])
    return SAC(LayerNormSACPolicy, env, buffer_size=1, learning_starts=0, seed=0, device="cpu",
               policy_kwargs={"net_arch": list(arch),
                              "obs_norm": ([0.0] * obs_dim, [1.0] * obs_dim)})


def _first_linear(seq):
    return next(m for m in seq if isinstance(m, torch.nn.Linear))


def _hidden_widths(seq):
    return [m.out_features for m in seq if isinstance(m, torch.nn.Linear)]


def _min_copy_gap(model):
    """Largest incoming-weight difference between a first-hidden-layer unit and
    its replica (0 iff the copies are identical, i.e. degenerate)."""
    w = _first_linear(model.policy.mlp_extractor.policy_net).weight.detach()
    half = w.shape[0] // 2
    return (w[:half] - w[half:]).abs().max().item()


def _train_steps(model, obs_dim, n=25):
    model.policy.set_training_mode(True)
    rng = np.random.default_rng(4)
    for _ in range(n):
        obs = torch.as_tensor((rng.standard_normal((64, obs_dim)) * 2).astype(np.float32))
        loss = (model.policy.get_distribution(obs).distribution.mean.pow(2).sum()
                + model.policy.predict_values(obs).pow(2).sum())
        model.policy.optimizer.zero_grad()
        loss.backward()
        model.policy.optimizer.step()


def _assert_preserved(algorithm, old_model, new_model, obs_dim, tol):
    rng = np.random.default_rng(1)
    obs_t = torch.as_tensor((rng.standard_normal((128, obs_dim)) * 2).astype(np.float32))
    old_model.policy.set_training_mode(False)
    new_model.policy.set_training_mode(False)
    with torch.no_grad():
        diff = _max_output_diff(algorithm, old_model.policy, new_model.policy, obs_t)
    assert diff < tol, diff


# --- widths / preservation -------------------------------------------------

def test_widths_and_default_doubling():
    old = _ppo(9, 3, [16, 24])
    new = widen_model(old, "ppo", None)                    # default = 2x every layer
    assert _hidden_widths(new.policy.mlp_extractor.policy_net) == [32, 48]
    assert _hidden_widths(new.policy.mlp_extractor.value_net) == [32, 48]


def test_default_noise_preserves_function():
    obs_dim = 11
    old = _ppo(obs_dim, 4, [16, 16, 16])
    new = widen_model(old, "ppo", [32, 32, 32])            # default noise
    _assert_preserved("ppo", old, new, obs_dim, tol=1e-2)  # headstart intact


def test_zero_noise_is_exact():
    obs_dim = 11
    old = _ppo(obs_dim, 4, [16, 16, 16])
    new = widen_model(old, "ppo", [32, 32, 32], noise=0.0)
    _assert_preserved("ppo", old, new, obs_dim, tol=1e-4)  # exact replication


def test_sac_default_noise_preserves_function():
    obs_dim = 10
    old = _sac(obs_dim, 3, [16, 16])
    new = widen_model(old, "sac", [32, 32])
    assert _hidden_widths(new.policy.actor.latent_pi) == [32, 32]
    _assert_preserved("sac", old, new, obs_dim, tol=1e-2)


def test_per_layer_integer_factors():
    obs_dim = 9
    old = _ppo(obs_dim, 3, [8, 8])
    new = widen_model(old, "ppo", [24, 16])                # 3x then 2x
    assert _hidden_widths(new.policy.mlp_extractor.policy_net) == [24, 16]
    _assert_preserved("ppo", old, new, obs_dim, tol=1e-2)


# --- non-degeneracy (the point of `noise`) ---------------------------------

def test_default_noise_seeds_distinct_copies():
    old = _ppo(11, 4, [16, 16])
    new = widen_model(old, "ppo", [32, 32])
    assert _min_copy_gap(new) > 1e-3, "replicas must start distinct, not tied"


def test_zero_noise_makes_identical_copies():
    old = _ppo(11, 4, [16, 16])
    new = widen_model(old, "ppo", [32, 32], noise=0.0)
    assert _min_copy_gap(new) == 0.0, "noise=0 must reproduce exact (degenerate) replicas"


def test_non_degenerate_after_updates():
    """After training, the seeded net keeps distinct units (uses its width) while
    the exact-replica net stays bitwise-tied -- a degenerate wide copy of the
    original. This is what `noise` buys and what a widened checkpoint must have."""
    obs_dim = 11
    seeded = widen_model(_ppo(obs_dim, 4, [16, 16]), "ppo", [32, 32])
    exact = widen_model(_ppo(obs_dim, 4, [16, 16]), "ppo", [32, 32], noise=0.0)

    gap_before = _min_copy_gap(seeded)
    _train_steps(seeded, obs_dim)
    _train_steps(exact, obs_dim)

    assert _min_copy_gap(seeded) > 1e-3, "seeded copies collapsed to a degenerate net"
    assert abs(_min_copy_gap(seeded) - gap_before) > 0, "network did not train"
    assert _min_copy_gap(exact) < 1e-12, "exact replicas must stay tied under Adam"


# --- rejections ------------------------------------------------------------

def test_rejects_non_integer_multiple():
    with pytest.raises(ValueError):
        widen_model(_ppo(9, 3, [16]), "ppo", [24])         # 24 / 16 = 1.5


def test_rejects_depth_change():
    with pytest.raises(ValueError):
        widen_model(_ppo(9, 3, [16, 16]), "ppo", [32, 32, 32])


# --- end-to-end zip round-trip ---------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    obs_dim = 11
    old = _ppo(obs_dim, 4, [16, 16])
    in_path, out_path = tmp_path / "old.zip", tmp_path / "wide.zip"
    old.save(str(in_path))
    widen_checkpoint(str(in_path), str(out_path), "ppo", [32, 32],
                     noise=3e-2, seed=0, tol=0.1, device="cpu")
    reloaded = PPO.load(str(out_path), device="cpu")
    assert _hidden_widths(reloaded.policy.mlp_extractor.policy_net) == [32, 32]
    assert _min_copy_gap(reloaded) > 1e-3
    _assert_preserved("ppo", old, reloaded, obs_dim, tol=1e-2)
