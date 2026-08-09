"""Lag-tap observation history contract (src/obs_history.py +
SO101BaseEnv._serve_obs).

A tap-indexing or reset-convention mismatch between training and the real
rollout scripts is invisible at deploy — the policy just silently reads
shifted history — so these tests pin the shared convention both sides use:
tap ordering (newest first), reset padding with the boot frame, ring
wraparound past the deepest lag, and the env layout
[state block per tap | current BPS block | single privileged tail].
"""

from dataclasses import replace

import numpy as np
import pytest
from hydra import compose, initialize

from src.lift_env import SO101LiftEnv
from src.obs_history import ObsHistory
from src.bps import BPS_OBS_DIM
from src.train import runtime_cfg_from_hydra


# ---------------------------------------------------------------- class contract


def test_taps_match_naive_reference():
    """Every tapped frame equals the naive 'keep all frames' lookup, with the
    boot frame standing in for pre-reset history — through several ring
    wraparounds past the deepest lag."""
    taps = (0, 2, 5)
    dim = 3
    hist = ObsHistory(taps, dim)
    rng = np.random.default_rng(0)
    frames = [rng.standard_normal(dim).astype(np.float32)]
    hist.reset(frames[0])
    for i in range(1, 4 * (taps[-1] + 1)):
        frames.append(rng.standard_normal(dim).astype(np.float32))
        tapped = hist.push(frames[i])
        expected = np.concatenate([frames[max(0, i - t)] for t in taps])
        np.testing.assert_array_equal(tapped, expected)


def test_reset_pads_with_boot_frame():
    hist = ObsHistory((0, 1, 4), 2)
    f0 = np.array([1.0, -2.0], np.float32)
    np.testing.assert_array_equal(hist.reset(f0), np.tile(f0, 3))
    # A reset after use re-pads — no leakage from the previous episode.
    hist.push(np.array([9.0, 9.0], np.float32))
    f1 = np.array([3.0, 4.0], np.float32)
    np.testing.assert_array_equal(hist.reset(f1), np.tile(f1, 3))


def test_single_tap_is_identity():
    hist = ObsHistory((0,), 4)
    f = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal(hist.reset(f), f)
    g = f + 1
    np.testing.assert_array_equal(hist.push(g), g)


def test_rejects_bad_taps():
    with pytest.raises(AssertionError):
        ObsHistory((1, 2), 3)     # must start at 0
    with pytest.raises(AssertionError):
        ObsHistory((0, 4, 4), 3)  # strictly ascending
    with pytest.raises(AssertionError):
        ObsHistory((0, 4, 2), 3)
    with pytest.raises(AssertionError):
        ObsHistory((), 3)


# ---------------------------------------------------------------- env integration


TAPS = (0, 2, 5)


def _lift_env(cfg, taps):
    runtime = replace(runtime_cfg_from_hydra(cfg), history_taps=taps)
    return SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                        cfg=runtime)


def test_env_serves_tapped_actor_blocks_and_single_tail():
    """A tapped env against an identically-seeded taps=(0,) twin: tap 0 and
    the tail must match the twin's obs exactly, and each deeper tap must be
    the twin's actor block from that many ticks ago (boot frame before
    that) — under full DR, so the held/noisy tag path is exercised too."""
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    env1 = _lift_env(cfg, (0,))
    envN = _lift_env(cfg, TAPS)
    state_dim = env1.state_dim
    assert envN.observation_space.shape == (state_dim * len(TAPS) + BPS_OBS_DIM
                                             + envN.priv_dim,)

    obs1, _ = env1.reset(seed=5)
    obsN, _ = envN.reset(seed=5)
    frames = [obs1[:state_dim]]
    np.testing.assert_array_equal(
        obsN, np.concatenate([np.tile(frames[0], len(TAPS)), obs1[state_dim:]]))

    rng = np.random.default_rng(0)
    for i in range(1, 3 * (TAPS[-1] + 1)):
        action = rng.uniform(-1.0, 1.0, 6).astype(np.float32)
        obs1, _, term1, trunc1, _ = env1.step(action)
        obsN, _, termN, truncN, _ = envN.step(action)
        assert (term1, trunc1) == (termN, truncN)
        frames.append(obs1[:state_dim])
        expected = np.concatenate(
            [frames[max(0, i - t)] for t in TAPS] + [obs1[state_dim:]])
        np.testing.assert_array_equal(obsN, expected)
        if term1 or trunc1:
            break
    env1.close()
    envN.close()
