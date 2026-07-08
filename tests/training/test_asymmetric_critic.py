"""Contract for the asymmetric critic: obs = [actor block | privileged tail].

Three things must be exactly right, and none of them crash when wrong:
the tail must carry the true state and the episode's sampled DR latents
(garbage here silently degrades the value function it exists to help), the
actor must be *structurally* unable to read it (a leak deploys a policy that
trained on ground truth the real rig cannot provide), and the
function-preserving migration (src.asymmetrize_checkpoint) must reproduce an
old checkpoint's outputs bit-for-bit while widening the critic input.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces
from hydra import compose, initialize
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.asymmetrize_checkpoint import _SpacesEnv, asymmetrize_model
from src.base_env import RuntimeEnvConfig, priv_dim_for
from src.lift_env import SO101LiftEnv
from src.networks import LayerNormActorCriticPolicy, TakeFirst
from src.train import (
    actor_obs_dim_for, build_fresh_model, obs_norm_for, runtime_cfg_from_hydra,
)

# Privileged tail layout for marker_include_rot=False (base_env._priv_tail),
# relative to the tail start.
CUBE_POS = slice(0, 3)
CUBE_ROT = slice(3, 6)
CUBE_VEL = slice(6, 12)
JAW_FLAGS = slice(12, 14)
QPOS_BIAS = slice(14, 20)
MARKER_POS_BIAS = slice(20, 26)
CUBE_BIAS = slice(26, 29)
CUBE_ROT_BIAS = slice(29, 32)
COMMON_BIAS = slice(32, 35)
CAM_DELAY = 35
HALF_EXTENTS = slice(36, 39)


@pytest.fixture(scope="module")
def lift_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=[
            "env=lift", "wandb.enabled=false", "train.n_envs=2",
            "ppo.n_steps=8", "train.net_arch=[32,32]",
        ])


def _lift_env(cfg):
    return SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                        cfg=runtime_cfg_from_hydra(cfg))


def _model(cfg, seed=0):
    venv = DummyVecEnv([lambda: _lift_env(cfg)])
    return build_fresh_model(cfg, venv, obs_norm_for(cfg, n_substeps=10),
                             [32, 32], seed=seed,
                             actor_obs_dim=actor_obs_dim_for(cfg), verbose=0)


# --------------------------------------------------------------------------
# Tail contents
# --------------------------------------------------------------------------

def test_priv_tail_carries_truth_and_latents(lift_cfg):
    """Under full DR, every tail channel equals the env's own ground truth /
    sampled latents at the obs instant."""
    env = _lift_env(lift_cfg)  # composed default: dr=full
    obs, _ = env.reset(seed=3)
    assert obs.shape == (env.obs_dim + env.priv_dim,)
    for _ in range(5):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert not (term or trunc)
    tail = obs[env.obs_dim:].astype(np.float64)

    np.testing.assert_allclose(tail[CUBE_POS], env._get_cube_pos(), atol=1e-6)
    np.testing.assert_allclose(
        tail[CUBE_VEL],
        env.data.qvel[env.cube_dofadr:env.cube_dofadr + 6], atol=1e-5)
    assert set(np.unique(tail[JAW_FLAGS])) <= {0.0, 1.0}
    # DR latents: the episode's sampled biases, verbatim (nonzero under dr=full,
    # so these comparisons are meaningful).
    assert np.linalg.norm(env._qpos_bias) > 0.0
    np.testing.assert_allclose(tail[QPOS_BIAS], env._qpos_bias, atol=1e-6)
    np.testing.assert_allclose(tail[MARKER_POS_BIAS],
                               env._marker_pos_bias.flatten(), atol=1e-7)
    np.testing.assert_allclose(tail[CUBE_BIAS], env._cube_bias, atol=1e-7)
    np.testing.assert_allclose(tail[CUBE_ROT_BIAS], env._cube_rot_bias, atol=1e-6)
    np.testing.assert_allclose(tail[COMMON_BIAS], env._common_pos_bias, atol=1e-7)
    assert tail[CAM_DELAY] == pytest.approx(env._camera.pipeline_delay_s, abs=1e-6)
    assert 0.042 <= tail[CAM_DELAY] <= 0.052  # dr=full delay range
    np.testing.assert_allclose(tail[HALF_EXTENTS], env.cube_half_extents, atol=1e-6)


def test_priv_tail_identical_in_both_views(lift_cfg):
    """privileged_obs (the distillation teacher view) differs from the student
    obs only in the tag channels; the tail is ground truth either way."""
    env = _lift_env(lift_cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset(seed=1)
    priv = env.privileged_obs()
    assert np.array_equal(obs[env.obs_dim:], priv[env.obs_dim:])


# --------------------------------------------------------------------------
# Structural actor slice
# --------------------------------------------------------------------------

def test_actor_is_blind_to_tail_but_critic_is_not(lift_cfg):
    model = _model(lift_cfg)
    policy = model.policy
    policy.set_training_mode(False)
    actor_dim = actor_obs_dim_for(lift_cfg)
    full_dim = model.observation_space.shape[0]
    assert actor_dim < full_dim

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((16, full_dim)).astype(np.float32)
    obs_tail_perturbed = obs.copy()
    obs_tail_perturbed[:, actor_dim:] = rng.standard_normal(
        (16, full_dim - actor_dim)).astype(np.float32)

    dev = policy.device
    with torch.no_grad():
        d1 = policy.get_distribution(torch.as_tensor(obs, device=dev))
        d2 = policy.get_distribution(torch.as_tensor(obs_tail_perturbed, device=dev))
        v1 = policy.predict_values(torch.as_tensor(obs, device=dev))
        v2 = policy.predict_values(torch.as_tensor(obs_tail_perturbed, device=dev))
    torch.testing.assert_close(d1.distribution.mean, d2.distribution.mean)
    assert not torch.allclose(v1, v2), "critic ignored the privileged tail"

    # The predict path (deployment) is equally blind to the tail.
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = model.predict(obs_tail_perturbed, deterministic=True)
    np.testing.assert_array_equal(a1, a2)


def test_actor_slice_rides_through_checkpoint(lift_cfg, tmp_path):
    model = _model(lift_cfg)
    path = tmp_path / "asym.zip"
    model.save(path)
    loaded = PPO.load(path)
    assert loaded.policy_kwargs["actor_obs_dim"] == actor_obs_dim_for(lift_cfg)
    first = loaded.policy.mlp_extractor.policy_net[0]
    assert isinstance(first, TakeFirst) and first.n == actor_obs_dim_for(lift_cfg)
    obs = np.random.default_rng(1).standard_normal(
        (4, model.observation_space.shape[0])).astype(np.float32)
    a, _ = model.predict(obs, deterministic=True)
    a_loaded, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_array_equal(a, a_loaded)


# --------------------------------------------------------------------------
# Function-preserving migration of old (symmetric) checkpoints
# --------------------------------------------------------------------------

def _old_layout_model(lift_cfg, actor_dim):
    """A checkpoint as the pre-asymmetric code built it: obs = bare actor
    block, symmetric nets, actor-block obs_norm."""
    full_norm = obs_norm_for(lift_cfg, n_substeps=10)
    old_norm = (list(full_norm[0][:actor_dim]), list(full_norm[1][:actor_dim]))
    obs_high = np.full(actor_dim, np.inf, dtype=np.float32)
    env = DummyVecEnv([lambda: _SpacesEnv(
        spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32))])
    return PPO(LayerNormActorCriticPolicy, env, n_steps=8, batch_size=8,
               policy_kwargs={"net_arch": [32, 32], "obs_norm": old_norm})


def test_asymmetrize_preserves_function_and_resumes(lift_cfg, tmp_path):
    actor_dim = actor_obs_dim_for(lift_cfg)
    old = _old_layout_model(lift_cfg, actor_dim)
    new = asymmetrize_model(old, lift_cfg)

    full_dim = actor_dim + priv_dim_for(bool(lift_cfg.marker_include_rot))
    assert new.observation_space.shape == (full_dim,)
    assert new.policy_kwargs["actor_obs_dim"] == actor_dim

    rng = np.random.default_rng(2)
    obs = (rng.standard_normal((32, full_dim)) * 2.0).astype(np.float32)
    old.policy.set_training_mode(False)
    new.policy.set_training_mode(False)
    old_dev, new_dev = old.policy.device, new.policy.device
    with torch.no_grad():
        d_old = old.policy.get_distribution(
            torch.as_tensor(obs[:, :actor_dim], device=old_dev))
        d_new = new.policy.get_distribution(torch.as_tensor(obs, device=new_dev))
        v_old = old.policy.predict_values(
            torch.as_tensor(obs[:, :actor_dim], device=old_dev))
        v_new = new.policy.predict_values(torch.as_tensor(obs, device=new_dev))
    torch.testing.assert_close(d_old.distribution.mean.cpu(),
                               d_new.distribution.mean.cpu(), atol=1e-6, rtol=0)
    torch.testing.assert_close(d_old.distribution.stddev.cpu(),
                               d_new.distribution.stddev.cpu(), atol=1e-6, rtol=0)
    torch.testing.assert_close(v_old.cpu(), v_new.cpu(), atol=1e-5, rtol=0)

    # The saved artifact resume-loads against the real (new-layout) env and the
    # critic's zeroed privileged columns receive gradient from the first update.
    path = tmp_path / "migrated.zip"
    new.save(path)
    venv = DummyVecEnv([lambda: _lift_env(lift_cfg)])
    resumed = PPO.load(path, env=venv)
    vf_first = resumed.policy.mlp_extractor.value_net[1]
    priv_cols_before = vf_first.weight.data[:, actor_dim:].clone()
    assert torch.count_nonzero(priv_cols_before) == 0
    resumed.learn(total_timesteps=16)
    assert torch.count_nonzero(
        resumed.policy.mlp_extractor.value_net[1].weight.data[:, actor_dim:]) > 0, \
        "privileged critic columns received no gradient"
    venv.close()


def test_asymmetrize_rejects_wrong_obs_dim(lift_cfg):
    """A checkpoint whose obs is not exactly the actor block (e.g. already
    migrated, or a different layout) must fail loud."""
    actor_dim = actor_obs_dim_for(lift_cfg)
    wrong = _old_layout_model(lift_cfg, actor_dim + 1)
    with pytest.raises(AssertionError, match="actor block"):
        asymmetrize_model(wrong, lift_cfg)
