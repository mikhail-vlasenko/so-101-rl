"""Fixed affine observation normalization (src/obs_norm.py + networks.ObsNorm).

The constants must cover the exact obs layout, keep real rollout data within a
few normalized units (this is the test that keeps the hand-set workspace boxes
honest), and the ObsNorm layer must behave like a constant: identical in train
and eval mode, no trainable parameters, and bit-stable through a checkpoint
save/load — the properties input BatchNorm lacked.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from hydra import compose, initialize
from stable_baselines3 import PPO, SAC

from src.base_env import obs_dim_for, priv_dim_for, state_dim_for
from src.lift_env import SO101LiftEnv
from src.networks import LayerNormActorCriticPolicy, LayerNormSACPolicy, ObsNorm
from src.obs_norm import build_obs_norm, build_reach_obs_norm
from src.pickplace_env import SO101PickPlaceEnv
from src.train import obs_norm_for, runtime_cfg_from_hydra

ACTION_SCALE = 0.07
N_SUBSTEPS = 10


def _compose(env_name):
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=[f"env={env_name}"])


# ---------------------------------------------------------------- constants


def test_shapes_match_obs_layout():
    for prev_n in (0, 1, 2):
        for include_rot in (False, True):
            center, scale = build_obs_norm(prev_n, include_rot, ACTION_SCALE, N_SUBSTEPS)
            expected = obs_dim_for(prev_n, include_rot) + priv_dim_for(include_rot)
            assert center.shape == scale.shape == (expected,)
            assert np.all(scale > 0)
            assert np.all(np.isfinite(center)) and np.all(np.isfinite(scale))


def test_obs_norm_for_tiles_state_block_per_tap():
    """Normalization repeats state constants per tap and keeps the current BPS
    block and critic tail single-copy."""
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config",
                      overrides=["env=lift", "history_taps=[0,4,16,48]"])
    n_substeps = int(cfg.lift_env.n_substeps)
    center, scale = obs_norm_for(cfg, n_substeps)
    state_dim = state_dim_for(int(cfg.prev_actions_n), bool(cfg.marker_include_rot))
    single_c, single_s = build_obs_norm(int(cfg.prev_actions_n),
                                        bool(cfg.marker_include_rot),
                                        float(cfg.action_scale), n_substeps)
    assert center == single_c[:state_dim].tolist() * 4 + single_c[state_dim:].tolist()
    assert scale == single_s[:state_dim].tolist() * 4 + single_s[state_dim:].tolist()


def test_reach_shapes():
    n_waypoints = 3
    center, scale = build_reach_obs_norm(2, n_waypoints, ACTION_SCALE, N_SUBSTEPS)
    expected = 2 * 6 + 3 + n_waypoints + 2 * 6
    assert center.shape == scale.shape == (expected,)
    assert np.all(scale > 0)


@pytest.mark.parametrize("env_name,env_cls,cfg_key,xml", [
    ("lift", SO101LiftEnv, "lift_env", "so101/scene_lift.xml"),
    ("pickplace", SO101PickPlaceEnv, "pickplace_env", "so101/scene_pickplace.xml"),
])
def test_rollout_obs_within_bounds(env_name, env_cls, cfg_key, xml):
    """Random-action episodes under full DR: every normalized obs dimension
    stays within a few units. If this fails after a workspace/task change, the
    boxes in src/obs_norm.py need re-centering."""
    cfg = _compose(env_name)
    env = env_cls(env_cfg=cfg[cfg_key], xml_path=xml, cfg=runtime_cfg_from_hydra(cfg))
    center, scale = build_obs_norm(int(cfg.prev_actions_n), False,
                                   float(cfg.action_scale), int(cfg[cfg_key].n_substeps))
    assert center.shape == (env.obs_dim + env.priv_dim,)

    worst = 0.0
    obs, _ = env.reset(seed=0)
    for i in range(400):
        z = (obs - center) / scale
        worst = max(worst, float(np.abs(z).max()))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset(seed=i)
    assert worst < 4.0, f"normalized obs reached {worst:.2f}; re-center src/obs_norm.py"


# ---------------------------------------------------------------- ObsNorm layer


def test_obsnorm_is_a_constant_affine():
    center = np.array([1.0, -2.0, 0.5])
    scale = np.array([2.0, 0.5, 1.0])
    layer = ObsNorm(center, scale)
    x = torch.tensor([[3.0, -1.0, 0.5], [1.0, -2.0, -0.5]])
    expected = (x - torch.tensor(center, dtype=torch.float32)) / \
        torch.tensor(scale, dtype=torch.float32)
    torch.testing.assert_close(layer(x), expected)
    # Mode-independent (the BatchNorm failure mode) and nothing to optimize.
    layer.train()
    y_train = layer(x)
    layer.eval()
    torch.testing.assert_close(y_train, layer(x))
    assert list(layer.parameters()) == []
    # Batch of one works in train mode — BatchNorm1d would raise here.
    layer.train()
    layer(x[:1])


def test_obsnorm_rejects_bad_scale():
    with pytest.raises(AssertionError):
        ObsNorm(np.zeros(3), np.array([1.0, 0.0, 1.0]))


# ---------------------------------------------------------------- SB3 integration


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


NORM4 = ([0.5, -0.5, 0.0, 2.0], [2.0, 1.0, 0.5, 4.0])


def test_ppo_policy_checkpoint_roundtrip(tmp_path):
    """obs_norm buffers ride in policy_kwargs + state_dict: a loaded checkpoint
    normalizes identically with no sidecar file."""
    model = PPO(LayerNormActorCriticPolicy, DummyEnv(), n_steps=8, batch_size=4,
                policy_kwargs={"net_arch": [8], "obs_norm": NORM4})
    norm = model.policy.mlp_extractor.policy_net[0]
    assert isinstance(norm, ObsNorm)
    obs = np.array([0.3, -0.9, 0.1, 1.5], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)

    path = tmp_path / "ckpt.zip"
    model.save(path)
    loaded = PPO.load(path, env=DummyEnv())
    loaded_norm = loaded.policy.mlp_extractor.policy_net[0]
    torch.testing.assert_close(loaded_norm.center, norm.center)
    torch.testing.assert_close(loaded_norm.scale, norm.scale)
    action_loaded, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_array_equal(action, action_loaded)


def test_sac_critic_extends_norm_with_action_dims():
    """The SAC critic input is [obs, action]; its ObsNorm must cover obs dims
    with the constants and pass the (already [-1,1]) action dims through."""
    model = SAC(LayerNormSACPolicy, DummyEnv(), buffer_size=100,
                policy_kwargs={"net_arch": [8], "obs_norm": NORM4})
    actor_norm = model.policy.actor.latent_pi[0]
    critic_norm = model.policy.critic.qf0[0]
    assert isinstance(actor_norm, ObsNorm) and isinstance(critic_norm, ObsNorm)
    assert actor_norm.center.shape == (4,)
    assert critic_norm.center.shape == (6,)
    torch.testing.assert_close(critic_norm.center[:4], actor_norm.center)
    torch.testing.assert_close(critic_norm.center[4:], torch.zeros_like(critic_norm.center[4:]))
    torch.testing.assert_close(critic_norm.scale[4:], torch.ones_like(critic_norm.scale[4:]))
    # Forward passes run through the norm on both heads.
    obs = torch.zeros(1, 4, device=model.device)
    act = torch.zeros(1, 2, device=model.device)
    q_values = model.policy.critic(obs, act)
    assert all(q.shape == (1, 1) for q in q_values)
