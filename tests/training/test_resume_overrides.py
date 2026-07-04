"""Resume must re-apply Hydra-owned hyperparameters onto a loaded checkpoint.

SB3's load() restores every hyperparameter from the zip; train.py passes
resume_overrides(cfg) as load(**kwargs), which SB3 applies on top of the
restored attributes before _setup_model(). These tests prove the overrides
land — including the values _setup_model() post-processes (lr schedule,
clip-range wrapper, rollout-buffer size) — while the policy weights still
come from the checkpoint, and that an update still runs afterwards.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from src.train import resume_overrides


class DummyEnv(gym.Env):
    """Minimal env for exercising PPO save/load."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def _resume_cfg():
    """Config whose every tunable differs from the checkpoint's values below."""
    return OmegaConf.create({
        "algorithm": "ppo",
        "train": {"lr_schedule": "constant", "learning_rate": 1e-3, "lr_min": 1e-3,
                  "batch_size": 4, "gamma": 0.9},
        "ppo": {"n_steps": 8, "n_epochs": 3, "clip_range": 0.1, "ent_coef": 0.02,
                "gae_lambda": 0.8, "vf_coef": 0.5},
    })


def test_resume_overrides_win_over_checkpoint(tmp_path):
    model = PPO("MlpPolicy", DummyEnv(), n_steps=16, batch_size=8, gamma=0.95,
                learning_rate=3e-4, n_epochs=5, clip_range=0.2, ent_coef=0.001,
                gae_lambda=0.95, vf_coef=1.0, policy_kwargs={"net_arch": [8]})
    path = tmp_path / "ckpt.zip"
    model.save(path)
    weight_key = "mlp_extractor.policy_net.0.weight"
    weight_saved = model.policy.state_dict()[weight_key].detach().cpu().clone()

    cfg = _resume_cfg()
    loaded = PPO.load(path, env=DummyEnv(), **resume_overrides(cfg))

    # Plain attributes come from the config, not the zip.
    assert loaded.gamma == 0.9
    assert loaded.batch_size == 4
    assert loaded.n_epochs == 3
    assert loaded.ent_coef == 0.02
    assert loaded.gae_lambda == 0.8
    assert loaded.vf_coef == 0.5
    # _setup_model() post-processing rebuilt from the overridden raw values.
    assert loaded.lr_schedule(1.0) == 1e-3
    assert loaded.clip_range(1.0) == 0.1
    assert loaded.n_steps == 8
    assert loaded.rollout_buffer.buffer_size == 8

    # The weights are the checkpoint's, not a re-init.
    assert torch.equal(weight_saved, loaded.policy.state_dict()[weight_key].cpu())

    # One rollout + update runs on the rebuilt buffer/schedules.
    loaded.learn(total_timesteps=8)


def test_resume_overrides_only_tunables():
    """Architecture-affecting keys must never be in the override set: the
    checkpoint's policy_kwargs / spaces have to keep matching the weights."""
    overrides = resume_overrides(_resume_cfg())
    assert set(overrides) == {"learning_rate", "batch_size", "gamma", "n_steps",
                              "n_epochs", "clip_range", "ent_coef", "gae_lambda",
                              "vf_coef"}
