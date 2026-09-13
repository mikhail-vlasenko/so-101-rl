"""Reward scaling survives all checkpoint paths without changing actor inputs."""

import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.reward_norm import (
    BestRewardNormCallback, RewardCheckpointCallback, reward_norm_path,
    save_reward_norm, training_reward_env,
)
import src.train as train_module


class RewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([2, 3], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.array([2, 3], dtype=np.float32), float(self.steps), self.steps == 4, False, {}


def vector_env(n_envs=1):
    return DummyVecEnv([lambda: Monitor(RewardEnv()) for _ in range(n_envs)])


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("checkpoint", [None, "legacy.zip"])
def test_missing_sidecar_starts_defaults(tmp_path, checkpoint, capsys):
    path = None if checkpoint is None else tmp_path / checkpoint
    env = training_reward_env(vector_env(), 0.91, path)
    try:
        assert env.ret_rms.mean == 0
        assert env.ret_rms.var == 1
        assert env.ret_rms.count == 1e-4
        assert env.gamma == 0.91
        assert env.training and env.norm_reward and not env.norm_obs
        np.testing.assert_array_equal(env.reset(), [[2, 3]])
        env.step(np.zeros((1, 1)))
        assert env.ret_rms.count > 1
        if checkpoint is not None:
            assert "starting with default statistics" in capsys.readouterr().out
    finally:
        env.close()


def test_resume_matches_reward_stream_and_critic_targets_after_episode_reset(tmp_path):
    env = training_reward_env(vector_env(), 0.99, None)
    env.reset()
    for _ in range(11):
        env.step(np.zeros((1, 1)))
    model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, device="cpu",
                policy_kwargs={"net_arch": [8]}, seed=7)
    path = tmp_path / "final_model.zip"
    model.save(path)
    save_reward_norm(model, path)
    restored = training_reward_env(vector_env(), 0.99, path)
    loaded = PPO.load(path, env=restored, device="cpu")
    try:
        assert env.returns[0] != 0
        np.testing.assert_array_equal(restored.returns, [0])
        np.testing.assert_array_equal(env.ret_rms.mean, restored.ret_rms.mean)
        np.testing.assert_array_equal(env.ret_rms.var, restored.ret_rms.var)
        assert env.ret_rms.count == restored.ret_rms.count
        np.testing.assert_array_equal(env.reset(), restored.reset())
        for _ in range(12):
            obs, reward, done, _ = env.step(np.zeros((1, 1)))
            resumed_obs, resumed_reward, resumed_done, _ = restored.step(np.zeros((1, 1)))
            np.testing.assert_array_equal(reward, resumed_reward)
            np.testing.assert_array_equal(obs, resumed_obs)
            np.testing.assert_array_equal(done, resumed_done)
            with torch.no_grad():
                value = model.policy.predict_values(torch.as_tensor(obs)).numpy().ravel()
                resumed_value = loaded.policy.predict_values(torch.as_tensor(resumed_obs)).numpy().ravel()
            np.testing.assert_array_equal(
                reward + 0.99 * (~done) * value,
                resumed_reward + 0.99 * (~resumed_done) * resumed_value)
    finally:
        env.close()
        restored.close()


def test_curriculum_resume_retains_statistics_but_updates_gamma_and_worker_count(tmp_path):
    env = training_reward_env(vector_env(), 0.99, None)
    env.reset()
    for _ in range(5):
        env.step(np.zeros((1, 1)))
    path = tmp_path / "curriculum.zip"
    env.save(reward_norm_path(path))
    restored = training_reward_env(vector_env(2), 0.9, path)
    try:
        assert restored.gamma == 0.9
        assert restored.ret_rms.count == env.ret_rms.count
        np.testing.assert_array_equal(restored.ret_rms.var, env.ret_rms.var)
        np.testing.assert_array_equal(restored.returns, [0, 0])
        restored.reset()
        restored.step(np.zeros((2, 1)))
        assert restored.ret_rms.count == env.ret_rms.count + 2
    finally:
        env.close()
        restored.close()


def test_best_periodic_and_final_save_matching_training_statistics(tmp_path):
    env = training_reward_env(vector_env(), 0.99, None)
    eval_env = VecNormalize(vector_env(), norm_obs=False, norm_reward=False, training=False)
    model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, n_epochs=1,
                device="cpu", policy_kwargs={"net_arch": [8]}, seed=7)
    callbacks = [
        RewardCheckpointCallback(save_freq=8, save_path=str(tmp_path), name_prefix="ppo"),
        EvalCallback(eval_env, eval_freq=4, n_eval_episodes=1,
                     best_model_save_path=str(tmp_path), verbose=0,
                     callback_on_new_best=BestRewardNormCallback(tmp_path / "best_model.zip")),
    ]
    try:
        model.learn(12, callback=callbacks)
        model.save(tmp_path / "final_model")
        save_reward_norm(model, tmp_path / "final_model")
        for name, samples in [("best_model", 4), ("ppo_8_steps", 8), ("final_model", 12)]:
            path = tmp_path / f"{name}.zip"
            assert path.exists()
            assert reward_norm_path(path).exists()
            restored = training_reward_env(vector_env(), 0.99, path)
            try:
                assert restored.ret_rms.count == pytest.approx(samples + 1e-4)
                assert not restored.norm_obs
            finally:
                restored.close()
        assert not eval_env.norm_reward
        assert not eval_env.training
    finally:
        env.close()
        eval_env.close()


def test_present_corrupt_sidecar_does_not_silently_reset(tmp_path):
    path = tmp_path / "broken.zip"
    reward_norm_path(path).write_bytes(b"not a pickle")
    vec = vector_env()
    try:
        with pytest.raises(pickle.UnpicklingError):
            training_reward_env(vec, 0.99, path)
    finally:
        vec.close()


def test_running_observation_normalization_is_rejected(tmp_path):
    path = tmp_path / "wrong_obs.zip"
    env = VecNormalize(vector_env(), norm_obs=True)
    env.save(reward_norm_path(path))
    vec = vector_env()
    try:
        with pytest.raises(AssertionError, match="fixed policy observation"):
            training_reward_env(vec, 0.99, path)
    finally:
        env.close()
        vec.close()


def test_training_entrypoint_resumes_statistics_and_explicit_seed(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "dr=none", "shaping=none", "wandb.enabled=false",
            "seed=7", "train.n_envs=1", "train.net_arch=[8]",
            "train.total_timesteps=8", "train.time_limit_minutes=null",
            "ppo.n_steps=4", "ppo.n_epochs=1", "train.batch_size=4",
            "train.checkpoint_freq=4", "train.eval_freq=4", "train.n_eval_episodes=1",
            "lift_env.max_steps=2", "train.run_name=first",
        ])
    # Isolate all training outputs while retaining the real scene/includes.
    (tmp_path / "so101").symlink_to(repo / "so101", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_module.hydra.utils, "get_original_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(train_module, "SubprocVecEnv", DummyVecEnv)
    train_module.train(cfg)
    first = tmp_path / "logs/ppo_lift/first"
    for name in ("final_model.zip", "best_model.zip", "checkpoints/ppo_4_steps.zip"):
        assert (first / name).exists()
        assert reward_norm_path(first / name).exists()

    cfg.resume = str(first / "final_model.zip")
    cfg.seed = 11
    cfg.train.gamma = 0.9
    cfg.train.run_name = "resumed"
    train_module.train(cfg)
    final = tmp_path / "logs/ppo_lift/resumed/final_model.zip"
    model = PPO.load(final, device="cpu")
    assert model.seed == 11
    assert model.gamma == 0.9
    with reward_norm_path(final).open("rb") as handle:
        normalizer = pickle.load(handle)
    assert normalizer.ret_rms.count == pytest.approx(16 + 1e-4)
    assert normalizer.gamma == 0.9
    assert not normalizer.norm_obs
