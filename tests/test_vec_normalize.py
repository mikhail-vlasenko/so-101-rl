"""Verify VecNormalize behavior: obs normalization is on by default.

train.py uses VecNormalize(env, norm_reward=True, gamma=gamma) which leaves
norm_obs at its default value of True. This means observations are also
normalized — not just rewards. These tests prove that and check that the eval
env (a separate VecNormalize instance) has divergent obs statistics, making
eval scores unreliable.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class ConstantEnv(gym.Env):
    """Env that returns fixed obs and reward, for deterministic testing."""

    def __init__(self, obs_value=5.0, reward_value=3.0):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._obs = np.full(4, obs_value, dtype=np.float32)
        self._reward = reward_value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs.copy(), {}

    def step(self, action):
        return self._obs.copy(), self._reward, False, False, {}


class VaryingEnv(gym.Env):
    """Env with non-trivial obs variance so normalization has visible effect."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.array([10.0 + self._step, 20.0, 30.0, 40.0], dtype=np.float32)
        return obs, 1.0, False, False, {}


def test_norm_obs_is_true_by_default():
    """VecNormalize normalizes observations by default (norm_obs=True)."""
    vec_env = DummyVecEnv([lambda: ConstantEnv(obs_value=5.0)])
    norm_env = VecNormalize(vec_env, norm_reward=True)

    assert norm_env.norm_obs is True


def test_obs_are_modified_when_norm_obs_true():
    """With default norm_obs=True, returned obs differ from raw env obs."""
    vec_env = DummyVecEnv([lambda: VaryingEnv()])
    norm_env = VecNormalize(vec_env, norm_reward=True)

    raw_obs = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    obs = norm_env.reset()

    # Step enough to build up running statistics
    for _ in range(50):
        obs, _, _, _ = norm_env.step(np.array([[0.0]]))

    # After many steps, obs should be normalized (roughly zero-mean, unit-var)
    # and therefore different from the raw values
    assert not np.allclose(obs, raw_obs, atol=1.0), (
        f"Obs should be normalized but got values close to raw: {obs}"
    )


def test_obs_unchanged_when_norm_obs_false():
    """With norm_obs=False, observations pass through unchanged."""
    raw_value = 5.0
    vec_env = DummyVecEnv([lambda: ConstantEnv(obs_value=raw_value)])
    norm_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    obs = norm_env.reset()
    for _ in range(20):
        obs, _, _, _ = norm_env.step(np.array([[0.0]]))

    np.testing.assert_allclose(obs[0], np.full(4, raw_value), atol=1e-6)


def test_reward_is_normalized():
    """norm_reward=True modifies reward values."""
    raw_reward = 3.0
    vec_env = DummyVecEnv([lambda: ConstantEnv(reward_value=raw_reward)])
    norm_env = VecNormalize(vec_env, norm_reward=True)

    norm_env.reset()
    rewards = []
    for _ in range(50):
        _, reward, _, _ = norm_env.step(np.array([[0.0]]))
        rewards.append(reward[0])

    # After running mean builds up, normalized reward should differ from raw
    assert not np.isclose(rewards[-1], raw_reward, atol=0.5), (
        f"Reward should be normalized but got {rewards[-1]} (raw={raw_reward})"
    )


def test_separate_vec_normalize_has_different_obs_stats():
    """Two independent VecNormalize instances have divergent obs statistics.

    This is what happens in train.py: the training env and eval env are
    separate VecNormalize instances. The eval env (training=False) never
    updates its running stats, so it normalizes with the initial (identity)
    statistics while the training env accumulates real statistics.
    """
    # Training env: accumulate stats over many steps
    train_vec = DummyVecEnv([lambda: VaryingEnv()])
    train_env = VecNormalize(train_vec, norm_reward=True)

    train_env.reset()
    for _ in range(100):
        train_env.step(np.array([[0.0]]))

    # Eval env: fresh instance with training=False (frozen default stats)
    eval_vec = DummyVecEnv([lambda: VaryingEnv()])
    eval_env = VecNormalize(eval_vec, training=False, norm_reward=False)

    eval_obs = eval_env.reset()

    # Feed same raw obs through both normalizers
    raw_obs = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    train_normalized = train_env.normalize_obs(raw_obs)
    eval_normalized = eval_env.normalize_obs(raw_obs)

    assert not np.allclose(train_normalized, eval_normalized, atol=0.1), (
        f"Train and eval should normalize differently but got:\n"
        f"  train: {train_normalized}\n"
        f"  eval:  {eval_normalized}"
    )
