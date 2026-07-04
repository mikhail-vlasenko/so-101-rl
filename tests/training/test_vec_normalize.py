"""Pin down VecNormalize's obs behavior and why train.py disables it.

VecNormalize defaults norm_obs=True, which normalizes observations with running
stats that two independent instances (train vs. eval) accumulate differently —
making eval scores unreliable. These tests demonstrate that divergence, which is
why train.py sets norm_obs=False (obs normalization is handled by the fixed
ObsNorm affine baked into the policy instead) and only uses VecNormalize for
reward scaling.
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

    The eval env (training=False) never updates its running stats, so it would
    normalize with the initial (identity) statistics while the training env
    accumulates real ones. This divergence is exactly why train.py keeps
    norm_obs=False and normalizes obs via the baked ObsNorm affine instead.
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
