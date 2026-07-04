"""Verify gradient step frequency in the SAC setup.

SB3's SAC defaults: train_freq=1, gradient_steps=1. With n_envs=8 in a
vectorized env, this means 1 gradient step per 8 environment transitions.
The original SAC paper uses a 1:1 ratio — so we're doing 8x fewer updates
per sample than the paper recommends.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class DummyEnv(gym.Env):
    """Minimal env for testing SAC internals."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def test_sac_defaults_one_gradient_step_per_env_step():
    """SAC defaults: 1 gradient step per 1 env.step() call."""
    env = DummyVecEnv([DummyEnv])
    model = SAC("MlpPolicy", env, verbose=0)

    assert model.train_freq.frequency == 1
    assert model.train_freq.unit.value == "step"
    assert model.gradient_steps == 1


def test_gradient_to_transition_ratio_with_n_envs():
    """With n_envs=8, each env.step() produces 8 transitions but only 1 gradient step.

    After learning_starts, over N vectorized steps we expect:
    - N * n_envs transitions added to the buffer
    - N gradient steps performed
    - Ratio: 1 gradient step per n_envs transitions
    """
    n_envs = 8
    learning_starts = 80  # aligned to n_envs
    total_timesteps = 320  # 240 steps after learning_starts

    vec_env = DummyVecEnv([DummyEnv for _ in range(n_envs)])
    model = SAC(
        "MlpPolicy", vec_env,
        learning_starts=learning_starts,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)

    assert model.replay_buffer.pos > 0 or model.replay_buffer.full

    # _n_updates tracks total gradient steps taken.
    # After learning_starts, each vectorized env.step() triggers 1 gradient step.
    # With n_envs=8, each env.step() adds 8 transitions but only 1 gradient update.
    steps_after_learning = total_timesteps - learning_starts
    vec_steps_after_learning = steps_after_learning // n_envs
    assert model._n_updates == vec_steps_after_learning, (
        f"Expected {vec_steps_after_learning} gradient steps, got {model._n_updates}. "
        f"Ratio is 1 gradient step per {n_envs} transitions."
    )


def test_increasing_gradient_steps_increases_updates():
    """Setting gradient_steps=n_envs gives ~1:1 update-to-transition ratio."""
    n_envs = 4
    gradient_steps = 4
    learning_starts = 100
    total_steps = 200

    vec_env = DummyVecEnv([DummyEnv for _ in range(n_envs)])
    model = SAC(
        "MlpPolicy", vec_env,
        learning_starts=learning_starts,
        gradient_steps=gradient_steps,
        verbose=0,
    )
    model.learn(total_timesteps=learning_starts + total_steps)

    vec_steps_after_learning = total_steps // n_envs
    expected_updates = vec_steps_after_learning * gradient_steps
    assert model._n_updates == expected_updates, (
        f"Expected {expected_updates} gradient steps, got {model._n_updates}. "
        f"With gradient_steps={gradient_steps}, each vec env.step() should "
        f"trigger {gradient_steps} updates."
    )
