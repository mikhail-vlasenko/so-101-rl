"""Tests for EpisodeCountCallback and ep_len_mean behavior with PPO.

Uses tiny dummy envs and real PPO to verify that:
- rollout/episodes counts completed episodes per rollout and resets between rollouts
- ep_len_mean from SB3's rolling deque(maxlen=100) produces exact values
  when all episodes in the buffer have the same length (explains the 200.0/300.0
  artifacts seen in multitask training)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import EpisodeCountCallback


class _FixedLenEnv(gym.Env):
    """Env that truncates after exactly `ep_len` steps."""

    def __init__(self, ep_len: int = 5):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.ep_len = ep_len
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= self.ep_len
        return np.zeros(4, dtype=np.float32), 0.0, False, truncated, {}


class _EarlyTermEnv(gym.Env):
    """Env that terminates after exactly `ep_len` steps."""

    def __init__(self, ep_len: int = 3):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.ep_len = ep_len
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self.ep_len
        return np.zeros(4, dtype=np.float32), 0.0, terminated, False, {}


class _CaptureWriter(KVWriter):
    def __init__(self):
        self.dumps: list[dict] = []

    def write(self, key_values: dict, key_excluded: dict, step: int) -> None:
        self.dumps.append(dict(key_values))

    def close(self) -> None:
        pass


def _episode_dumps(dumps: list[dict]) -> list[int]:
    return [d["rollout/episodes"] for d in dumps if "rollout/episodes" in d]


def _ep_len_dumps(dumps: list[dict]) -> list[float]:
    return [d["rollout/ep_len_mean"] for d in dumps if "rollout/ep_len_mean" in d]


def _run_ppo(env_fns: list, n_steps: int, total_timesteps: int,
             stats_window_size: int = 100) -> list[dict]:
    env = DummyVecEnv(env_fns)
    model = PPO("MlpPolicy", env, n_steps=n_steps,
                batch_size=n_steps * len(env_fns),
                n_epochs=1, verbose=0, stats_window_size=stats_window_size)

    writer = _CaptureWriter()
    model.set_logger(Logger(folder=None, output_formats=[writer]))

    cb = EpisodeCountCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb, log_interval=1)
    env.close()
    return writer.dumps


def _make_fixed(ep_len: int):
    return lambda: Monitor(_FixedLenEnv(ep_len=ep_len))


def _make_early(ep_len: int):
    return lambda: Monitor(_EarlyTermEnv(ep_len=ep_len))


def test_counts_truncated_episodes():
    """n_steps=10, ep_len=5 → 2 episodes per rollout."""
    counts = _episode_dumps(_run_ppo([_make_fixed(5)], n_steps=10,
                                     total_timesteps=20))
    assert counts == [2, 2]


def test_counts_terminated_episodes():
    """n_steps=9, ep_len=3 → 3 episodes per rollout."""
    counts = _episode_dumps(_run_ppo([_make_early(3)], n_steps=9,
                                     total_timesteps=18))
    assert counts == [3, 3]


def test_resets_between_rollouts():
    """Count resets each rollout — not cumulative."""
    counts = _episode_dumps(_run_ppo([_make_fixed(5)], n_steps=10,
                                     total_timesteps=40))
    assert counts == [2, 2, 2, 2]


def test_multiple_envs():
    """2 envs × ep_len=5 × n_steps=10 → 4 episodes per rollout."""
    counts = _episode_dumps(_run_ppo([_make_fixed(5)] * 2, n_steps=10,
                                     total_timesteps=20))
    assert counts == [4]


def test_mixed_lengths_across_envs():
    """2 envs with ep_len=3: n_steps=10 → 3 full episodes per env,
    with 1 step leftover each. 6 completed episodes per rollout."""
    counts = _episode_dumps(_run_ppo([_make_early(3)] * 2, n_steps=10,
                                     total_timesteps=20))
    # Each env does 10 steps: 3+3+3+1(partial) = 3 completed episodes per env
    assert counts == [6]


# --- ep_len_mean behavior with mixed episode lengths ---

def test_ep_len_mean_uniform_length():
    """All episodes same length → ep_len_mean is exactly that length."""
    lens = _ep_len_dumps(_run_ppo([_make_fixed(7)] * 2, n_steps=14,
                                  total_timesteps=28))
    assert all(v == 7.0 for v in lens)


def test_ep_len_mean_mixed_envs_averages():
    """Mixed ep_len envs produce a blended average, not exact values.

    1 env with ep_len=4, 1 env with ep_len=8, stats_window_size=4.
    n_steps=8: short env completes 2 eps (len 4,4), long env completes 1 ep (len 8).
    Deque after rollout 1: [4, 4, 8] → mean=5.33
    """
    dumps = _run_ppo([_make_fixed(4), _make_fixed(8)], n_steps=8,
                     total_timesteps=16, stats_window_size=4)
    lens = _ep_len_dumps(dumps)
    assert len(lens) >= 1
    # With mixed lengths, the mean should NOT be exactly 4 or 8
    assert lens[0] != 4.0
    assert lens[0] != 8.0


def test_ep_len_mean_deque_saturates_when_long_env_doesnt_complete():
    """When the long env doesn't complete within a rollout, only short
    episodes enter the deque, producing ep_len_mean exactly equal to
    the short env's length.

    This is the mechanism behind seeing 200.0 or 300.0 in multitask:
    the rollout boundary splits a long episode across two rollouts,
    so the deque fills with only short-episode completions.

    1 env with ep_len=3, 1 env with ep_len=25, stats_window_size=4.
    n_steps=18: short env completes 6 eps (at steps 3,6,9,12,15,18),
    long env completes 0 eps (needs 25 steps). Deque: [3,3,3,3] → 3.0.
    """
    dumps = _run_ppo([_make_fixed(3), _make_fixed(25)], n_steps=18,
                     total_timesteps=36, stats_window_size=4)
    lens = _ep_len_dumps(dumps)
    assert 3.0 in lens, f"Expected deque saturation at 3.0, got {lens}"
