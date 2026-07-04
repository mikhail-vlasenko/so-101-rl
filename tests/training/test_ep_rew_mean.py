"""Tests for SB3's built-in rollout/ep_rew_mean averaging behavior.

Explores how the deque(maxlen=stats_window_size) works:
- ep_rew_mean averages over the last N *completed episodes*, not per-rollout
- The deque persists across rollouts, so old episodes dilute new ones
- With stats_window_size=1 (our config), only the most recent episode matters
- Contrast with record_mean which averages over episodes within a single rollout
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class _FixedRewardEnv(gym.Env):
    """Env that gives a fixed reward per step and truncates after ep_len steps."""

    def __init__(self, reward_per_step: float, ep_len: int = 5):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.reward_per_step = reward_per_step
        self.ep_len = ep_len
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= self.ep_len
        return np.zeros(4, dtype=np.float32), self.reward_per_step, False, truncated, {}


class _RampingRewardEnv(gym.Env):
    """Env where reward increases by 1.0 each episode. Episode N gives reward N per step."""

    def __init__(self, ep_len: int = 5):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.ep_len = ep_len
        self._step_count = 0
        self._episode_num = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_num += 1
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= self.ep_len
        return np.zeros(4, dtype=np.float32), float(self._episode_num), False, truncated, {}


class _CaptureWriter(KVWriter):
    def __init__(self):
        self.dumps: list[dict] = []

    def write(self, key_values: dict, key_excluded: dict, step: int) -> None:
        self.dumps.append(dict(key_values))

    def close(self) -> None:
        pass


def _rew_dumps(dumps: list[dict]) -> list[float]:
    return [d["rollout/ep_rew_mean"] for d in dumps if "rollout/ep_rew_mean" in d]


def _run_ppo(env_fns: list, n_steps: int, total_timesteps: int,
             stats_window_size: int = 100) -> list[dict]:
    env = DummyVecEnv(env_fns)
    model = PPO("MlpPolicy", env, n_steps=n_steps,
                batch_size=n_steps * len(env_fns),
                n_epochs=1, verbose=0, stats_window_size=stats_window_size)

    writer = _CaptureWriter()
    model.set_logger(Logger(folder=None, output_formats=[writer]))

    model.learn(total_timesteps=total_timesteps, log_interval=1)
    env.close()
    return writer.dumps


def _make_fixed(reward_per_step: float, ep_len: int = 5):
    return lambda: Monitor(_FixedRewardEnv(reward_per_step=reward_per_step, ep_len=ep_len))


def _make_ramping(ep_len: int = 5):
    return lambda: Monitor(_RampingRewardEnv(ep_len=ep_len))


# --- Basic deque behavior ---

def test_constant_reward_gives_exact_mean():
    """1 env, reward=2.0/step, ep_len=5 → ep_return=10.0 every episode."""
    rews = _rew_dumps(_run_ppo([_make_fixed(2.0, ep_len=5)], n_steps=10,
                                total_timesteps=20))
    assert all(r == 10.0 for r in rews), f"Expected all 10.0, got {rews}"


def test_deque_default_size_is_100():
    """With stats_window_size=100 (default), deque holds up to 100 episodes."""
    # Run enough episodes to exceed 100, verify the mean only reflects recent ones.
    # 1 env, ep_len=3, ramping reward: ep N gives N per step → return = 3*N.
    # n_steps=30 → 10 eps per rollout, 15 rollouts → 150 eps total.
    dumps = _run_ppo([_make_ramping(ep_len=3)], n_steps=30,
                     total_timesteps=450, stats_window_size=100)
    rews = _rew_dumps(dumps)

    # After 150 episodes, the deque holds episodes 51-150.
    # Episode N has return 3*N. Mean of 3*51 + 3*52 + ... + 3*150 = 3 * mean(51..150) = 3 * 100.5 = 301.5
    # The last logged value should be close to this.
    assert rews[-1] > rews[0], f"Reward should increase over time: first={rews[0]}, last={rews[-1]}"
    # After enough episodes, the mean should NOT reflect the very first episodes
    # (which had low returns), proving the deque drops old values.
    assert rews[-1] > 200.0, f"Expected last mean > 200 (deque dropped early low-return eps), got {rews[-1]}"


def test_stats_window_size_1_tracks_last_episode_only():
    """With stats_window_size=1, ep_rew_mean is just the last completed episode's return.

    This is what our training config uses. It makes ep_rew_mean maximally jittery.
    """
    # Ramping env: episode N gives return = 5*N (ep_len=5, reward=N per step).
    # n_steps=5 → 1 episode per rollout. Each rollout's ep_rew_mean = that episode's return.
    dumps = _run_ppo([_make_ramping(ep_len=5)], n_steps=5,
                     total_timesteps=25, stats_window_size=1)
    rews = _rew_dumps(dumps)

    # Episode 1: return=5, Episode 2: return=10, etc.
    expected = [5.0, 10.0, 15.0, 20.0, 25.0]
    assert rews == expected, f"Expected {expected}, got {rews}"


def test_deque_persists_across_rollouts():
    """The deque does NOT reset between rollouts — old episodes dilute new ones.

    With window=4, ramping reward, ep_len=3, n_steps=3 (1 ep per rollout):
    - Rollout 1: deque=[3], mean=3
    - Rollout 2: deque=[3,6], mean=4.5
    - Rollout 3: deque=[3,6,9], mean=6
    - Rollout 4: deque=[3,6,9,12], mean=7.5
    - Rollout 5: deque=[6,9,12,15], mean=10.5  (oldest dropped)
    """
    dumps = _run_ppo([_make_ramping(ep_len=3)], n_steps=3,
                     total_timesteps=15, stats_window_size=4)
    rews = _rew_dumps(dumps)

    expected = [3.0, 4.5, 6.0, 7.5, 10.5]
    assert rews == expected, f"Expected {expected}, got {rews}"


def test_multiple_envs_interleave_into_same_deque():
    """Two envs with different rewards share one deque.

    env1: reward=1.0/step, ep_len=5 → return=5.0
    env2: reward=3.0/step, ep_len=5 → return=15.0
    n_steps=5 → each env completes 1 ep per rollout → 2 eps enter deque.
    stats_window_size=2: deque always has [5.0, 15.0] → mean=10.0.
    """
    rews = _rew_dumps(_run_ppo(
        [_make_fixed(1.0, ep_len=5), _make_fixed(3.0, ep_len=5)],
        n_steps=5, total_timesteps=10, stats_window_size=2,
    ))
    assert all(r == 10.0 for r in rews), f"Expected all 10.0, got {rews}"


def test_slow_env_skews_mean():
    """When one env has longer episodes, its completions are underrepresented.

    env1: reward=0.0/step, ep_len=3 → return=0.0 (completes often)
    env2: reward=10.0/step, ep_len=15 → return=150.0 (completes rarely)
    n_steps=15, stats_window_size=100.

    In 15 steps: env1 completes 5 episodes (returns: 0,0,0,0,0),
    env2 completes 1 episode (return: 150). Deque: [0,0,0,0,0,150] → mean=25.0.
    The high-reward env is underrepresented (1/6 of deque vs 1/2 of envs).
    """
    dumps = _run_ppo(
        [_make_fixed(0.0, ep_len=3), _make_fixed(10.0, ep_len=15)],
        n_steps=15, total_timesteps=30, stats_window_size=100,
    )
    rews = _rew_dumps(dumps)
    assert rews[0] == 25.0, f"Expected 25.0 (skewed by fast env), got {rews[0]}"
