"""Integration tests for _MeanMaxPhaseCallback with SAC.

Uses a tiny dummy env and real SAC to verify that mean_max_phase
is averaged correctly per log window via logger.record_mean().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from train import _MeanMaxPhaseCallback


class _PhaseEnv(gym.Env):
    """Tiny env that ends after `ep_len` steps and reports a fixed max_phase.

    max_phase cycles through `phase_sequence` across episodes.
    """

    def __init__(self, ep_len: int = 5, phase_sequence: list[int] = [1, 2, 3]):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.ep_len = ep_len
        self.phase_sequence = phase_sequence
        self._step_count = 0
        self._episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= self.ep_len
        phase = self.phase_sequence[self._episode % len(self.phase_sequence)]
        info = {"max_phase": phase} if done else {}
        if done:
            self._episode += 1
        return np.zeros(4, dtype=np.float32), 0.0, done, False, info


class _CaptureWriter(KVWriter):
    """Logger backend that captures all dumped key-value pairs."""

    def __init__(self):
        self.dumps: list[dict] = []

    def write(self, key_values: dict, key_excluded: dict, step: int) -> None:
        self.dumps.append(dict(key_values))

    def close(self) -> None:
        pass


def _train(ep_len: int, phase_sequence: list[int], total_timesteps: int,
           log_interval: int, n_envs: int = 1) -> list[dict]:
    """Run SAC with _MeanMaxPhaseCallback on _PhaseEnv, return captured logs."""
    def make():
        return Monitor(_PhaseEnv(ep_len=ep_len, phase_sequence=phase_sequence))

    env = DummyVecEnv([make for _ in range(n_envs)])
    model = SAC("MlpPolicy", env, learning_starts=0, train_freq=1,
                gradient_steps=1, verbose=0)

    writer = _CaptureWriter()
    model.set_logger(Logger(folder=None, output_formats=[writer]))

    cb = _MeanMaxPhaseCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb,
                log_interval=log_interval)
    env.close()
    return writer.dumps


def _phase_dumps(dumps: list[dict]) -> list[dict]:
    return [d for d in dumps if "rollout/mean_max_phase" in d]


def test_averages_phases_in_window():
    """mean_max_phase is the average of all episode max_phases in the window."""
    # phase_sequence=[1, 2, 3], log_interval=3 → first window sees 1, 2, 3
    # ep_len=5, so 6 episodes = 30 steps → 2 full windows
    dumps = _phase_dumps(_train(
        ep_len=5, phase_sequence=[1, 2, 3], total_timesteps=30, log_interval=3,
    ))
    assert len(dumps) == 2
    assert dumps[0]["rollout/mean_max_phase"] == 2.0  # (1+2+3)/3
    assert dumps[1]["rollout/mean_max_phase"] == 2.0  # (1+2+3)/3


def test_accumulator_resets_between_windows():
    """Each window is independent — accumulator doesn't carry over.

    Uses asymmetric phases so cumulative averaging would produce
    different results than windowed: [1,1,4,4] with log_interval=2
    gives windows [1,1]→1.0 and [4,4]→4.0. If cumulative,
    window 2 would be (1+1+4+4)/4=2.5 instead of 4.0.
    """
    dumps = _phase_dumps(_train(
        ep_len=5, phase_sequence=[1, 1, 4, 4], total_timesteps=25, log_interval=2,
    ))
    assert len(dumps) == 2
    assert dumps[0]["rollout/mean_max_phase"] == 1.0
    assert dumps[1]["rollout/mean_max_phase"] == 4.0


def test_single_episode_per_window():
    """With log_interval=1, each dump reflects exactly one episode."""
    # phase_sequence=[0, 3] alternates, ep_len=5
    # 20 steps → 4 episodes → 4 dumps
    dumps = _phase_dumps(_train(
        ep_len=5, phase_sequence=[0, 3], total_timesteps=20, log_interval=1,
    ))
    assert len(dumps) == 4
    assert [d["rollout/mean_max_phase"] for d in dumps] == [0.0, 3.0, 0.0, 3.0]


def test_vectorized_envs():
    """With 2 envs, episodes from both envs contribute to the same window.

    2 envs × ep_len=5 → both finish an episode every 5 steps.
    phase_sequence=[1, 3]: env0 gets phase 1, env1 gets phase 1 (both episode 0),
    then env0 gets 3, env1 gets 3 (both episode 1), etc.
    log_interval=4 → 4 episodes per window. 20 steps → 4 episodes → 1 dump.
    Phases: 1, 1, 3, 3 → mean=2.0
    """
    dumps = _phase_dumps(_train(
        ep_len=5, phase_sequence=[1, 3], total_timesteps=20,
        log_interval=4, n_envs=2,
    ))
    assert len(dumps) == 1
    assert dumps[0]["rollout/mean_max_phase"] == 2.0
