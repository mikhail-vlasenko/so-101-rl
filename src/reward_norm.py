"""Checkpoint-local reward scaling for training resumes.

Every training checkpoint has a matching <stem>.vecnormalize.pkl sidecar.
Missing sidecars (legacy or distilled policies) start with SB3 defaults and
report that choice. Present but invalid files fail loudly. Observation scaling
stays fixed inside the policy, never in VecNormalize.

Curriculum resumes retain return statistics and keep adapting them, using the
new config's gamma. Per-environment discounted returns start at zero because
episodes restart; this is not a snapshot of simulator/RNG/rollout-buffer state.
"""

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize


def reward_norm_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.suffix != ".zip":
        path = Path(f"{path}.zip")
    return path.with_suffix(".vecnormalize.pkl")


def training_reward_env(vec_env, gamma: float, checkpoint: str | Path | None):
    if checkpoint is not None:
        path = reward_norm_path(checkpoint)
        if path.exists():
            env = VecNormalize.load(path, vec_env)
            assert not env.norm_obs, "training checkpoints must use fixed policy observation normalization"
            env.training = True
            env.norm_reward = True
            env.gamma = gamma
            print(f"Restored reward normalization from {path}")
            return env
        print(f"No reward normalization at {path}; starting with default statistics")
    return VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=gamma)


def save_reward_norm(model, checkpoint: str | Path):
    env = model.get_vec_normalize_env()
    assert env is not None, "training checkpoints require a VecNormalize environment"
    assert not env.norm_obs, "observation normalization belongs inside the policy"
    env.save(reward_norm_path(checkpoint))


class RewardCheckpointCallback(CheckpointCallback):
    """Keep SB3's periodic model naming, with the same sidecar as best/final."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            save_reward_norm(self.model, self._checkpoint_path(extension="zip"))
        return result


class BestRewardNormCallback(BaseCallback):
    """Called only when EvalCallback saves a new best model; use training stats."""

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        self.checkpoint = checkpoint

    def _on_step(self) -> bool:
        save_reward_norm(self.model, self.checkpoint)
        return True
