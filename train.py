"""Train SAC on SO-101 tasks with Hydra config and W&B logging."""

import os

import gymnasium
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from pickplace_env import SO101PickPlaceEnv

class InfoLoggingCallback(BaseCallback):
    """Log per-episode custom info dict metrics (no averaging)."""

    def __init__(self, info_keys: list[str]):
        super().__init__()
        self.info_keys = info_keys

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            for key in self.info_keys:
                if key in info:
                    self.logger.record(f"rollout/{key}", float(info[key]))
        return True


class _MaxPhaseTracker(gymnasium.Wrapper):
    """Track max_phase at episode end for eval logging."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_phases: list[float] = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if (term or trunc) and "max_phase" in info:
            self.episode_phases.append(float(info["max_phase"]))
        return obs, rew, term, trunc, info

    def pop_phases(self) -> list[float]:
        phases = self.episode_phases.copy()
        self.episode_phases.clear()
        return phases


class _EvalPhaseCallback(BaseCallback):
    """Log mean max_phase after each evaluation round."""

    def __init__(self, phase_tracker: _MaxPhaseTracker):
        super().__init__()
        self.phase_tracker = phase_tracker

    def _on_step(self) -> bool:
        phases = self.phase_tracker.pop_phases()
        if phases:
            self.logger.record("eval/mean_max_phase", sum(phases) / len(phases))
        return True


ENV_REGISTRY = {
    "pickplace": SO101PickPlaceEnv,
}


def make_env(cfg: DictConfig, xml_path=None, render_mode=None, slow_factor=1):
    """Create env by name from Hydra config."""
    env_cls = ENV_REGISTRY[cfg.env_name]
    return env_cls(render_mode=render_mode, env_cfg=cfg.pickplace_env,
                   slow_factor=slow_factor, xml_path=xml_path)


def _make_env_fn(cfg, xml_path):
    """Factory closure for SubprocVecEnv — uses absolute XML path."""
    def _init():
        return Monitor(make_env(cfg, xml_path=xml_path))
    return _init


def train(cfg: DictConfig):
    # Hydra changes cwd — go back to original for MuJoCo XML paths
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)
    xml_path = os.path.join(orig_dir, SO101PickPlaceEnv.XML_PATH)

    gamma = cfg.train.gamma
    n_envs = cfg.train.n_envs

    vec_env = SubprocVecEnv([_make_env_fn(cfg, xml_path) for _ in range(n_envs)])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=gamma)

    phase_tracker = _MaxPhaseTracker(make_env(cfg, xml_path=xml_path))
    eval_vec_env = DummyVecEnv([lambda: Monitor(phase_tracker)])
    eval_env = VecNormalize(eval_vec_env, norm_obs=False, training=False, norm_reward=False)

    log_dir = os.path.join(orig_dir, "logs", f"sac_{cfg.env_name}")
    os.makedirs(log_dir, exist_ok=True)

    # W&B init (syncs metrics via tensorboard, no checkpoint uploads)
    run = None
    callbacks = []
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    callbacks.append(InfoLoggingCallback(info_keys=["max_phase"]))

    callbacks.append(CheckpointCallback(
        save_freq=cfg.train.checkpoint_freq // n_envs,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac",
    ))

    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=cfg.train.eval_freq // n_envs,
        n_eval_episodes=cfg.train.n_eval_episodes,
        deterministic=True,
        callback_after_eval=_EvalPhaseCallback(phase_tracker),
    ))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.train.learning_rate,
        batch_size=cfg.train.batch_size,
        buffer_size=cfg.train.buffer_size,
        learning_starts=cfg.train.learning_starts,
        tau=cfg.train.tau,
        gamma=cfg.train.gamma,
        policy_kwargs={"net_arch": list(cfg.train.net_arch)},
        stats_window_size=1,
        verbose=1,
        tensorboard_log=os.path.join(orig_dir, "logs"),
    )

    print(f"Training SAC ({cfg.env_name}) for {cfg.train.total_timesteps} steps...")
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks, log_interval=100)
    model.save(os.path.join(log_dir, "final_model"))
    print(f"Model saved to {log_dir}/final_model.zip")

    if run is not None:
        run.finish()

    env.close()
    eval_env.close()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
