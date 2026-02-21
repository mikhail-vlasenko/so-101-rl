"""Train SAC on SO-101 tasks with Hydra config and W&B logging."""

import os
from collections import deque

import gymnasium
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from reach_env import SO101ReachEnv
from pickplace_env import SO101PickPlaceEnv

class InfoLoggingCallback(BaseCallback):
    """Log rolling mean of custom info dict metrics across episodes."""

    def __init__(self, info_keys: list[str], window: int = 100):
        super().__init__()
        self.info_keys = info_keys
        self._windows: dict[str, deque] = {k: deque(maxlen=window) for k in info_keys}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            for key in self.info_keys:
                if key in info:
                    self._windows[key].append(float(info[key]))
                    self.logger.record(
                        f"rollout/mean_{key}",
                        sum(self._windows[key]) / len(self._windows[key]),
                    )
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
    "reach": SO101ReachEnv,
    "pickplace": SO101PickPlaceEnv,
}


def make_env(cfg: DictConfig, render_mode=None, slow_factor=1):
    """Create env by name from Hydra config."""
    env_name = cfg.env_name
    env_cls = ENV_REGISTRY[env_name]

    if env_name == "reach":
        env_cfg = cfg.env
    elif env_name == "pickplace":
        env_cfg = cfg.pickplace_env
    else:
        raise ValueError(f"Unknown env: {env_name}")

    return env_cls(render_mode=render_mode, env_cfg=env_cfg, slow_factor=slow_factor)


def _make_env_fn(cfg, orig_dir):
    """Factory closure for SubprocVecEnv — each subprocess needs the correct cwd."""
    def _init():
        os.chdir(orig_dir)
        return Monitor(make_env(cfg))
    return _init


def train(cfg: DictConfig):
    # Hydra changes cwd — go back to original for MuJoCo XML paths
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)

    gamma = cfg.train.gamma
    n_envs = cfg.train.n_envs

    vec_env = SubprocVecEnv([_make_env_fn(cfg, orig_dir) for _ in range(n_envs)])
    env = VecNormalize(vec_env, norm_reward=True, gamma=gamma)

    phase_tracker = _MaxPhaseTracker(make_env(cfg))
    eval_vec_env = DummyVecEnv([lambda: Monitor(phase_tracker)])
    eval_env = VecNormalize(eval_vec_env, training=False, norm_reward=False)

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
        verbose=1,
        tensorboard_log=os.path.join(orig_dir, "logs"),
    )

    print(f"Training SAC ({cfg.env_name}) for {cfg.train.total_timesteps} steps...")
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks)
    model.save(os.path.join(log_dir, "final_model"))
    print(f"Model saved to {log_dir}/final_model.zip")

    if run is not None:
        run.finish()

    env.close()
    eval_env.close()


def evaluate(cfg: DictConfig):
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)

    model_path = cfg.eval_model_path or os.path.join(
        orig_dir, "logs", f"sac_{cfg.env_name}", "best_model.zip")
    env = make_env(cfg, render_mode="human", slow_factor=2)
    model = SAC.load(model_path)

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: return={total_reward:.2f}")

    env.close()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.eval:
        evaluate(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
