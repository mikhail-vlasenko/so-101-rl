"""Train RL policies on SO-101 tasks with Hydra config and W&B logging."""

import os

import gymnasium
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from pickplace_env import SO101PickPlaceEnv

class _MeanMaxPhaseCallback(BaseCallback):
    """Log mean max_phase across episodes in each log window.

    Uses logger.record_mean() which accumulates values and automatically
    averages + resets on each dump_logs() call.
    """

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "max_phase" in info:
                self.logger.record_mean("rollout/mean_max_phase", float(info["max_phase"]))
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


def _sac_kwargs(cfg: DictConfig) -> dict:
    return {
        "buffer_size": cfg.sac.buffer_size,
        "learning_starts": cfg.sac.learning_starts,
        "train_freq": cfg.sac.train_freq,
        "gradient_steps": cfg.sac.gradient_steps,
        "tau": cfg.sac.tau,
    }


def _ppo_kwargs(cfg: DictConfig) -> dict:
    return {
        "n_steps": cfg.ppo.n_steps,
        "n_epochs": cfg.ppo.n_epochs,
        "clip_range": cfg.ppo.clip_range,
        "ent_coef": cfg.ppo.ent_coef,
        "gae_lambda": cfg.ppo.gae_lambda,
        "vf_coef": cfg.ppo.vf_coef,
    }


ALGORITHM_REGISTRY = {
    "sac": (SAC, _sac_kwargs),
    "ppo": (PPO, _ppo_kwargs),
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

    algo_name = cfg.algorithm
    algo_cls, algo_kwargs_fn = ALGORITHM_REGISTRY[algo_name]

    gamma = cfg.train.gamma
    n_envs = cfg.train.n_envs

    vec_env = SubprocVecEnv([_make_env_fn(cfg, xml_path) for _ in range(n_envs)])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=gamma)

    phase_tracker = _MaxPhaseTracker(make_env(cfg, xml_path=xml_path))
    eval_vec_env = DummyVecEnv([lambda: Monitor(phase_tracker)])
    eval_env = VecNormalize(eval_vec_env, norm_obs=False, training=False, norm_reward=False)

    log_dir = os.path.join(orig_dir, "logs", f"{algo_name}_{cfg.env_name}")
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

    callbacks.append(_MeanMaxPhaseCallback())

    callbacks.append(CheckpointCallback(
        save_freq=cfg.train.checkpoint_freq // n_envs,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=algo_name,
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

    if cfg.resume is not None:
        checkpoint_path = cfg.resume
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(orig_dir, checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        model = algo_cls.load(
            checkpoint_path,
            env=env,
            tensorboard_log=os.path.join(orig_dir, "logs"),
        )
    else:
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.learning_rate,
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
            policy_kwargs={"net_arch": list(cfg.train.net_arch)},
            stats_window_size=1,
            verbose=1,
            tensorboard_log=os.path.join(orig_dir, "logs"),
            **algo_kwargs_fn(cfg),
        )

    print(f"Training {algo_name.upper()} ({cfg.env_name}) for {cfg.train.total_timesteps} steps...")
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks, log_interval=cfg.train.log_interval)
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
