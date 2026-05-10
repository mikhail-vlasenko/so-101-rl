"""Train RL policies on SO-101 tasks with Hydra config and W&B logging."""

import os

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from callbacks import (
    CompletionRateCallback, CubeDragCallback, EpisodeCountCallback, EvalStatsCallback,
    EvalStatsTracker, FloorContactCallback, MaxCubeHeightCallback,
    MeanReturnCallback, RingContactCallback, TimeLimitCallback, XYProgressCallback,
)
from networks import LayerNormActorCriticPolicy, LayerNormSACPolicy
from lift_env import SO101LiftEnv
from pickplace_env import SO101PickPlaceEnv


def make_lr_schedule(name: str, initial_lr: float, min_lr: float):
    """Return a learning rate or schedule callable for SB3."""
    if name == "constant":
        return initial_lr
    if name == "linear":
        return lambda progress_remaining: min_lr + (initial_lr - min_lr) * progress_remaining
    raise ValueError(f"Unknown lr_schedule: {name!r}")


ENV_REGISTRY = {
    "pickplace": SO101PickPlaceEnv,
    "lift": SO101LiftEnv,
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
    "sac": (SAC, _sac_kwargs, LayerNormSACPolicy),
    "ppo": (PPO, _ppo_kwargs, LayerNormActorCriticPolicy),
}


def make_env(env_cls, env_cfg, xml_path, render_mode=None, slow_factor=1,
             obs_noise=None, obs_latency=0, obs_bias=None):
    """Create an env instance from class, config, and XML path."""
    return env_cls(render_mode=render_mode, env_cfg=env_cfg,
                   slow_factor=slow_factor, xml_path=xml_path,
                   obs_noise=obs_noise, obs_latency=obs_latency, obs_bias=obs_bias)


def _make_env_fn(env_cls, env_cfg, xml_path, obs_noise=None, obs_latency=0, obs_bias=None):
    """Factory closure for SubprocVecEnv."""
    def _init():
        return Monitor(env_cls(env_cfg=env_cfg, xml_path=xml_path,
                               obs_noise=obs_noise, obs_latency=obs_latency,
                               obs_bias=obs_bias))
    return _init


def _resolve_env(cfg, orig_dir, env_name):
    """Return (env_cls, env_cfg, xml_path) for a single env type."""
    env_cls = ENV_REGISTRY[env_name]
    return env_cls, cfg[f"{env_name}_env"], os.path.join(orig_dir, env_cls.XML_PATH)


def train(cfg: DictConfig):
    # Hydra changes cwd — go back to original for MuJoCo XML paths
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)

    algo_name = cfg.algorithm
    algo_cls, algo_kwargs_fn, policy_cls = ALGORITHM_REGISTRY[algo_name]

    seed = cfg.seed if cfg.seed is not None else None
    if seed is not None:
        set_random_seed(int(seed))

    gamma = cfg.train.gamma
    n_envs = cfg.train.n_envs

    obs_noise = OmegaConf.to_container(cfg.obs_noise, resolve=True)
    obs_latency = int(cfg.obs_latency)
    obs_bias = OmegaConf.to_container(cfg.obs_bias, resolve=True)

    if cfg.env_name == "multitask":
        lift_cls, lift_cfg, lift_xml = _resolve_env(cfg, orig_dir, "lift")
        pp_cls, pp_cfg, pp_xml = _resolve_env(cfg, orig_dir, "pickplace")
        n_lift = round(n_envs * cfg.lift_ratio)
        env_fns = [
            _make_env_fn(lift_cls, lift_cfg, lift_xml, obs_noise=obs_noise,
                         obs_latency=obs_latency, obs_bias=obs_bias) if i < n_lift
            else _make_env_fn(pp_cls, pp_cfg, pp_xml, obs_noise=obs_noise,
                              obs_latency=obs_latency, obs_bias=obs_bias)
            for i in range(n_envs)
        ]
        # Eval on pickplace (the harder task)
        eval_inner = make_env(pp_cls, pp_cfg, pp_xml, obs_noise=obs_noise,
                              obs_latency=obs_latency, obs_bias=obs_bias)
    else:
        env_cls, env_cfg, xml_path = _resolve_env(cfg, orig_dir, cfg.env_name)
        env_fns = [_make_env_fn(env_cls, env_cfg, xml_path, obs_noise=obs_noise,
                                obs_latency=obs_latency, obs_bias=obs_bias) for _ in range(n_envs)]
        eval_inner = make_env(env_cls, env_cfg, xml_path, obs_noise=obs_noise,
                              obs_latency=obs_latency, obs_bias=obs_bias)

    frame_stack = int(cfg.frame_stack)

    vec_env = SubprocVecEnv(env_fns)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    if seed is not None:
        vec_env.seed(int(seed))
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=gamma)

    stats_tracker = EvalStatsTracker(eval_inner)
    eval_vec_env = DummyVecEnv([lambda: Monitor(stats_tracker)])
    if frame_stack > 1:
        eval_vec_env = VecFrameStack(eval_vec_env, n_stack=frame_stack)
    if seed is not None:
        eval_vec_env.seed(int(seed) + 10_000)
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

    callbacks.append(MeanReturnCallback())
    callbacks.append(MaxCubeHeightCallback())
    callbacks.append(FloorContactCallback())
    callbacks.append(RingContactCallback())
    callbacks.append(CubeDragCallback())
    callbacks.append(XYProgressCallback())
    callbacks.append(CompletionRateCallback())
    callbacks.append(EpisodeCountCallback())

    if cfg.train.time_limit_minutes is not None:
        callbacks.append(TimeLimitCallback(cfg.train.time_limit_minutes))

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
        callback_after_eval=EvalStatsCallback(stats_tracker),
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
            policy_cls,
            env,
            learning_rate=make_lr_schedule(cfg.train.lr_schedule, cfg.train.learning_rate, cfg.train.lr_min),
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
            policy_kwargs={
                "net_arch": list(cfg.train.net_arch),
                "input_batchnorm": cfg.train.input_batchnorm,
            },
            stats_window_size=1,
            verbose=1,
            seed=int(seed) if seed is not None else None,
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
