"""Train RL policies on SO-101 tasks with Hydra config and W&B logging."""

import os

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from src.base_env import RuntimeEnvConfig, obs_dim_for
from src.callbacks import (
    CompletionRateCallback, CubeDragCallback, EpisodeCountCallback, EpisodeLengthCallback,
    EvalStatsCallback, EvalStatsTracker, FloorContactCallback, LiftSuccessCallback,
    MarkerHiddenCallback, MaxCubeHeightCallback, MeanReturnCallback, ReachStatsCallback,
    RingContactCallback, TimeLimitCallback, XYProgressCallback,
)
from src.networks import LayerNormActorCriticPolicy, LayerNormSACPolicy
from src.obs_norm import build_obs_norm, build_reach_obs_norm
from src.lift_env import SO101LiftEnv
from src.pickplace_env import SO101PickPlaceEnv
from src.reach_env import SO101ReachEnv


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
    "reach": SO101ReachEnv,
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


def resume_overrides(cfg: DictConfig) -> dict:
    """Hyperparameters to force onto a loaded checkpoint, as load() kwargs.

    SB3's load() restores every hyperparameter from the zip, so without these
    a stage-2 curriculum override (lr, gamma, ppo.*) would be silently ignored
    on resume. load() applies **kwargs on top of the restored attributes
    before _setup_model(), which then rebuilds the lr schedule, clip-range
    wrapper, and rollout/replay buffer from the new values — while the policy
    weights still come from the checkpoint. Architecture-affecting settings
    (policy_kwargs, obs/action spaces) must keep matching the checkpoint and
    are deliberately absent.
    """
    _, algo_kwargs_fn, _ = ALGORITHM_REGISTRY[cfg.algorithm]
    return {
        "learning_rate": make_lr_schedule(cfg.train.lr_schedule,
                                          cfg.train.learning_rate, cfg.train.lr_min),
        "batch_size": cfg.train.batch_size,
        "gamma": cfg.train.gamma,
        **algo_kwargs_fn(cfg),
    }


def make_env(env_cls, env_cfg, xml_path, render_mode=None, slow_factor=1,
             *, cfg: RuntimeEnvConfig):
    """Create an env instance from class, config, and XML path."""
    return env_cls(render_mode=render_mode, env_cfg=env_cfg,
                   slow_factor=slow_factor, xml_path=xml_path,
                   cfg=cfg)


def _make_env_fn(env_cls, env_cfg, xml_path, *, cfg: RuntimeEnvConfig):
    """Factory closure for SubprocVecEnv."""
    def _init():
        return Monitor(env_cls(env_cfg=env_cfg, xml_path=xml_path,
                               cfg=cfg))
    return _init


def runtime_cfg_from_hydra(cfg: DictConfig) -> RuntimeEnvConfig:
    """Build typed runtime env options from the composed Hydra config."""
    return RuntimeEnvConfig(
        obs_noise=OmegaConf.to_container(cfg.obs_noise, resolve=True),
        cam_latency=(OmegaConf.to_container(cfg.cam_latency, resolve=True)
                     if cfg.cam_latency is not None else None),
        obs_bias=OmegaConf.to_container(cfg.obs_bias, resolve=True),
        marker_dropout=OmegaConf.to_container(cfg.marker_dropout, resolve=True),
        marker_always_visible=bool(cfg.marker_always_visible),
        marker_include_rot=bool(cfg.marker_include_rot),
        prev_actions_n=int(cfg.prev_actions_n),
        cube_size_jitter=float(cfg.cube_size_jitter),
        history_taps=tuple(int(t) for t in cfg.history_taps),
    )


def _resolve_env(cfg, orig_dir, env_name):
    """Return (env_cls, env_cfg, xml_path) for a single env type."""
    env_cls = ENV_REGISTRY[env_name]
    return env_cls, cfg[f"{env_name}_env"], os.path.join(orig_dir, env_cls.XML_PATH)


def env_specs(cfg, orig_dir, runtime_cfg: RuntimeEnvConfig, n_envs: int):
    """(env_fns, eval_inner, n_substeps) for the configured env(s).

    Encapsulates the multitask-vs-single branching so training and the
    distillation rig (src/distill.py) build the identical env farm: a list of
    SubprocVecEnv factory closures, one raw eval env (pickplace for multitask —
    the harder task), and the shared control rate that fixes the obs_norm qvel
    scale."""
    if cfg.env_name == "multitask":
        lift_cls, lift_cfg, lift_xml = _resolve_env(cfg, orig_dir, "lift")
        pp_cls, pp_cfg, pp_xml = _resolve_env(cfg, orig_dir, "pickplace")
        n_lift = round(n_envs * cfg.lift_ratio)
        env_fns = [
            _make_env_fn(lift_cls, lift_cfg, lift_xml, cfg=runtime_cfg) if i < n_lift
            else _make_env_fn(pp_cls, pp_cfg, pp_xml, cfg=runtime_cfg)
            for i in range(n_envs)
        ]
        eval_inner = make_env(pp_cls, pp_cfg, pp_xml, cfg=runtime_cfg)
        assert int(lift_cfg["n_substeps"]) == int(pp_cfg["n_substeps"]), \
            "multitask envs must share a control rate for one obs_norm qvel scale"
        n_substeps = int(lift_cfg["n_substeps"])
    else:
        env_cls, env_cfg, xml_path = _resolve_env(cfg, orig_dir, cfg.env_name)
        env_fns = [_make_env_fn(env_cls, env_cfg, xml_path, cfg=runtime_cfg)
                   for _ in range(n_envs)]
        eval_inner = make_env(env_cls, env_cfg, xml_path, cfg=runtime_cfg)
        n_substeps = int(env_cfg["n_substeps"])
    return env_fns, eval_inner, n_substeps


def obs_norm_for(cfg, n_substeps: int) -> tuple:
    """Tiled (center, scale) lists for the policy's fixed ObsNorm, matching the
    configured env layout and history taps: the actor block's constants repeat
    once per tap, the privileged tail's appear once at the end — mirroring the
    env's [actor block per tap | priv tail] layout (each tap is a past copy of
    the same physical quantities, so it keeps the same constants). Single
    source shared by training and distillation so a distilled student's input
    normalization is identical to a trained one (the constants ship inside the
    checkpoint)."""
    if cfg.env_name == "reach":
        obs_center, obs_scale = build_reach_obs_norm(
            int(cfg.prev_actions_n), len(cfg.reach_env.waypoints),
            float(cfg.action_scale), n_substeps)
        return obs_center.tolist(), obs_scale.tolist()
    obs_center, obs_scale = build_obs_norm(
        int(cfg.prev_actions_n), bool(cfg.marker_include_rot),
        float(cfg.action_scale), n_substeps)
    n_taps = len(cfg.history_taps)
    actor_dim = obs_dim_for(int(cfg.prev_actions_n), bool(cfg.marker_include_rot))
    return (obs_center[:actor_dim].tolist() * n_taps + obs_center[actor_dim:].tolist(),
            obs_scale[:actor_dim].tolist() * n_taps + obs_scale[actor_dim:].tolist())


def actor_obs_dim_for(cfg) -> int | None:
    """Actor input width for the asymmetric critic (src/networks.TakeFirst).

    The cube envs' observation is [actor block per history tap | privileged
    tail] (base_env.obs_dim_for / priv_dim_for, src/obs_history.py) and only
    the critic may read the tail; the actor's slice covers every tapped actor
    block — the history is actor input, the tail never is. None for reach,
    which has no privileged tail (symmetric nets) and no history-tap wiring."""
    if cfg.env_name == "reach":
        assert list(cfg.history_taps) == [0], \
            "reach has no history-tap wiring; run it with history_taps=[0]"
        return None
    assert cfg.algorithm == "ppo", \
        "the asymmetric-critic obs layout is wired for PPO only (the SAC actor " \
        "would consume the privileged tail)"
    return len(cfg.history_taps) * obs_dim_for(int(cfg.prev_actions_n),
                                               bool(cfg.marker_include_rot))


def build_fresh_model(cfg, env, obs_norm, net_arch, seed, *, actor_obs_dim,
                      tensorboard_log=None, verbose=1):
    """Construct a from-scratch SB3 model with the fixed ObsNorm baked into its
    policy. Shared by training's cold start and the distillation rig's student,
    so both carry the identical architecture, normalization, and algorithm
    hyperparameters — and the student saves to a checkpoint `resume` restores
    directly."""
    algo_cls, algo_kwargs_fn, policy_cls = ALGORITHM_REGISTRY[cfg.algorithm]
    policy_kwargs = {
        "net_arch": list(net_arch),
        "obs_norm": obs_norm,
    }
    # Only the PPO policy accepts the asymmetric-critic slice; reach (the one
    # symmetric layout, actor_obs_dim None) omits the key so SAC stays valid.
    if actor_obs_dim is not None:
        policy_kwargs["actor_obs_dim"] = int(actor_obs_dim)
    return algo_cls(
        policy_cls,
        env,
        learning_rate=make_lr_schedule(cfg.train.lr_schedule,
                                       cfg.train.learning_rate, cfg.train.lr_min),
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        policy_kwargs=policy_kwargs,
        stats_window_size=1,
        verbose=verbose,
        seed=int(seed) if seed is not None else None,
        tensorboard_log=tensorboard_log,
        **algo_kwargs_fn(cfg),
    )


def train(cfg: DictConfig):
    # Hydra changes cwd — go back to original for MuJoCo XML paths
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)

    algo_name = cfg.algorithm
    algo_cls, _, _ = ALGORITHM_REGISTRY[algo_name]

    seed = cfg.seed if cfg.seed is not None else None
    if seed is not None:
        set_random_seed(int(seed))

    gamma = cfg.train.gamma
    n_envs = cfg.train.n_envs

    runtime_cfg = runtime_cfg_from_hydra(cfg)

    env_fns, eval_inner, n_substeps = env_specs(cfg, orig_dir, runtime_cfg, n_envs)

    # Fixed input normalization constants (src/obs_norm.py), tiled across
    # history taps. Passed as plain lists inside policy_kwargs so they
    # save/load cleanly with the checkpoint and real rollouts inherit the
    # identical transform through the loaded policy.
    obs_norm = obs_norm_for(cfg, n_substeps)

    vec_env = SubprocVecEnv(env_fns)
    if seed is not None:
        vec_env.seed(int(seed))
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=gamma)

    stats_tracker = EvalStatsTracker(eval_inner)
    eval_vec_env = DummyVecEnv([lambda: Monitor(stats_tracker)])
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
    callbacks.append(EpisodeLengthCallback())
    callbacks.append(LiftSuccessCallback())
    callbacks.append(MaxCubeHeightCallback())
    callbacks.append(FloorContactCallback())
    callbacks.append(RingContactCallback())
    callbacks.append(CubeDragCallback())
    callbacks.append(MarkerHiddenCallback())
    callbacks.append(XYProgressCallback())
    callbacks.append(CompletionRateCallback())
    callbacks.append(ReachStatsCallback())
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
        overrides = resume_overrides(cfg)
        print(f"Loading checkpoint from {checkpoint_path} "
              f"(hyperparameters re-applied from config: {sorted(overrides)})")
        model = algo_cls.load(
            checkpoint_path,
            env=env,
            tensorboard_log=os.path.join(orig_dir, "logs"),
            **overrides,
        )
    else:
        model = build_fresh_model(
            cfg, env, obs_norm, list(cfg.train.net_arch), seed,
            actor_obs_dim=actor_obs_dim_for(cfg),
            tensorboard_log=os.path.join(orig_dir, "logs"), verbose=1)

    print(f"Training {algo_name.upper()} ({cfg.env_name}) for {cfg.train.total_timesteps} steps...")
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks, log_interval=cfg.train.log_interval)
    model.save(os.path.join(log_dir, "final_model"))
    print(f"Model saved to {log_dir}/final_model.zip")

    if run is not None:
        run.finish()

    env.close()
    eval_env.close()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
