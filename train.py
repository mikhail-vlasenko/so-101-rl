"""Train SAC on SO-101 tasks with Hydra config and W&B logging."""

import os

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from reach_env import SO101ReachEnv
from pickplace_env import SO101PickPlaceEnv

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


def train(cfg: DictConfig):
    # Hydra changes cwd — go back to original for MuJoCo XML paths
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)

    env = make_env(cfg)
    eval_env = make_env(cfg)

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

    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=cfg.train.eval_freq,
        n_eval_episodes=cfg.train.n_eval_episodes,
        deterministic=True,
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
    env = make_env(cfg, render_mode="human", slow_factor=5)
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
        print(f"Episode {ep + 1}: reward={total_reward:.2f}")

    env.close()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.eval:
        evaluate(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
