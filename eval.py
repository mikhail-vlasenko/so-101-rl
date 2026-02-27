"""Evaluate a trained checkpoint with visual rendering.

Usage:
    python eval.py                          # latest checkpoint
    python eval.py model=best               # best_model from EvalCallback
    python eval.py model=path/to/model.zip  # specific path
    python eval.py episodes=20              # override episode count
    python eval.py algorithm=ppo            # evaluate PPO model
"""

import os

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from train import ENV_REGISTRY, _resolve_env, make_env

ALGORITHM_CLASSES = {
    "sac": SAC,
    "ppo": PPO,
}


def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recently created checkpoint file."""
    zips = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not zips:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(
        (os.path.join(checkpoint_dir, f) for f in zips),
        key=os.path.getmtime,
    )


def _resolve_model_path(model_arg: str, log_dir: str) -> str:
    if model_arg == "latest":
        return _find_latest_checkpoint(os.path.join(log_dir, "checkpoints"))
    if model_arg == "best":
        return os.path.join(log_dir, "best_model.zip")
    return model_arg


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    orig_dir = hydra.utils.get_original_cwd()
    os.chdir(orig_dir)
    # Multitask evals on pickplace (the harder task), matching training eval
    eval_env_name = "pickplace" if cfg.env_name == "multitask" else cfg.env_name
    env_cls, env_cfg, xml_path = _resolve_env(cfg, orig_dir, eval_env_name)

    algo_name = cfg.algorithm
    algo_cls = ALGORITHM_CLASSES[algo_name]

    log_dir = os.path.join(orig_dir, "logs", f"{algo_name}_{cfg.env_name}")
    model_arg = cfg.get("model", "latest")
    model_path = _resolve_model_path(model_arg, log_dir)

    episodes = int(cfg.get("episodes", 10))

    print(f"Loading {algo_name.upper()} model: {model_path}")
    model = algo_cls.load(model_path)

    raw_env = make_env(env_cls, env_cfg, xml_path, render_mode="human", slow_factor=2)
    env = DummyVecEnv([lambda: Monitor(raw_env)])

    try:
        for ep in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                total_reward += rewards[0]
                done = dones[0]
            info = infos[0]
            extras = ""
            if "phase" in info:
                extras += f"  phase={info['phase']}  max_phase={info['max_phase']}"
            if "max_cube_height" in info:
                extras += f"  max_height={info['max_cube_height']:.3f}"
            print(f"Episode {ep + 1}/{episodes}: return={total_reward:.2f}{extras}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
