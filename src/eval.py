"""Evaluate a trained checkpoint with visual rendering.

Usage:
    python eval.py                          # latest checkpoint
    python eval.py model=best               # best_model from EvalCallback
    python eval.py model=path/to/model.zip  # specific path
    python eval.py episodes=20              # override episode count
    python eval.py algorithm=ppo            # evaluate PPO model
    python eval.py seed=42                  # fixed seed (incremented per episode)
"""

import os

import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.checkpoints import resolve_model_path
from src.train import ENV_REGISTRY, _resolve_env, make_env, runtime_cfg_from_hydra

ALGORITHM_CLASSES = {
    "sac": SAC,
    "ppo": PPO,
}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
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
    model_path = resolve_model_path(model_arg, log_dir)

    episodes = int(cfg.get("episodes", 10))
    deterministic = bool(cfg.get("deterministic", True))
    base_seed = cfg.get("seed", None)

    print(f"Loading {algo_name.upper()} model: {model_path}")
    model = algo_cls.load(model_path)

    runtime_cfg = runtime_cfg_from_hydra(cfg)
    render_mode = "human" if cfg.render else None
    inner_env = make_env(env_cls, env_cfg, xml_path, render_mode=render_mode,
                         slow_factor=cfg.slow_factor,
                         cfg=runtime_cfg)
    vec_env = DummyVecEnv([lambda: inner_env])

    publisher = None
    if cfg.stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(inner_env.model, int(cfg.stream_port))

    drag_ratios = []
    successes = []
    try:
        for ep in range(episodes):
            seed = (base_seed + ep) if base_seed is not None else int(np.random.SeedSequence().entropy % (2**31))
            vec_env.seed(seed)
            obs = vec_env.reset()
            total_reward = 0.0
            done = False
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = vec_env.step(action)
                total_reward += float(reward[0])
                done = bool(dones[0])
                info = infos[0]
                if publisher is not None:
                    publisher.publish(inner_env.data)
            extras = f"  seed={seed}"
            if "placed" in info:
                successes.append(float(info["placed"]))
                extras += f"  placed={info['placed']}"
            elif "lift_success" in info:
                successes.append(float(info["lift_success"]))
                extras += f"  lift_success={info['lift_success']}"
            if "max_cube_height" in info:
                extras += f"  max_height={info['max_cube_height']:.3f}"
            if "cube_drag_ratio" in info:
                drag_ratios.append(info["cube_drag_ratio"])
                extras += f"  drag={info['cube_drag_ratio']:.3f}"
            print(f"Episode {ep + 1}/{episodes}: return={total_reward:.2f}{extras}")
        if successes:
            print(f"\nSuccess rate over {len(successes)} episodes: {np.mean(successes):.3f}")
        if drag_ratios:
            print(f"Mean cube_drag_ratio over {len(drag_ratios)} episodes: {np.mean(drag_ratios):.3f}")
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            publisher.close()
        vec_env.close()


if __name__ == "__main__":
    main()
