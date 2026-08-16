"""Evaluate a trained checkpoint with visual rendering.

Usage:
    python eval.py                          # latest checkpoint
    python eval.py model=best               # best_model from EvalCallback
    python eval.py model=path/to/model.zip  # specific path
    python eval.py episodes=20              # override episode count
    python eval.py algorithm=ppo            # evaluate PPO model
    python eval.py seed=42                  # fixed seed (incremented per episode)
    python eval.py env=lift model=old.zip teacher_obs=legacy_tag
"""

import os

import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.checkpoints import resolve_model_path
from src.bps import validate_checkpoint_bps
from src.base_env import (
    legacy_tag_actor_dim_for,
    obs_dim_for,
    priv_dim_for,
    state_dim_for,
)
from src.teacher_obs import teacher_observation, validate_teacher_obs_dim
from src.train import _resolve_env, make_env, runtime_cfg_from_hydra

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
    teacher_obs_mode = cfg.teacher_obs
    if teacher_obs_mode is None and cfg.env_name != "reach":
        validate_checkpoint_bps(model, runtime_cfg.bps_config)
    render_mode = "human" if cfg.render else None
    inner_env = make_env(env_cls, env_cfg, xml_path, render_mode=render_mode,
                         slow_factor=cfg.slow_factor,
                         cfg=runtime_cfg)
    vec_env = DummyVecEnv([lambda: inner_env])

    teacher_dims = None
    if teacher_obs_mode is not None:
        assert cfg.env_name != "reach", \
            "teacher_obs adapters apply only to cube environments"
        single_actor_dim = obs_dim_for(
            int(cfg.prev_actions_n), bool(cfg.marker_include_rot))
        state_dim = state_dim_for(
            int(cfg.prev_actions_n), bool(cfg.marker_include_rot))
        priv_dim = priv_dim_for(bool(cfg.marker_include_rot))
        legacy_actor_dim = legacy_tag_actor_dim_for(
            int(cfg.prev_actions_n), bool(cfg.marker_include_rot))
        teacher_obs_dim = model.observation_space.shape[0]
        validate_teacher_obs_dim(
            teacher_obs_mode, teacher_obs_dim,
            inner_env.observation_space.shape[0], single_actor_dim, priv_dim,
            legacy_actor_dim)
        if teacher_obs_mode != "legacy_tag":
            validate_checkpoint_bps(model, runtime_cfg.bps_config)
        teacher_dims = (state_dim, priv_dim, legacy_actor_dim, teacher_obs_dim)

    publisher = None
    if cfg.stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(inner_env.model, int(cfg.stream_port))

    drag_ratios = []
    successes = []
    grasp_ratios = []
    two_jaw_contact_ratios = []
    try:
        for ep in range(episodes):
            seed = (base_seed + ep) if base_seed is not None else int(np.random.SeedSequence().entropy % (2**31))
            vec_env.seed(seed)
            obs = vec_env.reset()
            total_reward = 0.0
            done = False
            info = {}
            while not done:
                policy_obs = obs
                if teacher_dims is not None:
                    policy_obs = teacher_observation(
                        vec_env, obs, teacher_obs_mode, *teacher_dims)
                action, _ = model.predict(policy_obs, deterministic=deterministic)
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
                grasp_ratios.append(info["grasp_ratio"])
                two_jaw_contact_ratios.append(info["two_jaw_contact_ratio"])
                extras += f"  lift_success={info['lift_success']}"
                extras += f"  grasp={info['grasp_ratio']:.3f}"
                extras += f"  two_jaw={info['two_jaw_contact_ratio']:.3f}"
            if "max_cube_height" in info:
                extras += f"  max_height={info['max_cube_height']:.3f}"
            if "cube_drag_ratio" in info:
                drag_ratios.append(info["cube_drag_ratio"])
                extras += f"  drag={info['cube_drag_ratio']:.3f}"
            print(f"Episode {ep + 1}/{episodes}: return={total_reward:.2f}{extras}")
        if successes:
            print(f"\nSuccess rate over {len(successes)} episodes: {np.mean(successes):.3f}")
        if grasp_ratios:
            print(f"Mean proper grasp_ratio: {np.mean(grasp_ratios):.3f}")
            print(f"Mean raw two_jaw_contact_ratio: "
                  f"{np.mean(two_jaw_contact_ratios):.3f}")
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
