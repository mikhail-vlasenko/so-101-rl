"""Manual one-update LR probe on an identical fresh PPO rollout buffer.

Run explicitly with RUN_LIFT_UPDATE_PROBE=1. Creates diagnostic checkpoints only;
does not change production settings. Both candidates start with the same policy,
Adam state, observations, advantages and minibatch permutation. This measures
one-update sensitivity, not convergence or the quality of a long training run.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import wandb
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.reward_norm import save_reward_norm, training_reward_env
from src.train import env_specs, resume_overrides, runtime_cfg_from_hydra


def test_one_update_learning_rate_probe():
    if os.environ.get("RUN_LIFT_UPDATE_PROBE") != "1":
        pytest.skip("Manual optimizer probe")
    output = Path("outputs/policy_diagnosis_20260919/update_probe")
    assert not output.exists()
    output.mkdir(parents=True)
    parent = Path("logs/ppo_lift/bps_table_slide_full_30m_20260915/final_model.zip")
    cfg = OmegaConf.load("outputs/table_slide_continue_30m_20260916/.hydra/config.yaml")
    # Reproduce the historical reward, not the current refinement defaults.
    cfg.lift_env.ee_cube_coeff = -0.5
    cfg.lift_env.jaw_contact_reward = 0.05
    cfg.lift_env.gripper_close_coeff = 0.05
    cfg.seed = 20260919
    specs, unused_eval, _ = env_specs(cfg, str(Path.cwd()), runtime_cfg_from_hydra(cfg),
                                    cfg.train.n_envs, "bps")
    unused_eval.close()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = training_reward_env(SubprocVecEnv(specs), cfg.train.gamma, parent)
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     name="lift_one_update_lr_probe_20260919",
                     config={"parent": str(parent), "seed": cfg.seed,
                             "learning_rates": [1e-4, 1e-5], "rollout_steps": 32768})
    try:
        source = PPO.load(parent, env=env, tensorboard_log=None, **resume_overrides(cfg))
        _, callback = source._setup_learn(32768)
        assert source.collect_rollouts(env, callback, source.rollout_buffer, cfg.ppo.n_steps)
        shared_buffer = copy.deepcopy(source.rollout_buffer)
        observations = shared_buffer.observations.reshape(-1, *source.observation_space.shape)[::16].copy()
        with torch.no_grad():
            tensor = torch.as_tensor(observations, device=source.device)
            distribution = source.policy.get_distribution(tensor).distribution
            old_mean = distribution.mean.clone()
            old_std = distribution.stddev.clone()
        results = []
        for label, learning_rate in (("lr_1e4", 1e-4), ("lr_1e5", 1e-5)):
            cfg.train.learning_rate = learning_rate
            model = PPO.load(parent, env=env, tensorboard_log=None, **resume_overrides(cfg))
            model.set_logger(configure(folder=str(output / label), format_strings=[]))
            model.rollout_buffer = copy.deepcopy(shared_buffer)
            model.num_timesteps = 32768
            np.random.seed(20260919)
            torch.manual_seed(20260919)
            model.train()
            with torch.no_grad():
                distribution = model.policy.get_distribution(tensor).distribution
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(old_mean, old_std), distribution).sum(-1)
                action_delta = (distribution.mean.clamp(-1, 1) - old_mean.clamp(-1, 1)).abs()
            row = {
                "learning_rate": learning_rate,
                "mean_kl": float(kl.mean()), "p95_kl": float(torch.quantile(kl, .95)),
                "mean_abs_action_delta": float(action_delta.mean()),
                "p95_abs_action_delta": float(torch.quantile(action_delta.flatten(), .95)),
            }
            checkpoint = output / f"{label}.zip"
            model.save(checkpoint)
            save_reward_norm(model, checkpoint)
            run.log({f"{label}/{key}": value for key, value in row.items()})
            results.append(row)
            print(row, flush=True)
        with (output / "results.json").open("x") as handle:
            json.dump({"wandb_url": run.url, "results": results}, handle, indent=2)
    finally:
        run.finish()
        env.close()
        torch.set_num_threads(old_threads)
