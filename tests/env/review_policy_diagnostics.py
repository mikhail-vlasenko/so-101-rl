"""Manual checkpoint diagnostic; run explicitly with pytest -s.

Uses current scene/config and a small disjoint seed sample. This is descriptive,
not a deployment acceptance test, and does not run in normal test discovery.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from stable_baselines3 import PPO

from src.lift_env import SO101LiftEnv
from src.train import runtime_cfg_from_hydra


def test_reference_policy_approach_diagnostics():
    checkpoint = Path("logs/ppo_lift/bps_proper_grasp_light3/final_model.zip")
    if not checkpoint.exists():
        pytest.skip("Local review checkpoint is unavailable")
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "dr=light", "shaping=light"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = SO101LiftEnv(env_cfg=cfg.lift_env, cfg=runtime_cfg_from_hydra(cfg))
    try:
        policy = PPO.load(checkpoint, device="cpu")
        for seed in range(20260912, 20260924):
            obs, _ = env.reset(seed=seed)
            approach_openings = []
            saturated = []
            first_contact = None
            for step in range(env.max_steps):
                distance = np.linalg.norm(env._get_ee_pos() - env._get_cube_pos())
                angle = env.data.qpos[env.joint_qposadr[env.gripper_idx]]
                if env._n_jaw_contacts() and first_contact is None:
                    first_contact = (step, round(float(angle), 3))
                if first_contact is None and distance < 0.08:
                    approach_openings.append(float(angle))
                action, _ = policy.predict(obs, deterministic=True)
                saturated.append(float(np.mean(np.abs(action) >= 0.99)))
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(reward) and np.all(np.isfinite(obs))
                if terminated or truncated:
                    break
            opening_range = (np.round(np.quantile(approach_openings, [0, .5, 1]), 3).tolist()
                             if approach_openings else None)
            print({"seed": seed, "success": bool(info["lift_success"]),
                   "steps": step + 1, "first_jaw_contact_step_angle_rad": first_contact,
                   "precontact_near_opening_min_median_max_rad": opening_range,
                   "saturated_action_fraction": round(float(np.mean(saturated)), 3),
                   "drag_ratio": round(float(info["cube_drag_ratio"]), 3)})
    finally:
        env.close()
        torch.set_num_threads(previous_threads)
