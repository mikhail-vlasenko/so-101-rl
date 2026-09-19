"""Manual checkpoint diagnostic; run explicitly with pytest -s.

Uses current scene/config and a small disjoint seed sample. This is descriptive,
not a deployment acceptance test, and does not run in normal test discovery.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from src.bps import validate_checkpoint_bps
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


def test_paired_refinement_benchmark():
    """Manual benchmark configured by LIFT_BENCHMARK_{CONFIG,MODEL,OUTPUT,SEED,EPISODES}.

    CONFIG is a saved Hydra config; MODEL and OUTPUT select one policy/result.
    Repeat with the same CONFIG, SEED and EPISODES for paired starts. This measures
    current first-height-crossing success, not sustained retention or real pickup.
    Floor forces are sampled at control ticks, not substep impact peaks.
    Near-table travel uses the existing drag metric's height proxy, without its
    speed threshold; it is not a direct measurement of table contact.
    table_slide_distance_m and motion_penalty_return come from the reward's
    physics-substep contact accumulator, not reconstructed control-tick speeds.
    """
    if "LIFT_BENCHMARK_CONFIG" not in os.environ:
        pytest.skip("Set LIFT_BENCHMARK_* to run the paired refinement benchmark")
    config_path = Path(os.environ["LIFT_BENCHMARK_CONFIG"])
    checkpoint = Path(os.environ["LIFT_BENCHMARK_MODEL"])
    output = Path(os.environ["LIFT_BENCHMARK_OUTPUT"])
    seed_start = int(os.environ["LIFT_BENCHMARK_SEED"])
    episodes = int(os.environ["LIFT_BENCHMARK_EPISODES"])
    assert episodes > 0
    assert not output.exists(), f"refusing to overwrite benchmark {output}"
    cfg = OmegaConf.load(config_path)
    assert cfg.env_name == "lift"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = SO101LiftEnv(env_cfg=cfg.lift_env, cfg=runtime_cfg_from_hydra(cfg))
    rows = []
    try:
        policy = PPO.load(checkpoint, device="cpu")
        validate_checkpoint_bps(policy, env.bps_config)
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            start_qpos = env.data.qpos.copy().tolist()
            cube_start = env._get_cube_pos().copy()
            cube_previous = cube_start.copy()
            xy_path = 0.0
            grasped_xy_path = 0.0
            near_table_xy_path = 0.0
            motion_penalty_return = 0.0
            table_slide_distance = 0.0
            # The local axis closest to world-up identifies flat/side/upright.
            vertical_axis = int(np.argmax(np.abs(env.data.geom_xmat[env.cube_geom_id].reshape(3, 3)[2])))
            orientation = ("upright", "side", "flat")[vertical_axis]
            pregrasp_displacement = 0.0
            ever_grasped = False
            grasp_losses = 0
            was_grasped = False
            floor_forces = []
            first_contact_angle = None
            total_reward = 0.0
            for step in range(env.max_steps):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(reward) and np.all(np.isfinite(obs))
                total_reward += float(reward)
                cube_current = env._get_cube_pos().copy()
                xy_step = float(np.linalg.norm(cube_current[:2] - cube_previous[:2]))
                xy_path += xy_step
                motion_penalty_return += info["motion_penalty"]
                table_slide_distance += info["table_slide_distance_m"]
                if cube_current[2] < env.cube_rest_half_z + env.DRAG_HEIGHT_TOL:
                    near_table_xy_path += xy_step
                cube_previous = cube_current
                if not ever_grasped:
                    displacement = np.linalg.norm(cube_current[:2] - cube_start[:2])
                    pregrasp_displacement = max(pregrasp_displacement, float(displacement))
                grasped = bool(info["grasped"])
                if grasped:
                    grasped_xy_path += xy_step
                grasp_losses += int(was_grasped and not grasped)
                ever_grasped |= grasped
                was_grasped = grasped
                floor_forces.append(env._arm_floor_contact_force())
                if first_contact_angle is None and env._n_jaw_contacts():
                    first_contact_angle = float(env.data.qpos[env.joint_qposadr[env.gripper_idx]])
                if terminated or truncated:
                    break
            row = {
                "seed": seed, "start_qpos": start_qpos, "orientation": orientation,
                "success": bool(info["lift_success"]), "steps": step + 1,
                "return": total_reward, "ever_grasped": ever_grasped,
                "grasp_losses": grasp_losses,
                "pregrasp_max_xy_displacement_m": pregrasp_displacement,
                "xy_path_m": xy_path, "grasped_xy_path_m": grasped_xy_path,
                "near_table_xy_path_m": near_table_xy_path,
                "motion_penalty_return": motion_penalty_return,
                "table_slide_distance_m": table_slide_distance,
                "mean_floor_force_n": float(np.mean(floor_forces)),
                "max_tick_floor_force_n": float(np.max(floor_forces)),
                "first_jaw_contact_angle_rad": first_contact_angle,
                "drag_ratio": float(info["cube_drag_ratio"]),
            }
            assert table_slide_distance == pytest.approx(info["episode_table_slide_distance_m"])
            assert motion_penalty_return == pytest.approx(info["episode_motion_penalty"])
            rows.append(row)
            print(f"{checkpoint.parent.name}: {len(rows)}/{episodes} "
                  f"success={row['success']} steps={row['steps']}", flush=True)
        artifacts = [checkpoint, config_path, Path("so101/so101.xml"),
                     Path("src/lift_env.py"), Path("src/base_env.py"),
                     Path("real/follower_calibration.json"), Path("real/calib/calibration.yaml")]
        result = {
            "checkpoint": str(checkpoint),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
            "episodes": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved {output}: {sum(row['success'] for row in rows)}/{episodes} successes", flush=True)
    finally:
        env.close()
        torch.set_num_threads(previous_threads)
