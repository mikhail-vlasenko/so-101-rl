"""Evaluate a checkpoint and report per-component reward breakdown."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from stable_baselines3 import PPO

from pickplace_env import (
    SO101PickPlaceEnv, Phase,
    TIME_PENALTY, JOINT_PASSIVE_COEFF,
    XY_PROGRESS_COEFF, EE_CUBE_COEFF,
    HEIGHT_MULT_MAX, HEIGHT_MULT_CEILING,
    RETURN_BONUS, RETURN_THRESHOLD,
)

MODEL_PATH = "logs/ppo_pickplace/checkpoints/ppo_20000000_steps.zip"
N_EPISODES = 32


def run():
    with open("conf/config.yaml") as f:
        cfg = yaml.safe_load(f)

    env = SO101PickPlaceEnv(env_cfg=cfg["pickplace_env"])
    model = PPO.load(MODEL_PATH)

    # Per-episode accumulators
    all_episodes = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False

        totals = {
            "time_penalty": 0.0,
            "floor_contact": 0.0,
            "joint_passive": 0.0,
            "xy_progress": 0.0,
            "xy_regress": 0.0,
            "height_mult_avg": [],
            "reach_ee_cube": 0.0,
            "return_bonus": 0.0,
            "total_reward": 0.0,
        }
        steps = 0
        final_phase = Phase.REACH
        max_phase = Phase.REACH

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # Capture pre-step state for reward decomposition
            cube_pos_before = env._get_cube_pos()
            prev_xy_dist = env._prev_xy_dist

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            # Decompose reward
            ee_pos = env._get_ee_pos()
            cube_pos = env._get_cube_pos()
            joint_pos = env._get_joint_pos()
            ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)
            phase = env.phase

            totals["time_penalty"] += TIME_PENALTY

            if env.floor_contact_penalty and env._has_floor_contact():
                totals["floor_contact"] += env.floor_contact_penalty

            joint_dist = np.linalg.norm(joint_pos - env.passive_pose)
            totals["joint_passive"] += JOINT_PASSIVE_COEFF * joint_dist

            # XY progress/regress (recompute from pre-step state)
            xy_dist = np.linalg.norm(cube_pos[:2] - env.place_target[:2])
            xy_delta = prev_xy_dist - xy_dist

            height_frac = np.clip(cube_pos[2] / HEIGHT_MULT_CEILING, 0.0, 1.0)
            height_mult = 1.0 + (HEIGHT_MULT_MAX - 1.0) * height_frac
            totals["height_mult_avg"].append(height_mult)

            if xy_delta >= 0:
                xy_reward = XY_PROGRESS_COEFF * xy_delta * height_mult
                totals["xy_progress"] += xy_reward
            else:
                xy_reward = XY_PROGRESS_COEFF * HEIGHT_MULT_MAX * xy_delta * height_mult
                totals["xy_regress"] += xy_reward

            if phase == Phase.REACH:
                totals["reach_ee_cube"] += EE_CUBE_COEFF * ee_cube_dist

            # Return bonus
            if phase == Phase.RETURN:
                jd = np.linalg.norm(joint_pos - env.passive_pose)
                if jd < RETURN_THRESHOLD and terminated:
                    totals["return_bonus"] += RETURN_BONUS

            totals["total_reward"] += reward
            final_phase = phase
            max_phase = max(max_phase, phase)

        totals["height_mult_avg"] = np.mean(totals["height_mult_avg"]) if totals["height_mult_avg"] else 1.0
        totals["steps"] = steps
        totals["final_phase"] = final_phase.name
        totals["max_phase"] = max_phase.name
        all_episodes.append(totals)

    env.close()

    # Report
    print(f"\n{'='*60}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Episodes: {N_EPISODES}")
    print(f"{'='*60}\n")

    # Phase distribution
    phase_counts = {}
    for ep in all_episodes:
        p = ep["max_phase"]
        phase_counts[p] = phase_counts.get(p, 0) + 1
    print("Max phase reached:")
    for p in ["REACH", "PLACE", "RETURN"]:
        print(f"  {p}: {phase_counts.get(p, 0)}/{N_EPISODES}")
    print()

    # Reward breakdown
    keys = ["time_penalty", "floor_contact", "joint_passive",
            "xy_progress", "xy_regress", "reach_ee_cube", "return_bonus", "total_reward"]
    print(f"{'Component':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for k in keys:
        vals = [ep[k] for ep in all_episodes]
        print(f"{k:<20} {np.mean(vals):>10.3f} {np.std(vals):>10.3f} {np.min(vals):>10.3f} {np.max(vals):>10.3f}")

    print()
    avg_steps = np.mean([ep["steps"] for ep in all_episodes])
    avg_hmult = np.mean([ep["height_mult_avg"] for ep in all_episodes])
    print(f"Avg steps/episode: {avg_steps:.1f}")
    print(f"Avg height multiplier: {avg_hmult:.3f}")


if __name__ == "__main__":
    run()
