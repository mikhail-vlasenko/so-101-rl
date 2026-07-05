"""Scripted attempt to lift the cube in lift_env without RL.

The point: verify whether the sim is *capable* of producing a lift before we
blame the policy. A simple inverse-kinematics-free heuristic that closes a
constant offset on the cube xy, descends, closes gripper, then ascends.

Usage:
    conda run -n mujoco_env python -m scripts.scripted_lift [--render]
"""

import argparse
import time

import numpy as np
from hydra import compose, initialize

from src.base_env import RuntimeEnvConfig
from src.lift_env import SO101LiftEnv


def make_env(render=False):
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    env = SO101LiftEnv(
        env_cfg=cfg.lift_env,
        xml_path="so101/scene_lift.xml",
        render_mode="human" if render else None,
        cfg=RuntimeEnvConfig(obs_noise=None, obs_bias=None),
    )
    return env, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    env, _ = make_env(render=args.render)
    n_joints = env.n_joints

    max_heights = []
    grasp_steps = []
    for ep in range(args.n):
        env.reset(seed=ep)
        cube0 = env._get_cube_pos().copy()
        print(f"ep={ep} cube0={np.round(cube0,3).tolist()}")
        # P controller toward cube_xy with arm fully extended downward.
        # Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        max_h = env._get_cube_pos()[2]
        ep_grasp = 0
        for step in range(env.max_steps):
            ee = env._get_ee_pos()
            cube = env._get_cube_pos()
            qpos = env._get_joint_pos()

            # Heuristic phases.
            err_xy = cube[:2] - ee[:2]
            xy_dist = np.linalg.norm(err_xy)
            z_above = ee[2] - cube[2]

            action = np.zeros(n_joints, dtype=np.float32)

            # shoulder_pan ≈ joint 0 controls azimuth.
            angle = -np.arctan2(cube[1], cube[0])  # shoulder_pan sign is flipped
            # Phase pose target (calibrated via probe_joints):
            # lift=+1.0, elbow=-1.0, wflex=+1.5 puts ee≈[0.337, 0, 0.001]
            if step < 80:
                target = [angle, 0.0, -1.0, 1.0, 0.0, -1.0]  # pre-grasp pose, gripper open
            elif step < 130:
                target = [angle, 1.0, -1.0, 1.5, 0.0, -1.0]  # descend toward cube
            elif step < 180:
                target = [angle, 1.0, -1.0, 1.5, 0.0, 1.0]   # close gripper
            else:
                target = [angle, -0.5, -0.5, 0.5, 0.0, 1.0]  # lift

            for j in range(n_joints):
                action[j] = np.clip(5.0 * (target[j] - qpos[j]), -1, 1)

            _, _, term, trunc, info = env.step(action)
            if ep == 0 and step % 30 == 0:
                ee_after = env._get_ee_pos()
                q_after = env._get_joint_pos()
                print(f"  step={step} ee={np.round(ee_after,3).tolist()} "
                      f"q={np.round(q_after,2).tolist()} cube_z={env._get_cube_pos()[2]:.3f}")
            cube_z = env._get_cube_pos()[2]
            max_h = max(max_h, cube_z)
            if env._detect_grasp():
                ep_grasp += 1
            if args.render:
                time.sleep(0.01)
            if term or trunc:
                break
        max_heights.append(max_h)
        grasp_steps.append(ep_grasp)
        print(f"ep={ep} max_cube_height={max_h:.4f}  grasp_steps={ep_grasp}/{env.max_steps}")

    print(f"\nMean max_cube_height across {args.n} episodes: {np.mean(max_heights):.4f}")
    print(f"Mean grasp_steps: {np.mean(grasp_steps):.1f}")
    env.close()


if __name__ == "__main__":
    main()
