"""Probe joint ranges and EE pose for a few joint targets."""

import numpy as np
import mujoco
from hydra import compose, initialize
from src.lift_env import SO101LiftEnv


def main():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    env = SO101LiftEnv(env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
                       obs_noise=None, obs_bias=None)
    env.reset(seed=0)
    print("Joint names:")
    from base_env import JOINT_NAMES
    for i, n in enumerate(JOINT_NAMES):
        print(f"  {i}: {n}  range=[{env.joint_low[i]:.3f}, {env.joint_high[i]:.3f}]")

    # Try a few hand-set poses and report EE position
    poses = [
        ("zeros", np.zeros(6)),
    ]
    # Sweep shoulder_lift while elbow=-1, wrist_flex=1.5
    for lift in np.linspace(-1.5, 1.5, 7):
        poses.append((f"lift={lift:+.2f}", np.array([0.0, lift, -1.0, 1.5, 0.0, 0.0])))
    # Sweep elbow with lift=1.0, wrist_flex=1.5
    for elb in np.linspace(-1.5, 1.5, 7):
        poses.append((f"elb={elb:+.2f}", np.array([0.0, 1.0, elb, 1.5, 0.0, 0.0])))

    for label, q in poses:
        env.data.qpos[env.joint_qposadr] = q
        env.data.ctrl[:env.n_joints] = q
        mujoco.mj_forward(env.model, env.data)
        ee = env._get_ee_pos()
        print(f"{label:14s} q={np.round(q,2).tolist()} -> ee={np.round(ee,3).tolist()}")


if __name__ == "__main__":
    main()
