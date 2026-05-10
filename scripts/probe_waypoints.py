"""Print EE z for candidate joint poses to help pick safe reach waypoints."""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_env import JOINT_NAMES


CANDIDATES = [
    [ 0.0, -1.4, 1.2,  0.0, 0.0, 0.5],
    [ 0.7, -1.4, 1.2,  0.0, 0.0, 0.5],
    [-0.7, -1.4, 1.2,  0.0, 0.0, 0.5],
    [ 0.0, -1.4, 1.0, -0.3, 0.0, 0.1],
]


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = mujoco.MjModel.from_xml_path(os.path.join(repo, "so101/scene.xml"))
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = model.jnt_qposadr[joint_ids]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    print(f"{'EE z':>7}  pose")
    for q in CANDIDATES:
        data.qpos[qposadr] = q
        mujoco.mj_kinematics(model, data)
        ee_z = float(data.site_xpos[ee_id][2])
        print(f"{ee_z:7.3f}  {q}")


if __name__ == "__main__":
    main()
