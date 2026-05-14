"""Sanity tests for the sysid trajectory library.

Verifies every trajectory:
  - Has shape (T, 6) with T >= 2.
  - Starts within a small tolerance of HOME (so back-to-back execution after
    a homing ramp is consistent).
  - Stays inside the MuJoCo joint limits with some safety margin.
  - Stays above the floor when forward-kinematics'd through so101/scene.xml.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from base_env import JOINT_NAMES
from sysid.trajectories import HOME, TRAJECTORIES

REPO_ROOT = Path(__file__).resolve().parent.parent
XML = REPO_ROOT / "so101" / "scene.xml"
EE_Z_FLOOR = 0.02   # min EE z (m) allowed during a sysid trajectory


@pytest.fixture(scope="module")
def model_data():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = model.jnt_qposadr[joint_ids]
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    return model, data, qposadr, ee_site


@pytest.mark.parametrize("name", sorted(TRAJECTORIES.keys()))
def test_shape_and_start(name):
    traj = TRAJECTORIES[name]
    assert traj.ndim == 2 and traj.shape[1] == 6, f"{name}: shape {traj.shape}"
    assert traj.shape[0] >= 2, f"{name}: too short"
    assert np.allclose(traj[0], HOME, atol=1e-6), \
        f"{name}: must start at HOME, got {traj[0]}"


@pytest.mark.parametrize("name", sorted(TRAJECTORIES.keys()))
def test_within_joint_limits(name, model_data):
    model, _data, _qposadr, _ee = model_data
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    low = model.jnt_range[joint_ids, 0]
    high = model.jnt_range[joint_ids, 1]
    traj = TRAJECTORIES[name]
    assert np.all(traj >= low - 1e-6), f"{name}: below joint low\n{(low - traj).max()}"
    assert np.all(traj <= high + 1e-6), f"{name}: above joint high\n{(traj - high).max()}"


@pytest.mark.parametrize("name", sorted(TRAJECTORIES.keys()))
def test_ee_above_floor(name, model_data):
    model, data, qposadr, ee_site = model_data
    traj = TRAJECTORIES[name]
    for i, q in enumerate(traj):
        data.qpos[qposadr] = q
        mujoco.mj_kinematics(model, data)
        z = data.site_xpos[ee_site, 2]
        assert z >= EE_Z_FLOOR, f"{name}[{i}]: EE z={z:.3f} below floor ({EE_Z_FLOOR})"
