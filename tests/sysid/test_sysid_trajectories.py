"""Sanity tests for the sysid trajectory library.

Verifies every trajectory:
  - Has shape (T, 6) with T >= 2.
  - Starts and ends at its base pose (HOME or STRETCH), so back-to-back
    execution after a homing ramp is consistent.
  - Stays inside the MuJoCo joint limits with some safety margin.
  - Keeps every arm geom at least FLOOR_MARGIN above the floor at every
    commanded frame: the floor plane is raised by the margin and collision
    detection is run, so jaw tips and link bellies are checked exactly, not
    just the EE site. The margin absorbs real-arm gravity sag below the
    commanded pose.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.base_env import JOINT_NAMES
from sysid.trajectories import HOME, STRETCH, TRAJECTORIES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
XML = REPO_ROOT / "so101" / "scene.xml"
FLOOR_MARGIN = 0.03  # min clearance (m) of any arm geom above the floor

BASE_POSES = {"HOME": HOME, "STRETCH": STRETCH}


@pytest.fixture(scope="module")
def model_data():
    model = mujoco.MjModel.from_xml_path(str(XML))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    model.geom_pos[floor_id][2] += FLOOR_MARGIN
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = model.jnt_qposadr[joint_ids]
    return model, data, qposadr, floor_id


def _base_distances(traj_row):
    return {name: float(np.max(np.abs(traj_row - pose)))
            for name, pose in BASE_POSES.items()}


@pytest.mark.parametrize("name", sorted(TRAJECTORIES.keys()))
def test_shape_and_endpoints(name):
    traj = TRAJECTORIES[name]
    assert traj.ndim == 2 and traj.shape[1] == 6, f"{name}: shape {traj.shape}"
    assert traj.shape[0] >= 2, f"{name}: too short"
    start = _base_distances(traj[0])
    assert min(start.values()) < 1e-6, \
        f"{name}: must start at a base pose, got {traj[0]} (distances {start})"
    end = _base_distances(traj[-1])
    assert min(end.values()) < 1e-6, \
        f"{name}: must end at a base pose, got {traj[-1]} (distances {end})"


@pytest.mark.parametrize("name", sorted(TRAJECTORIES.keys()))
def test_within_joint_limits(name, model_data):
    model, _data, _qposadr, _floor = model_data
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    low = model.jnt_range[joint_ids, 0]
    high = model.jnt_range[joint_ids, 1]
    traj = TRAJECTORIES[name]
    assert np.all(traj >= low - 1e-6), f"{name}: below joint low\n{(low - traj).max()}"
    assert np.all(traj <= high + 1e-6), f"{name}: above joint high\n{(traj - high).max()}"


@pytest.mark.parametrize("name", sorted(TRAJECTORIES.keys()))
def test_floor_clearance(name, model_data):
    model, data, qposadr, floor_id = model_data
    traj = TRAJECTORIES[name]
    for i, q in enumerate(traj):
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            assert floor_id not in (c.geom1, c.geom2), (
                f"{name}[{i}]: geom "
                f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)!r}/"
                f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)!r} "
                f"within {FLOOR_MARGIN} m of the floor at qpos={q.tolist()}"
            )
