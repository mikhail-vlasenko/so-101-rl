"""Contract tests for the encoder-bias calibration (real/calibrate_qpos.py).

The solver is exercised on synthetic data with a known bias and camera pose, so
the optimisation math is checked independent of real-rig noise: clean data must
recover the planted biases to numerical precision. Pose generation and the
calibration YAML round-trip are checked too.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.calibrate_qpos import (
    OBSERVABLE_JOINTS,
    generate_poses,
    solve_bias,
)
from real.calibration import load_calibration, save_calibration
from real.marker_spec import ARM_TAG_TO_SITE
from real.twin.mapping import load_joint_maps
from src.base_env import markers_visible, tag_cam_world_pos

REPO_ROOT = Path(__file__).resolve().parent.parent
XML = REPO_ROOT / "so101" / "scene.xml"
CAL = REPO_ROOT / "real" / "follower_calibration.json"


@pytest.fixture
def setup():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, CAL)
    site_ids = {tag: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                for tag, name in ARM_TAG_TO_SITE.items()}
    return model, data, jm, site_ids


def fk_site(model, data, qposadr, qpos, sid):
    data.qpos[qposadr] = qpos
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[sid].copy()


def synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n, seed):
    """Build (qpos, {tag: (rvec, tvec)}) samples consistent with b_true & camera."""
    rng = np.random.default_rng(seed)
    qposadr = jm.qposadr()
    lo, hi = jm.xml_low(), jm.xml_high()
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    samples = []
    for _ in range(n):
        qpos = rng.uniform(lo, hi)
        poses = {}
        for tag, sid in site_ids.items():
            p_base = fk_site(model, data, qposadr, qpos - b_true, sid)
            p_cam = R.T @ (p_base - t)          # invert T_base_cam: base -> camera
            poses[tag] = (np.zeros(3), p_cam)   # rvec unused by the position solve
        samples.append((qpos, poses))
    return samples


def test_recovers_planted_bias(setup):
    model, data, jm, site_ids = setup
    b_true = np.zeros(6)
    b_true[1] = np.radians(-3.0)   # shoulder_lift
    b_true[2] = np.radians(9.0)    # elbow_flex
    b_true[3] = np.radians(2.0)    # wrist_flex
    b_true[4] = np.radians(-5.0)   # wrist_roll
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", [10, -120, 5], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]

    samples = synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n=12, seed=1)
    b_est = solve_bias(samples, model, data, jm.qposadr(), site_ids)

    for i in OBSERVABLE_JOINTS:
        assert abs(b_est[i] - b_true[i]) < 1e-4, f"joint {i}: {b_est[i]} vs {b_true[i]}"
    assert b_est[0] == 0.0 and b_est[5] == 0.0   # pan & gripper pinned


def test_pan_bias_is_gauge_absorbed(setup):
    """A pure pan bias is unobservable (camera-yaw degenerate): the solver must
    NOT invent observable-joint biases to explain it."""
    model, data, jm, site_ids = setup
    b_true = np.zeros(6)
    b_true[0] = np.radians(7.0)    # pan only
    T_base_cam = np.eye(4)
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]
    samples = synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n=12, seed=2)
    b_est = solve_bias(samples, model, data, jm.qposadr(), site_ids)
    assert np.allclose(b_est, 0.0, atol=1e-4)   # camera absorbs the pan, biases stay 0


def test_generated_poses_are_valid(setup):
    model, data, jm, site_ids = setup
    poses, n_grid = generate_poses(model, data, jm)
    assert n_grid >= 24                  # the pan=middle grid should mostly solve
    assert len(poses) > n_grid           # plus some panned poses
    qposadr = jm.qposadr()
    lo, hi = jm.xml_low(), jm.xml_high()
    cam_pos = tag_cam_world_pos(model, data)
    arm_sites = list(site_ids.values())
    for q in poses:
        assert np.all(q >= lo) and np.all(q <= hi)
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
        assert markers_visible(data, arm_sites, cam_pos).all()   # both tags visible


def test_sweep_exercises_every_joint(setup):
    """Each non-pinned joint must span a real range across the sweep, and within
    the pan=middle grid no two neighbours may share the whole configuration."""
    model, data, jm, site_ids = setup
    poses, n_grid = generate_poses(model, data, jm)
    P = np.array(poses)
    for j in (1, 2, 3, 4):               # lift, elbow, wrist_flex, wrist_roll
        assert np.ptp(P[:, j]) > 0.2, f"joint {j} barely moves: span {np.ptp(P[:, j])}"
    assert np.ptp(P[:, 0]) > 1.0         # pan spans the panned blocks
    assert np.ptp(P[:n_grid, 4]) > 0.5   # wrist_roll sweeps wide within the pan=middle grid
    adj = np.abs(np.diff(P[:n_grid], axis=0))
    assert np.all(adj[:, 1:5].max(axis=1) > 0.01)   # every neighbour move touches a joint


def test_calibration_io_roundtrip(tmp_path):
    qpos_bias = np.array([0.0, -0.05, 0.16, 0.03, -0.09, 0.0])
    path = tmp_path / "calibration.yaml"
    save_calibration(path, qpos_bias, n_samples=12, rms_before_mm=11.3, rms_after_mm=2.8)
    assert np.allclose(load_calibration(path), qpos_bias)
