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
    OUTLIER_FLOOR_MM,
    _apply_gravity_slack,
    _backlash_dofs,
    generate_poses,
    reject_outliers,
    settled_site_xpos,
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
    # Gear-play settled FK: the same forward model solve_bias inverts, so clean
    # synthetic data must recover the planted bias to numerical precision.
    return settled_site_xpos(model, data, qposadr, qpos, sid)


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


def test_gravity_slack_matches_dynamic_settle(setup):
    """The quasi-static gear-play settle (sign of the gravity generalized force ->
    backlash limit) must agree with an actual dynamic settle, and the slack must
    move the tag by a real (mm-scale) amount vs rigid FK."""
    model, data, jm, site_ids = setup
    qposadr = jm.qposadr()
    play_qadr, play_dadr, play_rng = _backlash_dofs(model)
    assert len(play_qadr) == 4   # lift/elbow/wrist_flex/wrist_roll gear play wired

    poses, _ = generate_poses(model, data, jm)
    pose = poses[len(poses) // 3]   # a forward-reaching grid pose

    data.qpos[:] = 0.0
    data.qpos[qposadr] = pose
    _apply_gravity_slack(model, data, (play_qadr, play_dadr, play_rng))
    quasi = data.qpos[play_qadr].copy()
    assert np.allclose(np.abs(quasi), play_rng[:, 1])   # settled hard against a limit

    # Dynamic settle: hold the motors at the encoder pose, let only the play joints
    # evolve under gravity, and confirm each lands on the same side.
    data.qpos[:] = 0.0
    data.qpos[qposadr] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(3000):
        data.ctrl[:] = pose
        mujoco.mj_step(model, data)
        data.qpos[qposadr] = pose
        data.qvel[jm.qposadr()] = 0.0
    dynamic = data.qpos[play_qadr].copy()
    assert np.array_equal(np.sign(quasi), np.sign(dynamic)), f"{quasi} vs {dynamic}"

    # The slack actually displaces the wrist tag (not a no-op vs rigid FK).
    wrist_sid = site_ids[2]
    settled = settled_site_xpos(model, data, qposadr, pose, wrist_sid)
    data.qpos[:] = 0.0
    data.qpos[qposadr] = pose
    mujoco.mj_kinematics(model, data)
    rigid = data.site_xpos[wrist_sid].copy()
    assert np.linalg.norm(settled - rigid) > 1e-3   # > 1 mm shift from gear play


def test_rejects_corrupted_detection(setup):
    """One arm tag mis-detected at one pose pulls the naive bias off; rejection must
    drop exactly that point and recover the planted bias."""
    model, data, jm, site_ids = setup
    b_true = np.zeros(6)
    b_true[2] = np.radians(9.0)    # elbow_flex
    b_true[4] = np.radians(-5.0)   # wrist_roll
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", [8, -110, 4], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]

    samples = synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n=16, seed=7)
    bad_tag = next(iter(site_ids))
    samples[5][1][bad_tag] = (np.zeros(3), samples[5][1][bad_tag][1] + np.array([0.03, -0.02, 0.04]))

    qposadr = jm.qposadr()
    b_dirty = solve_bias(samples, model, data, qposadr, site_ids)
    kept, dropped = reject_outliers(samples, model, data, qposadr, site_ids)
    b_clean = solve_bias(kept, model, data, qposadr, site_ids)

    assert dropped == [(5, bad_tag, dropped[0][2])]   # exactly the corrupted point
    assert dropped[0][2] > OUTLIER_FLOOR_MM
    for i in OBSERVABLE_JOINTS:                        # clean solve recovers the plant
        assert abs(b_clean[i] - b_true[i]) < 1e-3, f"joint {i}: {b_clean[i]} vs {b_true[i]}"
    # The outlier genuinely mattered: the naive solve is visibly worse on some joint.
    assert max(abs(b_dirty[i] - b_true[i]) for i in OBSERVABLE_JOINTS) > 1e-2


def test_rejection_keeps_clean_data(setup):
    """On clean data nothing is dropped and the result matches the plain solve."""
    model, data, jm, site_ids = setup
    b_true = np.zeros(6)
    b_true[1] = np.radians(-3.0)
    b_true[2] = np.radians(9.0)
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", [10, -120, 5], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]

    samples = synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n=12, seed=3)
    kept, dropped = reject_outliers(samples, model, data, jm.qposadr(), site_ids)
    assert dropped == []
    assert np.allclose(solve_bias(kept, model, data, jm.qposadr(), site_ids),
                       solve_bias(samples, model, data, jm.qposadr(), site_ids))


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


def test_sweep_includes_wrist_up_poses(setup):
    """The wrist-up block contributes low poses with the gripper pitched skyward
    (approach axis toward world-up while the wrist tag stays low) — wrist_flex
    decorrelated from height for axial-bias observability."""
    model, data, jm, site_ids = setup
    poses, _ = generate_poses(model, data, jm)
    qposadr = jm.qposadr()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    wrist_sid = site_ids[2]   # marker_wrist
    n_up = 0
    for q in poses:
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        gripper_up = data.site_xmat[gid].reshape(3, 3)[2, 0]   # gripper x-axis, world z
        if gripper_up > 0.5 and data.site_xpos[wrist_sid][2] < 0.18:
            n_up += 1
    assert n_up >= 3, f"expected several low wrist-up poses, got {n_up}"


def test_calibration_io_roundtrip(tmp_path):
    qpos_bias = np.array([0.0, -0.05, 0.16, 0.03, -0.09, 0.0])
    path = tmp_path / "calibration.yaml"
    save_calibration(path, qpos_bias, n_samples=12, rms_before_mm=11.3, rms_after_mm=2.8)
    assert np.allclose(load_calibration(path), qpos_bias)
