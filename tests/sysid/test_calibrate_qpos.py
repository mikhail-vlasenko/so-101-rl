"""Contract tests for the encoder-bias calibration (real/calib/calibrate_qpos.py).

The solver is exercised on synthetic data with a known bias and camera pose, so
the optimisation math is checked independent of real-rig noise: clean data must
recover the planted biases to numerical precision. Pose generation and the
calibration YAML round-trip are checked too.
"""
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import real.calib.calibrate_qpos as cq
from real.calib.calibrate_qpos import (
    GRIPPER_FIXED,
    IK_SEED,
    MIN_EE_Z,
    OBSERVABLE_JOINTS,
    OUTLIER_FLOOR_MM,
    _apply_gravity_slack,
    _backlash_dofs,
    _true_poses,
    clears_observed_real_contacts,
    generate_poses,
    reject_outliers,
    settled_site_xpos,
    solve_bias,
    solve_calibration,
    verify_drive_safe,
    write_marker_sites,
)
from real.calib.calibration import load_calibration, load_compliance, save_calibration
from real.calib.compliance import COMP_JOINTS, gravity_deflection
from real.calib.extrinsics import mat_inv, mat_to_rt, rt_to_mat
from real.marker_spec import ARM_TAG_TO_SITE
from real.twin.mapping import load_joint_maps
from src.base_env import MARKER_SITE_NAMES, markers_visible, tag_cam_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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


def synth_samples_compliant(model, data, jm, site_ids, b_true, comp_true,
                            T_base_cam, n, seed):
    """Like synth_samples but the true link pose also carries a known gravity
    compliance: q_true = (q_enc - b_true) - deflection(q_enc - b_true), so the solve
    must recover both b_true and comp_true to invert the full forward model."""
    rng = np.random.default_rng(seed)
    qposadr = jm.qposadr()
    lo, hi = jm.xml_low(), jm.xml_high()
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    samples = []
    for _ in range(n):
        qpos = rng.uniform(lo, hi)
        q_bc = qpos - b_true
        q_true = q_bc - gravity_deflection(model, data, qposadr, q_bc, comp_true)
        poses = {}
        for tag, sid in site_ids.items():
            p_base = fk_site(model, data, qposadr, q_true, sid)
            poses[tag] = (np.zeros(3), R.T @ (p_base - t))
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


def test_dual_camera_capture_averages_positions_in_main_frame():
    T_aux_main = np.eye(4)
    T_aux_main[:3, :3] = Rotation.from_euler(
        "xyz", [2.0, -1.0, 3.0], degrees=True).as_matrix()
    T_aux_main[:3, 3] = [-0.11, 0.004, -0.017]
    T_main_tag = np.eye(4)
    T_main_tag[:3, :3] = Rotation.from_euler(
        "xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix()
    T_main_tag[:3, 3] = [0.08, -0.03, 0.42]
    main_measured = T_main_tag.copy()
    main_measured[:3, 3] += [0.002, -0.001, 0.0]
    aux_in_main = T_main_tag.copy()
    aux_in_main[:3, 3] += [-0.002, 0.001, 0.0]
    camera_poses = {
        "main": {0: mat_to_rt(main_measured)},
        "aux": {0: mat_to_rt(T_aux_main @ aux_in_main)},
    }

    fused = cq.fuse_rig_tag_poses(camera_poses, T_aux_main)
    T_fused = rt_to_mat(*fused[0])

    np.testing.assert_allclose(T_fused[:3, 3], T_main_tag[:3, 3], atol=1e-12)
    np.testing.assert_allclose(T_fused[:3, :3], main_measured[:3, :3], atol=1e-12)


def test_dual_camera_capture_uses_aux_only_tag_in_main_frame():
    T_aux_main = np.eye(4)
    T_aux_main[:3, 3] = [-0.1, 0.0, 0.0]
    T_main_tag = np.eye(4)
    T_main_tag[:3, 3] = [0.2, 0.1, 0.3]
    camera_poses = {
        "main": {},
        "aux": {2: mat_to_rt(T_aux_main @ T_main_tag)},
    }

    fused = cq.fuse_rig_tag_poses(camera_poses, T_aux_main)
    np.testing.assert_allclose(rt_to_mat(*fused[2]), T_main_tag, atol=1e-12)


def test_joint_board_anchors_recover_live_relative_camera_pose():
    T_aux_main = np.eye(4)
    T_aux_main[:3, :3] = Rotation.from_euler(
        "xyz", [1.0, 2.0, -3.0], degrees=True).as_matrix()
    T_aux_main[:3, 3] = [-0.11, 0.004, -0.017]
    T_base_main = np.eye(4)
    T_base_main[:3, :3] = Rotation.from_euler(
        "xyz", [10.0, -20.0, 30.0], degrees=True).as_matrix()
    T_base_main[:3, 3] = [0.15, -0.38, 0.21]
    camera_anchors = {
        "main": T_base_main,
        "aux": T_base_main @ mat_inv(T_aux_main),
    }

    measured = cq.measured_relative_camera_pose(camera_anchors)
    np.testing.assert_allclose(measured, T_aux_main, atol=1e-12)


def test_recovers_planted_mount_offset(setup, monkeypatch):
    """The joint solve recovers a planted *wrist*-tag centre offset together with the
    lift/elbow/wrist_flex biases: clean data pins both to precision, so the bias+mount
    math (offset tag site -> settled FK -> Umeyama camera) is correct. The wrist tag is
    fully observable (the finger tag, distal to wrist_flex, breaks the wrist-offset/
    wrist_flex degeneracy). Roll and the finger offset are *left at zero* here: they are
    an exact gauge pair (see test_prior_keeps_wrist_roll_observable), so a tiny prior
    pins that one flat direction without measurably shrinking the wrist offset."""
    monkeypatch.setattr(cq, "MOUNT_PRIOR_W", 0.5)
    model, data, jm, site_ids = setup
    qposadr = jm.qposadr()
    wrist_tag = 2
    b_true = np.zeros(6)
    b_true[1], b_true[2], b_true[3] = np.radians([-3.0, 9.0, 2.0])   # no roll bias
    wrist_off = np.array([0.004, -0.006, 0.005])
    nominal = model.site_pos[site_ids[wrist_tag]].copy()
    model.site_pos[site_ids[wrist_tag]] = nominal + wrist_off        # plant the physical mount
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", [10, -120, 5], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]
    samples = synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n=20, seed=11)
    model.site_pos[site_ids[wrist_tag]] = nominal                    # solver starts from XML

    b_est, off_est, comp_est = solve_calibration(samples, model, data, qposadr, site_ids)
    for i in OBSERVABLE_JOINTS:
        assert abs(b_est[i] - b_true[i]) < 1e-3, f"joint {i}: {b_est[i]} vs {b_true[i]}"
    assert np.linalg.norm(off_est[wrist_tag] - wrist_off) < 5e-4, off_est[wrist_tag]
    assert np.linalg.norm(off_est[0]) < 5e-4         # finger offset stays at XML (none planted)
    assert np.allclose(comp_est, 0.0, atol=2e-3)     # no compliance planted -> fits ~0


def test_prior_keeps_wrist_roll_observable(setup):
    """The finger offset is an exact gauge pair with wrist_roll bias. Freeing it with
    the production prior must resolve that gauge in favour of the bias: a planted roll
    bias is recovered and the offsets stay at ~XML (the prior trusts the mount)."""
    model, data, jm, site_ids = setup
    b_true = np.zeros(6)
    b_true[1], b_true[2], b_true[4] = np.radians([-3.0, 9.0, -5.0])   # incl. wrist_roll
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", [10, -120, 5], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]
    samples = synth_samples(model, data, jm, site_ids, b_true, T_base_cam, n=20, seed=12)

    b_est, off_est, comp_est = solve_calibration(samples, model, data, jm.qposadr(), site_ids)
    for i in OBSERVABLE_JOINTS:
        assert abs(b_est[i] - b_true[i]) < 5e-3, f"joint {i}: {b_est[i]} vs {b_true[i]}"
    for tag in site_ids:
        assert np.linalg.norm(off_est[tag]) < 1e-3   # prior keeps a zero-offset fit at XML
    assert np.allclose(comp_est, 0.0, atol=2e-3)     # no compliance planted -> fits ~0


def test_recovers_planted_compliance(setup):
    """Clean data generated WITH a known per-joint gravity compliance recovers both the
    biases and the compliance coefficients, and a bias-only fit -- which cannot absorb
    the load-dependent deflection -- leaves a much larger residual. Pins the
    bias+compliance forward model and its inverse in the solve."""
    model, data, jm, site_ids = setup
    qposadr = jm.qposadr()
    b_true = np.zeros(6)
    b_true[1], b_true[2], b_true[4] = np.radians([-3.0, 9.0, -5.0])
    comp_true = np.zeros(6)
    comp_true[list(COMP_JOINTS)] = [0.10, 0.05, -0.20]   # rad per N.m
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", [10, -120, 5], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [0.12, -0.42, 0.15]
    samples = synth_samples_compliant(model, data, jm, site_ids, b_true, comp_true,
                                      T_base_cam, n=30, seed=21)

    b_est, off_est, comp_est = solve_calibration(samples, model, data, qposadr, site_ids)
    for i in OBSERVABLE_JOINTS:
        assert abs(b_est[i] - b_true[i]) < 5e-3, f"bias joint {i}: {b_est[i]} vs {b_true[i]}"
    for j in COMP_JOINTS:
        assert abs(comp_est[j] - comp_true[j]) < 5e-3, \
            f"compliance joint {j}: {comp_est[j]} vs {comp_true[j]}"

    # Compliance genuinely mattered: a bias-only fit can't explain the deflection.
    from real.calib.calibrate_qpos import solve_camera
    b_only = solve_bias(samples, model, data, qposadr, site_ids)
    corr_only = [(qpos - b_only, poses) for qpos, poses in samples]
    _, rms_only, _, _ = solve_camera(corr_only, model, data, qposadr, site_ids)
    corr_full = _true_poses(samples, model, data, qposadr, b_est, comp_est)
    _, rms_full, _, _ = solve_camera(corr_full, model, data, qposadr, site_ids)
    assert rms_full < 0.3 * rms_only, f"full {rms_full:.2f} vs bias-only {rms_only:.2f} mm"


def test_gravity_slack_matches_dynamic_settle(setup):
    """The quasi-static gear-play settle (sign of the gravity generalized force ->
    backlash limit) must agree with an actual dynamic settle, and the slack must
    move the tag by a real (mm-scale) amount vs rigid FK."""
    model, data, jm, site_ids = setup
    qposadr = jm.qposadr()
    play_qadr, play_dadr, play_rng = _backlash_dofs(model)
    assert len(play_qadr) == 4   # lift/elbow/wrist_flex/wrist_roll gear play wired

    # This dynamics contract must not depend on the visibility-filtered calibration
    # sweep: remounting a camera legitimately changes that sweep's membership/order.
    pose = np.array([0.0, *IK_SEED, GRIPPER_FIXED])

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
    # Threshold scales with the model's play range so it tracks the backlash
    # class: play_rad * 0.1 m is a 10 cm effective lever, well under the real
    # ~0.3 m reach (±0.2° measured play shifts the tag ~0.5 mm at this pose).
    wrist_sid = site_ids[2]
    settled = settled_site_xpos(model, data, qposadr, pose, wrist_sid)
    data.qpos[:] = 0.0
    data.qpos[qposadr] = pose
    mujoco.mj_kinematics(model, data)
    rigid = data.site_xpos[wrist_sid].copy()
    min_shift = play_rng[:, 1].max() * 0.1
    assert np.linalg.norm(settled - rigid) > min_shift


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
    cam = tag_cam_model(model, data)
    arm_sites = list(site_ids.values())
    for q in poses:
        assert np.all(q >= lo) and np.all(q <= hi)
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
        assert markers_visible(data, arm_sites, cam).all()   # both tags visible


def test_sweep_avoids_observed_real_contact_regions(setup):
    model, data, jm, _ = setup
    poses, _ = generate_poses(model, data, jm)
    assert all(clears_observed_real_contacts(q) for q in poses)


def test_sweep_covers_medium_reach_wrist_up_hover(setup):
    model, data, jm, _ = setup
    poses, _ = generate_poses(model, data, jm)
    qposadr = jm.qposadr()
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    covered = []
    for q in poses:
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        x, _, z = data.site_xpos[ee_id]
        if 0.21 <= x <= 0.33 and 0.035 <= z <= 0.09 and q[2] > 1.2 and q[3] < -0.4:
            covered.append(q)
    assert len(covered) >= 3


def test_sweep_exercises_every_joint(setup):
    """Each non-pinned joint must span a real range across the sweep so its bias is
    observable, and every consecutive (drive-order) move must touch a joint."""
    model, data, jm, site_ids = setup
    poses, _ = generate_poses(model, data, jm)
    P = np.array(poses)
    for j in (1, 2, 3, 4):               # lift, elbow, wrist_flex, wrist_roll
        assert np.ptp(P[:, j]) > 0.2, f"joint {j} barely moves: span {np.ptp(P[:, j])}"
    # The observed base-contact exclusion removes the risky negative-35-degree
    # branch; both signs at 17.5 degrees plus the safe positive-35 branch remain.
    assert P[:, 0].min() <= np.radians(-17.5)
    assert P[:, 0].max() >= np.radians(35.0)
    assert np.ptp(P[:, 3]) > 0.8         # wrist_flex stays observable without the wrist-up block
    assert np.ptp(P[:, 4]) > 1.0         # wrist_roll sweeps wide
    adj = np.abs(np.diff(P, axis=0))
    assert np.all(adj[:, 1:5].max(axis=1) > 0.01)   # every neighbour move touches a joint


def test_sweep_hugs_the_table(setup):
    """The sweep concentrates near the table (where the bias matters for a grasp):
    every realized fingertip stays above the safety floor yet the band is low, and at
    least one marker reaches the contact regime the old high band never sampled."""
    model, data, jm, site_ids = setup
    poses, _ = generate_poses(model, data, jm)
    qposadr = jm.qposadr()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    marker_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                   for n in MARKER_SITE_NAMES]
    ee_z, min_marker_z = [], []
    for q in poses:
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        ee_z.append(float(data.site_xpos[gid][2]))
        min_marker_z.append(min(float(data.site_xpos[s][2]) for s in marker_sids))
    ee_z, min_marker_z = np.array(ee_z), np.array(min_marker_z)

    assert ee_z.min() >= MIN_EE_Z - 1e-9        # the fingertip floor holds for every pose
    assert ee_z.max() < 0.12                    # nothing reaches up high (old band hit 0.27)
    assert ee_z.mean() < 0.07                    # the band is concentrated low
    assert min_marker_z.min() < 0.09             # a marker reaches the contact regime


def test_sweep_is_drive_safe(setup):
    """The straight-line interpolation drive_to ramps along between consecutive poses
    must be collision-free in sim, and consecutive poses must stay close so it never
    has to dive toward the table to get between two samples."""
    model, data, jm, site_ids = setup
    poses, _ = generate_poses(model, data, jm)
    qposadr = jm.qposadr()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    # verify_drive_safe raises if any transition collides; the sag stays small.
    sag = verify_drive_safe(model, data, qposadr, gid, poses)
    assert sag < 0.02, f"interpolation sags {sag * 1000:.0f} mm below endpoints"

    # No single transition swings a joint wildly (the old block boundaries hit ~180 deg).
    jumps = np.abs(np.diff(np.array(poses), axis=0))
    assert jumps.max() < np.radians(100), f"max joint jump {np.degrees(jumps.max()):.0f} deg"


def test_calibration_io_roundtrip(tmp_path):
    qpos_bias = np.array([0.0, -0.05, 0.16, 0.03, -0.09, 0.0])
    compliance = np.array([0.0, 0.15, 0.07, -0.55, 0.0, 0.0])
    path = tmp_path / "calibration.yaml"
    save_calibration(path, qpos_bias, compliance, n_samples=12,
                     rms_before_mm=11.3, rms_after_mm=2.8)
    assert np.allclose(load_calibration(path), qpos_bias)
    assert np.allclose(load_compliance(path), compliance)


SITE_XML = """<mujoco>
  <worldbody>
    <body name="b">
      <site name="marker_wrist" type="box" size="0.01 0.01 0.0005"
            pos="0 -0.020 -0.0095" quat="0 1 0 0" rgba="0.9 0.1 0.1 1"/>
      <site name="other" pos="1 2 3"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_write_marker_sites_updates_only_pos(tmp_path):
    xml = tmp_path / "arm.xml"
    xml.write_text(SITE_XML)
    write_marker_sites(xml, {"marker_wrist": np.array([0.0017, -0.0221, -0.0099])})
    text = xml.read_text()
    assert 'pos="0.0017 -0.0221 -0.0099"' in text
    assert 'quat="0 1 0 0"' in text                    # attributes around pos untouched
    assert '<site name="other" pos="1 2 3"/>' in text  # other sites untouched


def test_write_marker_sites_fails_loud(tmp_path):
    xml = tmp_path / "arm.xml"
    xml.write_text(SITE_XML)
    with pytest.raises(AssertionError, match="not found"):
        write_marker_sites(xml, {"marker_nope": np.zeros(3)})
    xml.write_text(SITE_XML.replace('<site name="other" pos="1 2 3"/>',
                                    '<site name="marker_wrist" pos="1 2 3"/>'))
    with pytest.raises(AssertionError, match="not unique"):
        write_marker_sites(xml, {"marker_wrist": np.zeros(3)})
    xml.write_text(SITE_XML.replace('pos="0 -0.020 -0.0095" ', ""))
    with pytest.raises(AssertionError, match="no pos attribute"):
        write_marker_sites(xml, {"marker_wrist": np.zeros(3)})


def test_write_marker_sites_handles_real_xml(tmp_path, setup):
    """The production so101.xml's multi-line site elements round-trip: the written
    positions parse back exactly and nothing outside the two pos attributes moves."""
    model, _, _, site_ids = setup
    xml = tmp_path / "so101.xml"
    xml.write_text((REPO_ROOT / "so101" / "so101.xml").read_text())
    new = {ARM_TAG_TO_SITE[tag]: model.site_pos[sid] + np.array([0.002, -0.0021, 0.0037])
           for tag, sid in site_ids.items()}
    write_marker_sites(xml, new)
    text = xml.read_text()
    for name, pos in new.items():
        element = re.search(rf'<site name="{name}".*?/>', text, re.S).group(0)
        written = [float(v) for v in re.search(r'pos="([^"]+)"', element).group(1).split()]
        assert np.allclose(written, pos, atol=1e-6), f"{name}: {written} vs {pos}"
