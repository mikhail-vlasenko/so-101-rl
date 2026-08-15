"""Integration test for the position-based calibration solve (real/calib/calib_solve.py),
no camera or servos.

We pick a synthetic camera pose and table-tag pose, then *fabricate* exactly the
measurements the camera would report (each tag expressed in the camera frame)
using the real MuJoCo model's FK for the arm sites. The position-based solve must
recover both the camera and table poses — and must keep recovering the camera
even when each arm tag is glued at a quarter turn (positions don't move when a tag
spins about its centre). `determine_quarter_turns` must still vote that glue back
for the rotation channel. These cover the position registration, the tag->site
mapping, and the glue handling end to end.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.calib.calib_solve import (
    determine_quarter_turns,
    load_rig_samples,
    load_samples,
    save_samples,
    site_mat,
    solve_camera,
    solve_table_anchors,
)
from real.calib.extrinsics import mat_inv, mat_to_rt, quarter_turn_mat
from real.marker_spec import ARM_TAG_TO_SITE, TABLE_TAG_IDS

DEFAULT_XML = Path(__file__).resolve().parent.parent.parent / "so101" / "scene_lift.xml"
NO_TURNS = {tag: 0 for tag in ARM_TAG_TO_SITE}


def make_T(rotvec, pos):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def build_model():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    qposadr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
         for n in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]])
    site_ids = {t: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                for t, s in ARM_TAG_TO_SITE.items()}
    return model, data, qposadr, site_ids


def fabricate_sample(model, data, qposadr, site_ids, qpos, T_base_cam, table_anchors,
                     glue_turns=NO_TURNS):
    """Forward-render the camera measurements for one arm pose.

    Each arm tag is glued at glue_turns[tag]·90° relative to its sim site."""
    data.qpos[qposadr] = qpos
    mujoco.mj_kinematics(model, data)
    T_cam_base = mat_inv(T_base_cam)
    poses = {tag: mat_to_rt(T_cam_base @ table_anchors[tag])
             for tag in TABLE_TAG_IDS}
    for tag_id, sid in site_ids.items():
        T_base_realtag = site_mat(data, sid) @ quarter_turn_mat(glue_turns[tag_id])
        poses[tag_id] = mat_to_rt(T_cam_base @ T_base_realtag)
    return qpos.copy(), poses


QPOSES = [
    np.array([0.0, -0.3, 0.5, 0.1, 0.0, 0.2]),
    np.array([0.4, -0.6, 0.8, -0.3, 0.5, 0.0]),
    np.array([-0.5, -0.1, 0.2, 0.4, -0.4, 0.3]),
]


def test_position_solve_recovers_extrinsics():
    model, data, qposadr, site_ids = build_model()
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    table_anchors = {
        10: make_T([0.0, 0.0, 1.4], [0.25, 0.05, 0.0]),
        11: make_T([0.0, 0.0, 3.0], [0.40, 0.05, 0.0]),
    }

    samples = [fabricate_sample(model, data, qposadr, site_ids, q,
                                T_base_cam, table_anchors) for q in QPOSES]

    cam, rms_mm, tags, err_mm = solve_camera(samples, model, data, qposadr, site_ids)
    solved_anchors, _, _ = solve_table_anchors(samples, cam)
    assert rms_mm < 1e-3
    assert len(tags) == 6  # 3 poses x 2 arm tags
    np.testing.assert_allclose(cam, T_base_cam, atol=1e-6)
    for tag in TABLE_TAG_IDS:
        np.testing.assert_allclose(solved_anchors[tag], table_anchors[tag], atol=1e-6)


def test_position_solve_immune_to_glue():
    # Tags glued at quarter turns must NOT move the position solve — that's the
    # whole reason we register from centres instead of bridging orientation.
    model, data, qposadr, site_ids = build_model()
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    table_anchors = {
        10: make_T([0.0, 0.0, 1.4], [0.25, 0.05, 0.0]),
        11: make_T([0.0, 0.0, 3.0], [0.40, 0.05, 0.0]),
    }
    glue = dict(zip(site_ids, [1, 3]))

    samples = [fabricate_sample(model, data, qposadr, site_ids, q,
                                T_base_cam, table_anchors, glue_turns=glue) for q in QPOSES]
    cam, rms_mm, _, _ = solve_camera(samples, model, data, qposadr, site_ids)
    assert rms_mm < 1e-3
    np.testing.assert_allclose(cam, T_base_cam, atol=1e-6)


def test_glue_offset_voted():
    model, data, qposadr, site_ids = build_model()
    T_base_cam = make_T([0.1, -0.5, 0.2], [0.12, -0.42, 0.30])
    table_anchors = {
        10: make_T([0.0, 0.0, 1.4], [0.25, 0.05, 0.0]),
        11: make_T([0.0, 0.0, 3.0], [0.40, 0.05, 0.0]),
    }
    glue = dict(zip(site_ids, [1, 3]))  # finger 90°, wrist 270° glued wrong

    samples = [fabricate_sample(model, data, qposadr, site_ids, q,
                                T_base_cam, table_anchors, glue_turns=glue) for q in QPOSES]
    # The true camera rotation stands in for the (approximate) sim-camera prior.
    quarter_turns, report = determine_quarter_turns(
        samples, model, data, qposadr, site_ids, T_base_cam[:3, :3])
    assert quarter_turns == glue
    for _, res, _ in report.values():
        assert res.max() < 1e-6   # a clean quarter turn leaves no residual


def _assert_poses_equal(actual, expected):
    assert set(actual) == set(expected)
    for tag in expected:
        np.testing.assert_allclose(actual[tag][0], expected[tag][0])
        np.testing.assert_allclose(actual[tag][1], expected[tag][1])


def test_stereo_sample_round_trip_preserves_native_measurements(tmp_path):
    qpos = np.linspace(-0.3, 0.3, 6)
    fused = {
        0: (np.array([0.1, 0.2, 0.3]), np.array([0.04, -0.02, 0.35])),
        10: (np.array([0.0, 0.0, 0.2]), np.array([0.10, 0.03, 0.42])),
    }
    camera_samples = [{
        "main": {
            0: (np.array([0.11, 0.21, 0.31]), np.array([0.041, -0.021, 0.351])),
            10: (np.array([0.0, 0.0, 0.21]), np.array([0.101, 0.03, 0.421])),
        },
        "aux": {
            0: (np.array([0.09, 0.19, 0.29]), np.array([-0.06, -0.02, 0.36])),
            10: (np.array([0.0, 0.0, 0.19]), np.array([0.001, 0.03, 0.43])),
        },
    }]
    T_aux_main = make_T([0.01, -0.02, 0.03], [-0.105, 0.002, -0.01])
    path = tmp_path / "samples.json"

    save_samples(
        path, [(qpos, fused)], camera_samples=camera_samples,
        T_aux_main=T_aux_main)

    loaded = load_samples(path)
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0][0], qpos)
    _assert_poses_equal(loaded[0][1], fused)

    rig_samples, loaded_cameras, loaded_T = load_rig_samples(path)
    np.testing.assert_allclose(rig_samples[0][0], qpos)
    _assert_poses_equal(rig_samples[0][1], fused)
    for camera in ("main", "aux"):
        _assert_poses_equal(loaded_cameras[0][camera], camera_samples[0][camera])
    np.testing.assert_allclose(loaded_T, T_aux_main)


def test_legacy_sample_file_still_loads(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "samples": [{
            "qpos": [0.0] * 6,
            "tags": {"0": {"rvec": [0.1, 0.2, 0.3],
                            "tvec": [0.04, -0.02, 0.35]}},
        }],
    }))

    samples = load_samples(path)

    assert len(samples) == 1
    np.testing.assert_allclose(samples[0][0], 0.0)
    np.testing.assert_allclose(samples[0][1][0][1], [0.04, -0.02, 0.35])
    with pytest.raises(RuntimeError, match="does not contain preserved stereo"):
        load_rig_samples(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
