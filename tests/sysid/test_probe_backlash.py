"""Contract tests for the backlash/hysteresis probe (sysid/probe_backlash.py).

The analysis is exercised on synthetic records with a known play width and
motor-side hysteresis: the camera sees the *link* pose, the encoder the *motor*
pose, and the Jacobian projection must recover the planted difference. The drive
plan is checked for safety and pairing structure, and the records JSON round-trips.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.marker_spec import ARM_TAG_TO_SITE, TABLE_TAG_IDS
from real.twin.mapping import JOINT_NAMES, load_joint_maps
from sysid.probe_backlash import (
    MIN_LEVER_M,
    PROBE_JOINTS,
    build_plan,
    hysteresis_estimates,
    load_records,
    save_records,
)

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


def synth_pair(model, data, jm, site_ids, T_base_cam, q, joint, play_rad, enc_hyst_rad):
    """Minus/plus records for one probe pair: the encoder reads the motor pose, the
    camera sees the link pose displaced ±play/2 about it (plus the motor's own
    hysteresis on the plus side), so link Δ − enc Δ = play by construction."""
    j = JOINT_NAMES.index(joint)
    qposadr = jm.qposadr()
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    records = []
    for side, enc_off, link_off in (("minus", 0.0, -play_rad / 2.0),
                                    ("plus", enc_hyst_rad, enc_hyst_rad + play_rad / 2.0)):
        q_enc = q.copy()
        q_enc[j] += enc_off
        q_link = q.copy()
        q_link[j] += link_off
        data.qpos[:] = 0.0
        data.qpos[qposadr] = q_link
        mujoco.mj_kinematics(model, data)
        tags = {str(tag): (R.T @ (data.site_xpos[sid] - t)).tolist()
                for tag, sid in site_ids.items()}
        records.append({"pose": 0, "joint": joint, "side": side,
                        "qpos": q_enc.tolist(), "tags": tags})
    return records


BASE_Q = np.array([0.1, -0.5, 0.9, 0.5, 0.4, 0.5])
T_BASE_CAM = np.eye(4)
T_BASE_CAM[:3, :3] = Rotation.from_euler("xyz", [10, -120, 5], degrees=True).as_matrix()
T_BASE_CAM[:3, 3] = [0.12, -0.42, 0.15]


@pytest.mark.parametrize("joint", ["shoulder_lift", "elbow_flex", "wrist_flex"])
def test_recovers_planted_play(setup, joint):
    model, data, jm, site_ids = setup
    play, enc_hyst = np.radians(0.8), np.radians(0.15)
    records = synth_pair(model, data, jm, site_ids, T_BASE_CAM,
                         BASE_Q, joint, play, enc_hyst)
    rows = hysteresis_estimates(records, model, data, jm,
                                np.zeros(6), T_BASE_CAM[:3, :3])
    assert len(rows) == 2   # both tags see lift/elbow/wrist_flex
    for r in rows:
        assert r["joint"] == joint
        assert abs(r["enc_delta_deg"] - np.degrees(enc_hyst)) < 1e-6
        # Jacobian projection linearizes the ±0.4 deg arc: recovery within a few %.
        assert abs(r["play_deg"] - np.degrees(play)) < 0.05, r


def test_bias_only_shifts_the_linearization_point(setup):
    """A calibrated encoder bias changes where the Jacobian is evaluated, not the
    planted play: recovery must hold with the production bias applied."""
    model, data, jm, site_ids = setup
    bias = np.zeros(6)
    bias[2] = np.radians(8.0)
    play = np.radians(1.0)
    # Encoder reads biased angles; the physical pose is q_enc - bias.
    records = synth_pair(model, data, jm, site_ids, T_BASE_CAM,
                         BASE_Q - bias, "elbow_flex", play, 0.0)
    for rec in records:
        rec["qpos"] = (np.array(rec["qpos"]) + bias).tolist()
    rows = hysteresis_estimates(records, model, data, jm, bias, T_BASE_CAM[:3, :3])
    for r in rows:
        assert abs(r["play_deg"] - np.degrees(play)) < 0.05, r


def test_zero_lever_tag_is_skipped(setup):
    """The wrist tag is proximal to wrist_roll (zero lever): only the finger tag
    may contribute a roll estimate, and it must still recover the plant."""
    model, data, jm, site_ids = setup
    play = np.radians(1.0)
    records = synth_pair(model, data, jm, site_ids, T_BASE_CAM,
                         BASE_Q, "wrist_roll", play, 0.0)
    rows = hysteresis_estimates(records, model, data, jm,
                                np.zeros(6), T_BASE_CAM[:3, :3])
    assert [r["tag"] for r in rows] == [0]      # finger only
    assert rows[0]["lever_m"] >= MIN_LEVER_M
    assert abs(rows[0]["play_deg"] - np.degrees(play)) < 0.1


def test_table_tag_drift_reported(setup):
    model, data, jm, site_ids = setup
    records = synth_pair(model, data, jm, site_ids, T_BASE_CAM,
                         BASE_Q, "shoulder_lift", np.radians(0.5), 0.0)
    for table_tag in TABLE_TAG_IDS:
        for rec, tvec in zip(records, ([0.1, 0.2, 0.5], [0.1, 0.2, 0.5024])):
            rec["tags"][str(table_tag)] = tvec
    rows = hysteresis_estimates(records, model, data, jm,
                                np.zeros(6), T_BASE_CAM[:3, :3])
    assert all(abs(r["table_drift_mm"] - 2.4) < 1e-6 for r in rows)


def test_plan_is_safe_and_paired(setup):
    model, data, jm, _ = setup
    steps = build_plan(model, data, jm, PROBE_JOINTS, n_poses=2,
                       approach_rad=np.radians(6.0))
    lo, hi = jm.xml_low(), jm.xml_high()
    qposadr = jm.qposadr()
    captures = [s for s in steps if s.key is not None]
    assert len(captures) % 2 == 0 and len(captures) >= 2 * 2 * len(PROBE_JOINTS) - 4
    for s in steps:
        assert np.all(s.qpos >= lo - 1e-9) and np.all(s.qpos <= hi + 1e-9)
        data.qpos[qposadr] = s.qpos
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
    # Captures come in minus/plus pairs at the *same* target pose.
    for minus, plus in zip(captures[0::2], captures[1::2]):
        assert minus.key[:2] == plus.key[:2]
        assert (minus.key[2], plus.key[2]) == ("minus", "plus")
        assert np.array_equal(minus.qpos, plus.qpos)


def test_records_roundtrip(tmp_path):
    records = [{"pose": 0, "joint": "elbow_flex", "side": "minus",
                "qpos": [0.1] * 6, "tags": {"0": [0.1, 0.2, 0.3]}}]
    path = tmp_path / "records.json"
    save_records(path, records, approach_deg=6.0)
    assert load_records(path) == records
