"""Tests for the gravity-compliance forward model (real/compliance.py): the torque
signs that justify which joints are compliant, and that the read (subtract deflection)
and write (add deflection) halves are inverses to first order.
"""
from pathlib import Path

import mujoco
import numpy as np

from real.compliance import (
    COMP_JOINTS,
    encoder_from_true,
    gravity_deflection,
    gravity_torque,
    motor_dofadr,
)
from real.twin.mapping import load_joint_maps

REPO_ROOT = Path(__file__).resolve().parent.parent
XML = REPO_ROOT / "so101" / "scene.xml"
LIFT_XML = REPO_ROOT / "so101" / "scene_lift.xml"
CAL = REPO_ROOT / "real" / "follower_calibration.json"

BENT_POSE = np.array([0.5, -0.6, 0.8, 0.3, 0.2, 0.5])   # a forward-reaching arm pose


def _setup(xml=XML):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, CAL)
    return model, data, jm


def test_motor_dofadr_excludes_play():
    """motor_dofadr returns the six actuated joints, distinct from the passive *_play
    gear-play dofs (the model carries extra dofs beyond the six)."""
    model, _, _ = _setup()
    adr = motor_dofadr(model)
    assert adr.shape == (6,)
    assert len(set(adr.tolist())) == 6
    assert adr.max() < model.nv
    assert model.nv > 6            # play joints add dofs the torque read must skip


def test_pan_has_no_gravity_torque():
    """shoulder_pan is a vertical axis, so vertical gravity exerts ~no torque about it
    for any pose (justifies pinning its compliance to 0), while lift/elbow are loaded."""
    model, data, jm = _setup()
    tau = gravity_torque(model, data, jm.qposadr(), BENT_POSE)
    assert abs(tau[0]) < 1e-6
    assert abs(tau[1]) > 1e-3 and abs(tau[2]) > 1e-3


def test_deflection_zero_on_rigid_joints():
    model, data, jm = _setup()
    compliance = np.zeros(6)
    compliance[list(COMP_JOINTS)] = [0.15, 0.07, -0.30]
    defl = gravity_deflection(model, data, jm.qposadr(), BENT_POSE, compliance)
    for j in range(6):
        if j not in COMP_JOINTS:
            assert defl[j] == 0.0
    assert np.abs(defl[list(COMP_JOINTS)]).max() > np.radians(0.1)


def test_read_write_inverse_consistency():
    """The read map (subtract the deflection at the bias-corrected pose) and its write
    inverse (encoder_from_true, fixed-point) round-trip to well under an encoder count.
    The iteration matters: the single first-order guess alone leaves visibly more."""
    model, data, jm = _setup()
    qposadr = jm.qposadr()
    compliance = np.zeros(6)
    compliance[list(COMP_JOINTS)] = [0.15, 0.07, -0.30]
    q_bc = BENT_POSE
    q_true = q_bc - gravity_deflection(model, data, qposadr, q_bc, compliance)

    q_bc_back = encoder_from_true(model, data, qposadr, q_true, compliance)
    assert np.abs(q_bc_back - q_bc).max() < np.radians(0.002)   # << one encoder count

    first_order = q_true + gravity_deflection(model, data, qposadr, q_true, compliance)
    assert np.abs(first_order - q_bc).max() > np.radians(0.02)  # iteration is needed


def test_free_body_stays_valid():
    """On a rollout scene with a free-body cube, gravity_torque must not corrupt the
    cube into a zero quaternion (it resets to qpos0, not all-zeros) — the result is
    finite and the cube dof count doesn't leak into the six motor torques."""
    model, data, jm = _setup(LIFT_XML)
    tau = gravity_torque(model, data, jm.qposadr(), BENT_POSE)
    assert tau.shape == (6,)
    assert np.all(np.isfinite(tau))
