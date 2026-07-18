"""Encoder zero-offset + gravity-compliance calibration IO (`calibration.yaml`).

`qpos_bias[6]` is the per-joint encoder offset in rad and `compliance[6]` the
per-joint load-dependent deflection (rad per N.m, zero on the rigid joints). The true
joint angle is

    theta_true = theta_encoder - qpos_bias - compliance * tau_grav(theta_encoder - qpos_bias)

(see `real/calib/compliance.py`). The deploy-time wrapper applies both before the policy sees
the observation, so the arm's forward kinematics (and therefore the FK end-effector and
marker channels) line up with what the camera measures. See `calibration_plan.md` Step 1.

Solved together by `real/calib/calibrate_qpos.py`. The cube vision offset (`cube_pos_bias`,
Step 2) is a separate pass and is not written here yet.
"""
import os

import numpy as np
import yaml

from real.twin.mapping import JOINT_NAMES

CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.yaml")


def save_calibration(path, qpos_bias, compliance, n_samples, rms_before_mm, rms_after_mm):
    """Write qpos_bias and compliance (rad, 6 each) plus the fit's quality metadata."""
    qpos_bias = np.asarray(qpos_bias, dtype=np.float64)
    compliance = np.asarray(compliance, dtype=np.float64)
    assert qpos_bias.shape == (6,), f"qpos_bias must be (6,), got {qpos_bias.shape}"
    assert compliance.shape == (6,), f"compliance must be (6,), got {compliance.shape}"
    data = {
        "qpos_bias": qpos_bias.tolist(),
        "qpos_bias_deg": np.degrees(qpos_bias).round(3).tolist(),
        "compliance": compliance.tolist(),
        "compliance_deg_per_Nm": np.degrees(compliance).round(3).tolist(),
        "joint_names": JOINT_NAMES,
        "n_samples": int(n_samples),
        "rms_before_mm": float(rms_before_mm),
        "rms_after_mm": float(rms_after_mm),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def load_calibration(path=CALIBRATION_PATH):
    """Load qpos_bias (rad, np.ndarray (6,)). Raises if the file lacks the key."""
    with open(path) as f:
        data = yaml.safe_load(f)
    qpos_bias = np.array(data["qpos_bias"], dtype=np.float64)
    assert qpos_bias.shape == (6,), f"qpos_bias must be (6,), got {qpos_bias.shape}"
    return qpos_bias


def load_compliance(path=CALIBRATION_PATH):
    """Load compliance (rad per N.m, np.ndarray (6,)). Raises if the file lacks the
    key -- a calibration.yaml from before the compliance solve must be re-solved."""
    with open(path) as f:
        data = yaml.safe_load(f)
    compliance = np.array(data["compliance"], dtype=np.float64)
    assert compliance.shape == (6,), f"compliance must be (6,), got {compliance.shape}"
    return compliance
