"""Linear gravity-compliance model for the SO-101 encoder -> true-angle mapping.

Under gravity load the shoulder-lift, elbow and wrist-flex links flex elastically
(confirmed elastic rather than gear backlash by sysid/probe_backlash.py: the link
tracks the motor to <=0.2 deg with no approach-direction hysteresis), so the true link
angle lags the encoder by an amount proportional to the gravity torque on that joint:

    q_true = q_enc - qpos_bias - compliance * tau_grav(q_enc - qpos_bias)

`compliance` (rad per N.m, 6-vector, zero on the rigid joints) is solved jointly with
the encoder bias in real/calib/calibrate_qpos.py and stored in calibration.yaml. The
calibration solve and the real-arm deployment (real/rollout/rollout_common.py) both correct
poses through this one module, so the sim FK the bias was fit against and the obs the
policy sees at deployment share the exact same forward model. Cross-validated on the
qpos-calib sweep it roughly halves the held-out marker error (3.6 -> 2.0 mm mean,
8.2 -> 4.7 mm max).
"""
import mujoco
import numpy as np

from real.twin.mapping import JOINT_NAMES

# The joints whose links measurably deflect under gravity load: shoulder_lift,
# elbow_flex, wrist_flex. shoulder_pan (vertical axis: no gravity torque), wrist_roll
# (roll axis: ~no torque) and gripper stay rigid, so their compliance is pinned to 0.
COMP_JOINTS = (1, 2, 3)


def motor_dofadr(model):
    """DOF addresses of the six actuated arm joints (excludes the passive *_play
    gear-play joints), in JOINT_NAMES order."""
    return np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in JOINT_NAMES])


def gravity_torque(model, data, qposadr, q):
    """Gravity generalized force (N.m) at the six actuated joints for motor pose `q`,
    with qvel=0 and the gear-play joints at slack 0 -- the load that deflects the
    compliant joints. Resets to model.qpos0 first so any free body (e.g. the cube in a
    rollout scene) keeps a valid pose and stays out of the arm's qfrc_bias."""
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[qposadr] = q
    mujoco.mj_forward(model, data)
    return data.qfrc_bias[motor_dofadr(model)].copy()


def gravity_deflection(model, data, qposadr, q, compliance):
    """Elastic deflection (rad, 6-vector) = compliance * gravity_torque(q), zero on the
    rigid joints. The read path subtracts it from the bias-corrected encoder pose to get
    the true link pose:  q_true = q_bc - gravity_deflection(q_bc). This single evaluation
    is the forward model the calibration solve fits (real/calibrate_qpos._true_poses)."""
    return compliance * gravity_torque(model, data, qposadr, q)


# Fixed-point iterations to invert the read map for the write path. The deflection is a
# small contraction (|compliance * dtau/dq| << 1), so a few sweeps converge to well under
# one encoder count; the first-order guess alone leaves up to ~0.3 deg on the loaded lift.
_INVERT_ITERS = 3


def encoder_from_true(model, data, qposadr, q_true, compliance):
    """Inverse of the read map for the write path: the bias-corrected encoder pose whose
    gravity deflection lands the link at `q_true` (i.e. solve x - gravity_deflection(x) =
    q_true for x). Fixed-point iteration x <- q_true + gravity_deflection(x), so a hold
    target round-trips to the present encoder exactly rather than only to first order and
    the commanded arm does not creep. Needs the arm's gravity torque, hence model/data."""
    q_bc = q_true.copy()
    for _ in range(_INVERT_ITERS):
        q_bc = q_true + gravity_deflection(model, data, qposadr, q_bc, compliance)
    return q_bc
