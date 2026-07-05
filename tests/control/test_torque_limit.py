"""Sim/real torque-cap consistency: the so101.xml actuator forcerange and the
real-side Torque_Limit register value must both derive from
SERVO_TORQUE_LIMIT_FRAC, or the sim plant and the servo cap drift apart."""

import mujoco
import numpy as np

from real.rollout_common import REPO_ROOT
from real.twin.constants import (
    SERVO_TORQUE_LIMIT,
    SERVO_TORQUE_LIMIT_FRAC,
    STS3215_STALL_TORQUE_NM,
)
from src.base_env import JOINT_NAMES


def test_xml_forcerange_matches_torque_cap():
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "so101" / "scene.xml"))
    expected = SERVO_TORQUE_LIMIT_FRAC * STS3215_STALL_TORQUE_NM
    for name in JOINT_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"actuator '{name}' not found"
        lo, hi = model.actuator_forcerange[aid]
        np.testing.assert_allclose(hi, expected, atol=5e-3,
                                   err_msg=f"{name} forcerange upper")
        np.testing.assert_allclose(lo, -expected, atol=5e-3,
                                   err_msg=f"{name} forcerange lower")


def test_register_value_matches_fraction():
    # register is 0-1000 = fraction of stall
    assert SERVO_TORQUE_LIMIT == round(1000 * SERVO_TORQUE_LIMIT_FRAC)
    assert 0 < SERVO_TORQUE_LIMIT <= 1000
