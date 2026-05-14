"""Tests for the shared servo-quantization + stiction-deadzone helper that
bridges sim→real by zeroing sub-deadzone commands and rounding to raw units.
"""

import numpy as np
import pytest

from base_env import SERVO_DEADZONE_RAW, SERVO_RAW_UNIT_RAD, action_to_target


JOINT_LOW = np.full(6, -2.0)
JOINT_HIGH = np.full(6, 2.0)


def test_below_deadzone_is_zeroed():
    """action_scale=0.035, |action|=0.05 → ~1.14 raw units < deadzone (4) → no motion."""
    current = np.zeros(6)
    action = np.full(6, 0.05)
    target = action_to_target(current, action, 0.035, JOINT_LOW, JOINT_HIGH)
    assert np.allclose(target, current)


def test_at_deadzone_boundary_is_zeroed():
    """Just below the deadzone is still zero; ensures the boundary is exclusive."""
    current = np.zeros(6)
    raw_just_under = SERVO_DEADZONE_RAW - 0.01
    action_scale = 0.05
    action = np.full(6, raw_just_under * SERVO_RAW_UNIT_RAD / action_scale)
    target = action_to_target(current, action, action_scale, JOINT_LOW, JOINT_HIGH)
    assert np.allclose(target, current)


def test_above_deadzone_is_quantized():
    """action=0.2, scale=0.035 → 4.57 raw units → rounds to 5 raw units of motion."""
    current = np.zeros(6)
    action = np.full(6, 0.2)
    target = action_to_target(current, action, 0.035, JOINT_LOW, JOINT_HIGH)
    expected = np.full(6, 5 * SERVO_RAW_UNIT_RAD)
    assert np.allclose(target, expected)


def test_negative_action_quantizes_symmetrically():
    """Negative actions round the same way (toward zero magnitude past 0.5)."""
    current = np.zeros(6)
    action = np.full(6, -0.2)
    target = action_to_target(current, action, 0.035, JOINT_LOW, JOINT_HIGH)
    expected = np.full(6, -5 * SERVO_RAW_UNIT_RAD)
    assert np.allclose(target, expected)


def test_clips_to_joint_range():
    """Target is clipped to [joint_low, joint_high] after quantization."""
    current = np.full(6, 1.99)
    action = np.ones(6)
    target = action_to_target(current, action, 0.05, JOINT_LOW, JOINT_HIGH)
    assert np.all(target <= JOINT_HIGH)
    assert np.all(target >= JOINT_LOW)


def test_zero_action_is_noop():
    current = np.array([0.1, -0.3, 0.0, 0.5, -1.2, 0.8])
    target = action_to_target(current, np.zeros(6), 0.035, JOINT_LOW, JOINT_HIGH)
    assert np.allclose(target, current)


def test_per_joint_independence():
    """Some joints below deadzone (no motion), others above (quantized motion)."""
    current = np.zeros(6)
    action = np.array([0.05, 0.2, -0.05, -0.2, 0.0, 1.0])
    target = action_to_target(current, action, 0.035, JOINT_LOW, JOINT_HIGH)
    # 0.05 → ~1.14 raw → 0
    # 0.2  → ~4.57 raw → 5
    # 1.0  → ~22.83 raw → 23
    expected = np.array([0, 5, 0, -5, 0, 23], dtype=float) * SERVO_RAW_UNIT_RAD
    assert np.allclose(target, expected)


@pytest.mark.parametrize("action_scale", [0.02, 0.035, 0.05, 0.1])
def test_max_action_is_above_deadzone(action_scale):
    """|action|=1.0 must always overcome the deadzone, otherwise the policy
    can never move the arm regardless of what it commands."""
    raw_at_max = action_scale / SERVO_RAW_UNIT_RAD
    assert raw_at_max > SERVO_DEADZONE_RAW, (
        f"action_scale={action_scale}: max command is {raw_at_max:.2f} raw units, "
        f"which is below deadzone={SERVO_DEADZONE_RAW}. Increase action_scale."
    )
