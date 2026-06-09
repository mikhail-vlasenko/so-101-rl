"""Contract tests for the shared real-arm rollout core (real.rollout_common.ArmLoop)
against a fake bus — no hardware. The loop must shape actions exactly like
training (action_to_target quantization + deadzone), gate all writes on
execute, and report sim-time velocities regardless of --slow.
"""

import mujoco
import numpy as np
import pytest

from real.rollout_common import DEFAULT_CAL, REPO_ROOT, ArmLoop
from real.twin.constants import SERVO_POSITION_DEADZONE, SERVO_POSITION_KP
from real.twin.mapping import load_joint_maps, rad_to_raw, raw_to_rad
from src.units import action_to_target
from real.twin.control import clamp_raw_delta


class FakeBus:
    """Ideal servo bus: writes land instantly, reads return the last write."""

    def __init__(self, raw0: np.ndarray):
        self.raw = np.array(raw0, dtype=np.int64)
        self.writes: list[np.ndarray] = []
        self.torque_on = False
        self.kp = None
        self.deadzone = None

    def read_all(self) -> np.ndarray:
        return self.raw.copy()

    def write_all(self, raw: np.ndarray, speed: int, accel: int) -> None:
        self.writes.append(np.array(raw, dtype=np.int64))
        self.raw = np.array(raw, dtype=np.int64)

    def set_position_kp(self, kp) -> None:
        self.kp = tuple(kp)

    def set_position_deadzone(self, dz) -> None:
        self.deadzone = tuple(dz)

    def enable_torque_all(self) -> None:
        self.torque_on = True


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(REPO_ROOT / "so101" / "scene.xml"))


@pytest.fixture(scope="module")
def jm(model):
    return load_joint_maps(model, DEFAULT_CAL)


def _mid_raw(jm):
    """Raw reading for the center of each joint's XML range."""
    direction = np.ones(len(jm.items), dtype=np.int8)
    rad_mid = 0.5 * (jm.xml_low() + jm.xml_high())
    return rad_to_raw(rad_mid, jm, direction)


def _make_loop(model, jm, bus, execute=True, slow=1.0, ema_alpha=1.0,
               prev_actions_n=1, action_scale=0.07):
    # interp_hz = control rate -> one sub-target per tick, so each test tick
    # sleeps only one wall tick.
    return ArmLoop(model=model, n_substeps=10, jm=jm, bus=bus,
                   action_scale=action_scale, prev_actions_n=prev_actions_n,
                   execute=execute, ema_alpha=ema_alpha, slow=slow,
                   interp_hz=15.0)


def test_dry_run_never_writes_or_torques(model, jm):
    bus = FakeBus(_mid_raw(jm))
    loop = _make_loop(model, jm, bus, execute=False)
    loop.boot()
    loop.tick(np.ones(6))
    assert bus.writes == []
    assert not bus.torque_on
    assert bus.kp is None and bus.deadzone is None


def test_execute_boot_sets_gains_and_torque(model, jm):
    bus = FakeBus(_mid_raw(jm))
    loop = _make_loop(model, jm, bus, execute=True)
    loop.boot()
    assert bus.torque_on
    assert bus.kp == SERVO_POSITION_KP
    assert bus.deadzone == SERVO_POSITION_DEADZONE


def test_tick_matches_training_shaping(model, jm):
    """The raw command on the bus must equal action_to_target -> rad_to_raw ->
    clamp, i.e. exactly the training-time semantics plus the safety clamp."""
    raw0 = _mid_raw(jm)
    bus = FakeBus(raw0)
    loop = _make_loop(model, jm, bus, execute=True)
    qpos0 = loop.boot()

    action = np.full(6, 0.5)
    loop.tick(action)

    expected_qpos = action_to_target(qpos0, action, loop.action_scale,
                                     jm.xml_low(), jm.xml_high())
    expected_raw = clamp_raw_delta(raw0, rad_to_raw(expected_qpos, jm, loop.direction),
                                   loop.max_raw_delta)
    assert np.array_equal(bus.writes[-1], expected_raw)
    assert np.array_equal(loop.prev_raw_target, expected_raw)


def test_below_deadzone_action_commands_no_motion(model, jm):
    """|action|=0.05 at action_scale=0.07 is ~2.3 raw units < deadzone -> the
    bus must be commanded to hold position."""
    raw0 = _mid_raw(jm)
    bus = FakeBus(raw0)
    loop = _make_loop(model, jm, bus, execute=True)
    loop.boot()
    loop.tick(np.full(6, 0.05))
    assert np.array_equal(bus.writes[-1], raw0)


def test_qvel_is_sim_time_regardless_of_slow(model, jm):
    """Under --slow the arm covers the same per-tick delta in more wall time;
    the reported qvel must stay in sim-time units (training distribution)."""
    action = np.full(6, 0.5)
    qvels = []
    for slow in (1.0, 3.0):
        bus = FakeBus(_mid_raw(jm))
        loop = _make_loop(model, jm, bus, execute=True, slow=slow)
        qpos0 = loop.boot()
        loop.tick(action)
        expected = (raw_to_rad(bus.raw, jm, loop.direction) - qpos0) / loop.control_dt
        np.testing.assert_allclose(loop.qvel, expected)
        qvels.append(loop.qvel.copy())
    np.testing.assert_allclose(qvels[0], qvels[1])


def test_prev_actions_buffer_shifts(model, jm):
    bus = FakeBus(_mid_raw(jm))
    loop = _make_loop(model, jm, bus, execute=False, prev_actions_n=2)
    loop.boot()
    a1 = np.full(6, 0.3, dtype=np.float64)
    a2 = np.full(6, -0.7, dtype=np.float64)
    loop.tick(a1)
    loop.tick(a2)
    np.testing.assert_allclose(loop.prev_actions[0], a1.astype(np.float32))
    np.testing.assert_allclose(loop.prev_actions[1], a2.astype(np.float32))


def test_ema_smoothing_applies_to_buffer_and_command(model, jm):
    bus = FakeBus(_mid_raw(jm))
    loop = _make_loop(model, jm, bus, execute=False, ema_alpha=0.5)
    loop.boot()
    loop.tick(np.ones(6))
    smoothed = loop.tick(np.zeros(6))
    np.testing.assert_allclose(smoothed, np.full(6, 0.5))
    np.testing.assert_allclose(loop.prev_actions[-1], np.full(6, 0.5, dtype=np.float32))


def test_boot_aborts_on_pose_outside_calibration(model, jm):
    """A boot pose past the XML joint range (bad calibration) must abort
    before any torque is enabled."""
    frac = -0.2 / (jm.xml_high() - jm.xml_low())  # 0.2 rad below xml_low
    raw_bad = np.round(jm.cal_min() + frac * (jm.cal_max() - jm.cal_min())).astype(np.int64)
    bus = FakeBus(raw_bad)
    loop = _make_loop(model, jm, bus, execute=True)
    with pytest.raises(SystemExit):
        loop.boot()
    assert not bus.torque_on
