"""Real-rollout arm-loop behavior that does not require arm hardware."""

from pathlib import Path

import mujoco
import numpy as np

from real.rollout.rollout_common import ArmLoop
from real.twin.constants import FOLDED_REST_QPOS
from real.twin.mapping import JOINT_NAMES
from src.units import max_raw_delta_per_step


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class FakeBus:
    def __init__(self, raw):
        self.raw = np.asarray(raw, dtype=np.int64)
        self.writes = []

    def write_all(self, raw, speed, accel):
        self.raw = raw.copy()
        self.writes.append(raw.copy())

    def read_all(self):
        return self.raw.copy()


def _rest_loop(execute=True):
    loop = ArmLoop.__new__(ArmLoop)
    loop.execute = execute
    loop.n_joints = 2
    loop.xml_low = np.array([-1.0, -1.0])
    loop.xml_high = np.array([1.0, 1.0])
    loop.control_dt = 0.1
    loop.n_interp = 1
    loop.sub_dt = 0.0
    loop.qpos = np.array([0.5, -0.5])
    loop.qvel = np.ones(2)
    loop.prev_raw_target = np.array([50, -50], dtype=np.int64)
    loop.bus = FakeBus(loop.prev_raw_target)
    loop._true_to_encoder_raw = lambda qpos: np.rint(qpos * 100).astype(np.int64)
    loop._encoder_to_true = lambda raw: raw.astype(np.float64) / 100.0
    return loop


def test_return_to_rest_reaches_target_with_reduced_raw_clamp():
    loop = _rest_loop()
    initial = loop.prev_raw_target.copy()
    rest = np.array([-0.2, 0.3])
    action_scale = 0.01

    loop.return_to_rest(rest, duration_s=0.2,
                        rest_action_scale=action_scale, settle_s=0.1,
                        should_stop=lambda: False)

    writes = np.vstack([initial, *loop.bus.writes])
    assert np.max(np.abs(np.diff(writes, axis=0))) \
        <= max_raw_delta_per_step(action_scale)
    np.testing.assert_array_equal(loop.prev_raw_target, np.array([-20, 30]))
    np.testing.assert_allclose(loop.qpos, rest)
    np.testing.assert_array_equal(loop.qvel, np.zeros(2))


def test_return_to_rest_is_noop_in_dry_run():
    loop = _rest_loop(execute=False)

    loop.return_to_rest(np.zeros(2), duration_s=1.0,
                        rest_action_scale=0.01, settle_s=0.0,
                        should_stop=lambda: False)

    assert loop.bus.writes == []


def test_return_to_rest_honors_stop_before_moving():
    loop = _rest_loop()

    completed = loop.return_to_rest(
        np.zeros(2), duration_s=1.0, rest_action_scale=0.01,
        settle_s=0.0, should_stop=lambda: True)

    assert not completed
    assert loop.bus.writes == []


def test_folded_rest_pose_is_in_range_and_off_the_floor():
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "so101" / "scene.xml"))
    data = mujoco.MjData(model)
    joint_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    qpos = np.asarray(FOLDED_REST_QPOS)
    low, high = model.jnt_range[joint_ids].T
    assert np.all(qpos >= low)
    assert np.all(qpos <= high)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    data.qpos[model.jnt_qposadr[joint_ids]] = qpos
    mujoco.mj_forward(model, data)

    for contact in data.contact[:data.ncon]:
        assert floor_id not in (contact.geom1, contact.geom2)
