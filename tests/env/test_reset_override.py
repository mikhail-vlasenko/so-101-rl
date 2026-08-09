"""reset(options=...) state override: sysid/replay_rollout.py relies on this to
restart a sim episode from a recorded real-arm pose."""

import numpy as np
import pytest

from src.base_env import RuntimeEnvConfig, sample_cube_orientation
from src.lift_env import SO101LiftEnv
from src.train import make_env
from real.rollout.rollout_common import load_env_cfg

QPOS = np.array([0.3, -1.5, 1.57, -0.4, 1.45, -0.01])


@pytest.fixture(scope="module")
def env():
    env_cfg, prev_actions_n, marker_include_rot, history_taps = load_env_cfg("lift")
    return make_env(SO101LiftEnv, env_cfg, SO101LiftEnv.XML_PATH,
                    cfg=RuntimeEnvConfig(marker_include_rot=marker_include_rot,
                                         prev_actions_n=prev_actions_n,
                                         history_taps=history_taps))


def test_make_env_can_map_runtime_cfg():
    env_cfg, _, _, _ = load_env_cfg("lift")
    runtime_cfg = {
        "obs_noise": {"qpos_sigma": 0.0, "marker_rot_sigma": 0.0,
                      "tag_px_noise": 0.0, "tag_depth_factor": 2.0,
                      "live_sigma": 0.0, "precise_sigma": 0.0},
        "cam_latency": None,
        "obs_bias": {"qpos_sigma": 0.0, "marker_pos_sigma": 0.0,
                     "marker_rot_sigma": 0.0, "live_sigma": 0.0, "precise_sigma": 0.0,
                     "marker_common_sigma": 0.0},
        "marker_dropout": {"near": 0.0, "far": 0.0},
        "marker_always_visible": True,
        "marker_include_rot": False,
        "prev_actions_n": 1,
        "cube_size_jitter": 0.0,
        "history_taps": (0,),
    }
    env = make_env(SO101LiftEnv, env_cfg, SO101LiftEnv.XML_PATH,
                   cfg=RuntimeEnvConfig(**runtime_cfg))
    assert env.marker_always_visible is True
    assert env.prev_actions_n == 1
    env.close()


def test_reset_pins_qpos_and_cube(env):
    rng = np.random.default_rng(0)
    cube_quat, rest_half_z = sample_cube_orientation(rng, env.cube_half_extents)
    cube_pos = np.array([0.2, -0.05, rest_half_z])
    env.reset(seed=0, options={"qpos": QPOS, "cube_pos": cube_pos,
                               "cube_quat": cube_quat})
    np.testing.assert_allclose(env._get_joint_pos(), QPOS, atol=1e-12)
    np.testing.assert_allclose(env._get_cube_pos(), cube_pos, atol=1e-12)
    assert env.cube_rest_half_z == rest_half_z
    # actuator/profile state must start at the pinned pose, not snap from 0
    np.testing.assert_allclose(env.data.ctrl[:6], QPOS, atol=1e-12)
    # holding still keeps the arm at the pinned pose (no reset transient)
    env.step(np.zeros(6))
    np.testing.assert_allclose(env._get_joint_pos(), QPOS, atol=0.02)


def test_partial_override_keeps_sampling(env):
    obs1, _ = env.reset(seed=3, options={"qpos": QPOS})
    np.testing.assert_allclose(env._get_joint_pos(), QPOS, atol=1e-12)
    cube1 = env._get_cube_pos()
    env.reset(seed=4, options={"qpos": QPOS})
    assert not np.allclose(env._get_cube_pos(), cube1)  # cube still sampled


def test_cube_pos_requires_quat(env):
    with pytest.raises(AssertionError):
        env.reset(seed=0, options={"cube_pos": np.array([0.2, 0.0, 0.01])})
