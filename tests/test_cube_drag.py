"""Tests for the cube_drag_ratio metric.

Cube is considered "dragged" on a step when:
  cube_z < cube_half_z + DRAG_HEIGHT_TOL  AND  ||cube_xy_velocity|| > DRAG_SPEED_THRESH

The metric is the fraction of episode steps where this holds. Reported in info
at episode end as `cube_drag_ratio`.
"""

import numpy as np
import pytest

import mujoco
from hydra import compose, initialize

from pickplace_env import SO101PickPlaceEnv


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _pickplace(cfg):
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=None, obs_bias=None)


def _zero_action():
    return np.zeros(6, dtype=np.float32)


def _run_episode(env, n_steps):
    """Step until terminated/truncated or n_steps reached; return final info."""
    info = {}
    for _ in range(n_steps):
        _, _, term, trunc, info = env.step(_zero_action())
        if term or trunc:
            break
    return info


def test_cube_drag_ratio_emitted_at_episode_end(cfg):
    env = _pickplace(cfg)
    env.reset(seed=0)
    info = _run_episode(env, env.max_steps + 1)
    assert "cube_drag_ratio" in info
    assert 0.0 <= info["cube_drag_ratio"] <= 1.0


def test_static_cube_does_not_count_as_drag(cfg):
    """Cube sitting at rest, no commanded action — must not register drag."""
    env = _pickplace(cfg)
    env.reset(seed=0)
    info = _run_episode(env, env.max_steps + 1)
    # Some tiny settling motion is allowed, but drag ratio should be near zero.
    assert info["cube_drag_ratio"] < 0.05


def test_lateral_drag_registers(cfg):
    """Teleport the cube laterally on the floor each step and measure the metric.

    Setting qvel and relying on physics is unreliable because floor friction
    bleeds the cube's velocity to ~0 inside the substep loop. Setting qpos
    directly is robust: the integrator preserves the position we forced.
    """
    env = _pickplace(cfg)
    env.reset(seed=0)
    cube_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_dofadr = env.model.jnt_dofadr[cube_joint_id]

    # 5mm shift / step at step_dt≈0.02s ⇒ ≈0.25 m/s, well above DRAG_SPEED_THRESH.
    n_drag = 200
    for _ in range(n_drag):
        env.data.qpos[env.cube_qpos_idx] += 0.005
        env.data.qpos[env.cube_qpos_idx + 2] = env.cube_half_z
        env.data.qvel[cube_dofadr:cube_dofadr + 6] = 0
        _, _, term, trunc, info = env.step(_zero_action())
        if term or trunc:
            break
    env.step_count = env.max_steps
    _, _, _, _, info = env.step(_zero_action())
    assert info["cube_drag_ratio"] > 0.5


def test_high_cube_does_not_count_as_drag(cfg):
    """Cube lifted well above the floor — even with lateral motion it must not count."""
    env = _pickplace(cfg)
    env.reset(seed=0)
    cube_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_dofadr = env.model.jnt_dofadr[cube_joint_id]

    for _ in range(20):
        # Park cube 10cm up, give it lateral motion.
        env.data.qpos[env.cube_qpos_idx + 2] = 0.10
        env.data.qvel[cube_dofadr:cube_dofadr + 3] = [0.10, 0.0, 0.0]
        _, _, term, trunc, info = env.step(_zero_action())
        if term or trunc:
            break
    env.step_count = env.max_steps
    _, _, _, _, info = env.step(_zero_action())
    # Cube falls freely so it'll eventually approach floor — but for the first
    # frames of the episode it's high. We just want this to not be near 1.0.
    assert info["cube_drag_ratio"] < 0.5
