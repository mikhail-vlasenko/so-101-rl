"""Tests for the SO-101 reach env.

Verifies the configured waypoints are safe (EE above ee_z_min), basic
reset/step semantics, dwell-based termination, and that an unsafe waypoint
is rejected at env construction.
"""

import os

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.reach_env import SO101ReachEnv


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_reach_cfg():
    # Compose the full Hydra config so ${action_scale} (and any other top-level
    # interpolation referenced from env/reach.yaml) resolves correctly.
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["env=reach"])
    return OmegaConf.to_container(cfg.reach_env, resolve=True)


def _make_env(cfg=None):
    if cfg is None:
        cfg = _load_reach_cfg()
    return SO101ReachEnv(env_cfg=cfg,
                         xml_path=os.path.join(REPO_ROOT, "so101/scene.xml"))


def test_configured_waypoints_above_ee_z_min():
    env = _make_env()
    for i, wp in enumerate(env.waypoints):
        ee_z = env._ee_z_at(wp)
        assert ee_z >= env.ee_z_min, (
            f"waypoint {i} {wp.tolist()} has EE z={ee_z:.3f} < {env.ee_z_min}"
        )


def test_reset_obs_shape_and_onehot():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    onehot_start = 2 * env.n_joints + 3
    onehot = obs[onehot_start:onehot_start + env.n_waypoints]
    assert onehot.sum() == 1.0
    assert ((onehot == 0.0) | (onehot == 1.0)).all()
    # Prev actions occupy the tail of the obs and are zero right after reset.
    prev_n = env.n_joints * 2
    assert np.array_equal(obs[-prev_n:], np.zeros(prev_n, dtype=np.float32))


def test_step_info_keys_and_init_safety():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    # Initial pose must satisfy the safety bound.
    assert env._get_ee_pos()[2] >= env.ee_z_min
    _, _, _, _, info = env.step(np.zeros(env.n_joints, dtype=np.float32))
    for key in ("waypoint_idx", "joint_dist", "in_target", "dwell_count", "ee_z", "task_name"):
        assert key in info
    assert info["task_name"] == "reach"


def test_terminates_after_dwell_when_snapped_to_target():
    env = _make_env()
    env.reset(seed=0)
    # Snap arm exactly onto the target waypoint and zero velocity. Also snap
    # the actuator filter state (data.act) — position actuators with a
    # timeconst drive toward act, not ctrl, so leaving act at the reset pose
    # would pull the arm back to HOME on the first step.
    env.data.qpos[env.joint_qposadr] = env.target_qpos
    env.data.qvel[env.joint_dofadr] = 0.0
    env.data.ctrl[:env.n_joints] = env.target_qpos
    if env.model.na > 0:
        env.data.act[:] = env.target_qpos
    mujoco.mj_forward(env.model, env.data)
    env._prev_dist = 0.0

    terminated = False
    for step in range(env.dwell_steps + 5):
        _, _, terminated, _, info = env.step(np.zeros(env.n_joints, dtype=np.float32))
        if terminated:
            assert info["dwell_count"] >= env.dwell_steps
            assert info["reached"] is True
            break
    assert terminated, "Episode did not terminate after dwell_steps in-target"


def test_unsafe_waypoint_rejected_on_init():
    cfg = _load_reach_cfg()
    cfg["ee_z_min"] = 5.0  # taller than the robot — every waypoint is unsafe
    with pytest.raises(AssertionError):
        _make_env(cfg)


def test_out_of_range_waypoint_rejected_on_init():
    cfg = _load_reach_cfg()
    cfg["waypoints"] = [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # shoulder_pan beyond range
    with pytest.raises(AssertionError):
        _make_env(cfg)


def test_waypoint_inside_limit_zone_rejected_on_init():
    cfg = _load_reach_cfg()
    # shoulder_lift at 0.95 of its half-range → inside the limit penalty zone.
    cfg["waypoints"] = [[0.0, -1.65, 1.2, 0.0, 0.0, 0.5]]
    cfg["ee_z_min"] = 0.0  # decouple from EE check so we isolate the limit assertion
    with pytest.raises(AssertionError, match="joint-limit penalty zone"):
        _make_env(cfg)


def test_joint_limit_penalty_fires_when_arm_pushed_to_limit():
    env = _make_env()
    env.reset(seed=0)
    # Force shoulder_lift into the limit zone.
    q = env._get_joint_pos()
    q[1] = -1.7  # near lower limit of -1.745
    env.data.qpos[env.joint_qposadr] = q
    env.data.qvel[env.joint_dofadr] = 0.0
    env.data.ctrl[:env.n_joints] = q
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, info = env.step(np.zeros(env.n_joints, dtype=np.float32))
    assert info["limit_excess_sq"] > 0.0
