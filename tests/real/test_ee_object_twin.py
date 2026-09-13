"""Actor EE-object geometry must use the real rollout's observed-joint FK."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize

from real.rollout.rollout_lift import build_state_frame
from src.base_env import EE_OBJECT_DELTA_DIM
from src.lift_env import SO101LiftEnv
from src.train import runtime_cfg_from_hydra


@pytest.mark.parametrize("error_source", ["bias", "noise", "passive_play"])
@pytest.mark.parametrize("marker_include_rot", [False, True])
def test_actor_and_teacher_use_observed_fk_without_mutating_physics(
        error_source, marker_include_rot):
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    runtime = runtime_cfg_from_hydra(cfg)
    noise = dict(runtime.obs_noise)
    if error_source != "noise":
        noise["qpos_sigma"] = 0.0
    runtime = replace(runtime, obs_noise=noise,
                      marker_include_rot=marker_include_rot)
    env = SO101LiftEnv(env_cfg=cfg.lift_env, cfg=runtime)
    try:
        env.reset(seed=7)
        env._qpos_bias[:] = 0.0
        if error_source == "bias":
            env._qpos_bias[0] = 0.02
        elif error_source == "passive_play":
            joint = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift_play")
            assert joint >= 0
            env.data.qpos[env.model.jnt_qposadr[joint]] = env.model.jnt_range[joint, 1]
        mujoco.mj_forward(env.model, env.data)
        true_ee = env._get_ee_pos()
        physical_qpos = env.data.qpos.copy()
        physical_qvel = env.data.qvel.copy()
        physical_ctrl = env.data.ctrl.copy()
        physical_sites = env.data.site_xpos.copy()
        physical_time = env.data.time

        # Independent real-camera FK reconstruction: only measured motor joints
        # are written, exactly as the camera rollout's scene is updated.
        real_data = mujoco.MjData(env.model)
        for _ in range(3):
            obs = env._serve_obs(reset=True)
            qpos, qvel = env._last_encoder_obs
            real_data.qpos[env.joint_qposadr] = qpos
            mujoco.mj_forward(env.model, real_data)
            ee = real_data.site_xpos[env.ee_site_id].copy()
            assert np.linalg.norm(ee - true_ee) > 1e-5
            live, age = env._obj.serve(env.data.time)
            _, marker_age = env._tag_obs(
                env._held_marker_pos, env._held_marker_rot, env._marker_last_capture_t)
            real_frame = build_state_frame(
                qpos, qvel, env._held_marker_pos, env._held_marker_rot,
                marker_age, live, age, env._prev_actions, ee, marker_include_rot)
            np.testing.assert_array_equal(obs[:env.state_dim], real_frame)

            teacher = env.privileged_obs()
            teacher_live, _ = env._obj_state_priv.serve(env.data.time)
            delta = slice(env.state_dim - EE_OBJECT_DELTA_DIM, env.state_dim)
            np.testing.assert_allclose(teacher[delta], ee - teacher_live, atol=1e-7)
            np.testing.assert_array_equal(teacher[:12], obs[:12])
            np.testing.assert_array_equal(
                obs[-env.priv_dim:], env._priv_tail().astype(np.float32))
            np.testing.assert_array_equal(env.data.qpos, physical_qpos)
            np.testing.assert_array_equal(env.data.qvel, physical_qvel)
            np.testing.assert_array_equal(env.data.ctrl, physical_ctrl)
            np.testing.assert_array_equal(env.data.site_xpos, physical_sites)
            assert env.data.time == physical_time
    finally:
        env.close()
