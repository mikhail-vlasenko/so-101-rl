"""Contract for SO101BaseEnv.privileged_obs — the distillation teacher view.

privileged_obs() must return exactly what a marker_always_visible=true policy
would see on the current state: every tag pose fresh, while sharing the paired
_compute_obs's encoder read, task extras, and prev-actions so the two views
differ only in the tag channels. The distillation rig regresses a camera-obs
student onto a teacher queried on this view (src/distill.py), so a bug here —
stale tags leaking in, or a desynced encoder read — silently corrupts every
label.

Obs layout (lift/pickplace, prev_actions_n=1, marker_include_rot=False):
actor block [qpos(6), qvel(6), markers(2*3), marker_age(2), live(3),
 live_age(1), center(3), sqrtM(6), precise_age(1), extra(4), prev_actions(6)]
= 44 dims, followed by the privileged tail (priv_dim_for, critic-only ground
truth) — identical in both views, since it is GT either way.
"""

import numpy as np
import pytest
from hydra import compose, initialize
from stable_baselines3.common.vec_env import DummyVecEnv

from src.base_env import RuntimeEnvConfig, priv_dim_for
from src.lift_env import SO101LiftEnv
from src.pickplace_env import SO101PickPlaceEnv

ENCODER = slice(0, 12)          # qpos + qvel — always shared between the views
TAGS = slice(12, 34)            # markers, marker_age, live(+age), center/sqrtM(+age)
LIVE_POS = slice(20, 23)
LIVE_AGE = 23
CENTER = slice(24, 27)
PRECISE_AGE = 33
EXTRA = slice(34, 38)
PREV_ACTIONS = slice(38, 44)
MARKER_AGE_CAP_S = 1.0


@pytest.fixture(scope="module")
def lift_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=lift"])


@pytest.fixture(scope="module")
def pickplace_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


# Filled by the module-scoped fixture below; the env config is a plain
# dict-like Hydra node consumed by the env constructors.
_LIFT_ENV_CFG = None
_PICKPLACE_ENV_CFG = None


@pytest.fixture(autouse=True, scope="module")
def _load_env_cfgs(lift_cfg, pickplace_cfg):
    global _LIFT_ENV_CFG, _PICKPLACE_ENV_CFG
    _LIFT_ENV_CFG = lift_cfg.lift_env
    _PICKPLACE_ENV_CFG = pickplace_cfg.pickplace_env


def _step_zero(env, n):
    """Reset then take n zero-action steps; return (last student obs, env)."""
    obs, _ = env.reset(seed=0)
    for _ in range(n):
        obs, _, term, trunc, _ = env.step(np.zeros(6, dtype=np.float32))
        if term or trunc:
            obs, _ = env.reset(seed=1)
    return obs


def test_privileged_equals_student_when_always_visible():
    """marker_always_visible=true feeds every tag fresh to the student too, so
    the held state and the privileged mirror coincide exactly — the two views
    are identical every tick."""
    runtime = RuntimeEnvConfig(marker_always_visible=True, prev_actions_n=1)
    env = SO101LiftEnv(env_cfg=_LIFT_ENV_CFG, xml_path="so101/scene_lift.xml",
                       cfg=runtime)
    obs, _ = env.reset(seed=0)
    assert np.array_equal(obs, env.privileged_obs())
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert np.array_equal(obs, env.privileged_obs()), "views diverged"
        if term or trunc:
            break


def test_privileged_fresh_while_student_stale_under_full_dropout():
    """With every frame dropped, the student never updates a tag: its cube pose
    stays at the reset zero with age pinned at the cap, while the privileged
    mirror ingests every frame and holds the true, freshly-aged pose."""
    runtime = RuntimeEnvConfig(
        obs_noise=None, obs_bias=None, cam_latency=None,
        marker_dropout={"near": 1.0, "far": 1.0}, marker_always_visible=False,
        prev_actions_n=1)
    env = SO101LiftEnv(env_cfg=_LIFT_ENV_CFG, xml_path="so101/scene_lift.xml",
                       cfg=runtime)
    student = _step_zero(env, 5)
    priv = env.privileged_obs()

    # Student: live/precise never detected -> held values zero, ages at the cap.
    assert np.allclose(student[LIVE_POS], 0.0)
    assert np.allclose(student[CENTER], 0.0)
    assert student[LIVE_AGE] == pytest.approx(MARKER_AGE_CAP_S)
    assert student[PRECISE_AGE] == pytest.approx(MARKER_AGE_CAP_S)
    # Privileged: real (noise-free here) cube center, small fresh ages.
    assert np.linalg.norm(priv[LIVE_POS]) > 0.01
    assert priv[LIVE_AGE] < MARKER_AGE_CAP_S
    assert priv[PRECISE_AGE] < MARKER_AGE_CAP_S


def test_privileged_shares_encoder_and_prev_actions():
    """The two views must differ ONLY in the tag channels: the encoder read
    (qpos/qvel) and prev-actions are byte-identical, proving privileged_obs
    reuses the paired _compute_obs's cached encoder rather than re-drawing it."""
    runtime = RuntimeEnvConfig(
        obs_noise=None, obs_bias=None, cam_latency=None,
        marker_dropout={"near": 1.0, "far": 1.0}, marker_always_visible=False,
        prev_actions_n=1)
    env = SO101LiftEnv(env_cfg=_LIFT_ENV_CFG, xml_path="so101/scene_lift.xml",
                       cfg=runtime)
    student = _step_zero(env, 5)
    priv = env.privileged_obs()
    assert np.array_equal(student[ENCODER], priv[ENCODER])
    assert np.array_equal(student[PREV_ACTIONS], priv[PREV_ACTIONS])
    # The tag region genuinely diverged (student stale, privileged fresh).
    assert not np.allclose(student[TAGS], priv[TAGS])


def test_privileged_extra_uses_privileged_cube_on_pickplace():
    """pickplace's extra channels carry cube_to_target, derived from the held
    cube pose. The privileged view must derive it from the fresh privileged cube
    (via _obs_extra), so under dropout the extra channels differ from the
    student's stale-cube version."""
    runtime = RuntimeEnvConfig(
        obs_noise=None, obs_bias=None, cam_latency=None,
        marker_dropout={"near": 1.0, "far": 1.0}, marker_always_visible=False,
        prev_actions_n=1)
    env = SO101PickPlaceEnv(env_cfg=_PICKPLACE_ENV_CFG,
                            xml_path="so101/scene_pickplace.xml", cfg=runtime)
    student = _step_zero(env, 5)
    priv = env.privileged_obs()
    # cube_to_target xy (first two extra dims) must differ: student sees a zero
    # cube, privileged sees the real one.
    assert not np.allclose(student[EXTRA][:2], priv[EXTRA][:2])


def test_privileged_obs_requires_prior_compute_obs():
    """privileged_obs before any _compute_obs (stale/None cached encoder) must
    fail loud, not silently serve a wrong encoder read."""
    runtime = RuntimeEnvConfig(marker_always_visible=True, prev_actions_n=1)
    env = SO101LiftEnv(env_cfg=_LIFT_ENV_CFG, xml_path="so101/scene_lift.xml",
                       cfg=runtime)
    env.reset(seed=0)
    env._last_encoder_obs = None
    with pytest.raises(AssertionError):
        env.privileged_obs()


def test_env_method_privileged_obs_through_vec_env():
    """The distillation collector pulls the teacher view across the SubprocVecEnv
    boundary via env_method('privileged_obs'); lock in that the method resolves
    through the Monitor/vec wrapper and returns the right shape."""
    runtime = RuntimeEnvConfig(marker_always_visible=True, prev_actions_n=1)
    venv = DummyVecEnv([lambda: SO101LiftEnv(
        env_cfg=_LIFT_ENV_CFG, xml_path="so101/scene_lift.xml", cfg=runtime)])
    venv.reset()
    priv = np.stack(venv.env_method("privileged_obs"))
    assert priv.shape == (1, 44 + priv_dim_for(False))
    venv.close()
