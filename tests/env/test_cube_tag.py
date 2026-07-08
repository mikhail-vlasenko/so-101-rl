"""Cube-tag observation contract: the sponge is tracked via the AprilTag on its
largest face. The obs carries the raw tag pose (world xyz + axis-angle rotation
vector — never an inferred sponge center) plus an age channel, under the same
hold-last-pose convention as the arm markers. Detection is gated by the marker
angle/height geometry, the DR dropout rolls, and — cube tag only — an mj_ray
occlusion test at capture time."""

import mujoco
import numpy as np
import pytest

from src.base_env import (
    MARKER_AGE_CAP_S,
    RuntimeEnvConfig,
    cube_tag_occluded,
    cube_tag_visible,
    marker_world_poses,
)
from src.lift_env import SO101LiftEnv


def _cfg():
    return {
        "action_scale": 0.07,
        "use_servo_profile": True,
        "max_steps": 300,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "cube_smallest_face_only": False,
        "floor_contact_penalty": 0.0,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": 0.0,
        "floor_force_coeff": 0.0,
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "target_height": 0.10,
    }


# Default layout (marker_include_rot=False, prev_actions_n=2):
# [qpos(6), qvel(6), markers(6), marker_age(2), cube_tag_pos(3),
#  cube_tag_rot(3), cube_age(1), extra(4), prev_actions(12)]
CUBE_POS = slice(20, 23)
CUBE_ROT = slice(23, 26)
CUBE_AGE = 26


@pytest.fixture(scope="module")
def env():
    return SO101LiftEnv(env_cfg=_cfg(), cfg=RuntimeEnvConfig())


def _refresh_detection(env):
    """Re-roll a camera detection of the env's current (manually posed) state
    and ingest it — what the synchronous camera does at each obs instant."""
    frame = env._process_frame(env._capture_camera_state())
    env._ingest_frame(env.data.time, frame)
    return frame


def _cube_tag_pose(env):
    (pos,), (rot,) = marker_world_poses(env.data, [env.cube_tag_site_id])
    return pos, rot


def test_spawn_detects_on_first_frame(env):
    """The tag-visible spawn guarantees the reset frame detects: the obs
    carries the live tag pose with zero age (synchronous camera)."""
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert env._cam_frame.cube_detected
        pos, rot = _cube_tag_pose(env)
        np.testing.assert_allclose(obs[CUBE_POS], pos, atol=1e-6)
        np.testing.assert_allclose(obs[CUBE_ROT], rot, atol=1e-6)
        assert obs[CUBE_AGE] < 1e-4


def test_rot_obs_is_rotation_vector(env):
    """Reconstructing the quaternion from the rot obs must reproduce the tag
    site's orientation (same axis-angle convention as the arm markers)."""
    obs, _ = env.reset(seed=0)
    quat_expected = np.empty(4)
    mujoco.mju_mat2Quat(quat_expected, env.data.site_xmat[env.cube_tag_site_id])
    rot = obs[CUBE_ROT].astype(np.float64)
    angle = np.linalg.norm(rot)
    quat_rebuilt = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat_rebuilt, rot / angle, angle)
    assert (np.allclose(quat_rebuilt, quat_expected, atol=1e-5)
            or np.allclose(quat_rebuilt, -quat_expected, atol=1e-5))


def test_turned_away_tag_freezes_and_ages(env):
    """Yaw the cube 180° so the tag faces away from the camera: the obs must
    hold the last detected pose while the age channel grows with sim time."""
    obs0, _ = env.reset(seed=0)
    held_pos = obs0[CUBE_POS].copy()
    held_rot = obs0[CUBE_ROT].copy()

    quat = env.data.qpos[env.cube_qpos_idx + 3:env.cube_qpos_idx + 7].copy()
    yaw180 = np.array([0.0, 0.0, 0.0, 1.0])  # 180° about world z
    flipped = np.empty(4)
    mujoco.mju_mulQuat(flipped, yaw180, quat)
    env.data.qpos[env.cube_qpos_idx + 3:env.cube_qpos_idx + 7] = flipped
    mujoco.mj_forward(env.model, env.data)
    assert not cube_tag_visible(env.model, env.data, env.cube_tag_site_id,
                                env.tag_cam, env.cube_body_id)

    prev_age = 0.0
    for _ in range(5):
        obs, *_ = env.step(np.zeros(6, dtype=np.float32))
        assert not env._cam_frame.cube_detected
        np.testing.assert_array_equal(obs[CUBE_POS], held_pos)
        np.testing.assert_array_equal(obs[CUBE_ROT], held_rot)
        assert obs[CUBE_AGE] > prev_age
        prev_age = obs[CUBE_AGE]


def test_never_seen_reads_zero_with_capped_age():
    """Full detector dropout starves the cube tag from reset on: zero pose,
    age pinned at MARKER_AGE_CAP_S (the tag-visible spawn only guarantees
    geometry, not the dropout roll)."""
    env = SO101LiftEnv(env_cfg=_cfg(),
                       cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0}))
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(obs[CUBE_POS], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(obs[CUBE_ROT], np.zeros(3, dtype=np.float32))
        assert obs[CUBE_AGE] == np.float32(MARKER_AGE_CAP_S)


def test_always_visible_bypasses_dropout_and_occlusion():
    """marker_always_visible feeds the cube tag fresh regardless of dropout —
    the easy-mode crutch covers the cube exactly like the arm markers."""
    env = SO101LiftEnv(
        env_cfg=_cfg(),
        cfg=RuntimeEnvConfig(marker_dropout={"near": 1.0, "far": 1.0},
                             marker_always_visible=True),
    )
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(6, dtype=np.float32))
        assert env._cam_frame.cube_detected
        pos, _ = _cube_tag_pose(env)
        np.testing.assert_allclose(obs[CUBE_POS], pos, atol=1e-6)
        assert obs[CUBE_AGE] < 1e-4


# Minimal scene for the occlusion unit test: a tag-carrying cube, a wall that
# can block the ray, and nothing else.
_OCCLUSION_XML = """
<mujoco>
  <worldbody>
    <body name="cube" pos="0 0 0.0125">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.03 0.02 0.0125"/>
      <site name="cube_tag" pos="0 0 0.0125" type="box" size="0.01 0.01 0.0005"/>
    </body>
    <body name="wall" pos="0 -0.2 0.05">
      <geom name="wall_geom" type="box" size="0.05 0.005 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_cube_tag_occluded_unit():
    model = mujoco.MjModel.from_xml_string(_OCCLUSION_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube_body = model.geom_bodyid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_tag")

    # Camera behind the wall: the wall geom blocks the ray.
    assert cube_tag_occluded(model, data, site_id, np.array([0.0, -0.4, 0.1]),
                             cube_body)
    # Camera on a clear line of sight: no occluder.
    assert not cube_tag_occluded(model, data, site_id, np.array([0.0, 0.4, 0.1]),
                                 cube_body)
    # The cube's own body never occludes its tag (bodyexclude), even when the
    # ray grazes the cube: camera level with the tag across the cube's face.
    assert not cube_tag_occluded(model, data, site_id, np.array([0.4, 0.1, 0.025]),
                                 cube_body)


# Corner-occlusion scene: a thin bar parked just off the tag's centre line so it
# crosses a +x corner's sightline to a front camera but never the centre ray
# (x=0). The tag box spans ±0.01 in x/y, so its +x corners sit near x=0.01.
_CORNER_OCCLUSION_XML = """
<mujoco>
  <worldbody>
    <body name="cube" pos="0 0 0.0125">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.03 0.02 0.0125"/>
      <site name="cube_tag" pos="0 0 0.0125" type="box" size="0.01 0.01 0.0005"/>
    </body>
    <body name="bar" pos="0.008 0.1 0.025">
      <geom name="bar_geom" type="box" size="0.004 0.003 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_cube_tag_corner_occluded():
    """A partial occluder blocking only a tag corner (centre in clear view)
    still hides the tag — detection is gated by the four corner rays, not just
    the centre, because a real AprilTag decode fails on a clipped quad."""
    model = mujoco.MjModel.from_xml_string(_CORNER_OCCLUSION_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube_body = model.geom_bodyid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_tag")

    # Camera in front, level with the tag: the centre ray (x=0) clears the bar,
    # but the +x corner rays cross it -> occluded.
    assert cube_tag_occluded(model, data, site_id, np.array([0.0, 0.4, 0.025]),
                             cube_body)
    # Camera behind the tag: the bar is out of every sightline -> visible.
    assert not cube_tag_occluded(model, data, site_id, np.array([0.0, -0.4, 0.025]),
                                 cube_body)


def test_arm_occlusion_forces_miss(env):
    """A cube tag facing the camera but shadowed by the arm must be undetected:
    the angle check alone would pass, only the occlusion test catches it.
    Deterministic seeded search over arm poses for a blocking configuration."""
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(3000):
        qpos = rng.uniform(env.joint_low, env.joint_high)
        env.data.qpos[env.joint_qposadr] = qpos
        mujoco.mj_forward(env.model, env.data)
        if cube_tag_occluded(env.model, env.data, env.cube_tag_site_id,
                             env.tag_cam_pos, env.cube_body_id):
            break
    else:
        pytest.fail("no arm pose occluded the cube tag in 3000 samples")
    # The spawn faced the camera and the cube hasn't moved — only occlusion
    # can hide it now.
    frame = _refresh_detection(env)
    assert not frame.cube_detected
    assert not cube_tag_visible(env.model, env.data, env.cube_tag_site_id,
                                env.tag_cam, env.cube_body_id)


def test_cube_hidden_ratio_in_info(env):
    env.reset(seed=0)
    info = {}
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        if term or trunc:
            break
    assert "cube_hidden_ratio" in info
    assert 0.0 <= info["cube_hidden_ratio"] <= 1.0
