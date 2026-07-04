"""Fixed affine observation normalization — the (center, scale) constants the
policy's input layer applies (src/networks.ObsNorm).

Every obs dimension has a known physical scale, so nothing is estimated from
data: joint limits come from the model XML, the velocity scale derives from
src/units.py, ages from MARKER_AGE_CAP_S, and camera-space positions use the
workspace box below. A constant transform cannot be contaminated by marker
dropout rates or curriculum-stage distribution shifts, and it is identical in
rollout collection, gradient updates, eval, and on the real arm (the buffers
ship inside the checkpoint). Precision is uncritical — every hidden layer is
LayerNorm'ed, so scales only need to be right within ~2x — but the values must
be stable: changing them invalidates existing checkpoints exactly like an
obs-dim change.

tests/training/test_obs_norm.py rolls random episodes under full DR and
asserts every normalized dimension stays within sane bounds — that test keeps
the hand-set boxes honest if the workspace ever changes.
"""

from pathlib import Path

import mujoco
import numpy as np

from src.base_env import JOINT_NAMES, MARKER_AGE_CAP_S, N_MARKERS, obs_dim_for
from src.units import max_joint_speed_rad_s

REPO_ROOT = Path(__file__).resolve().parent.parent
# Joint ranges and the physics timestep live in so101.xml, which every scene
# includes; the plain scene is the canonical standalone way to load them.
SCENE_XML = REPO_ROOT / "so101" / "scene.xml"

# Box the camera-space world positions (markers, cube, reach ee) live in: arm
# base at the origin, reach ~0.35 m, most action low over the table. Scale
# reference only — ObsNorm never clips, values outside just normalize past ±1.
POS_CENTER = np.array([0.1, 0.0, 0.15])
POS_SCALE = np.array([0.35, 0.35, 0.25])

# Axis-angle rotation-vector components (marker_include_rot layout): bounded
# by pi, typically around 1 in magnitude.
ROT_SCALE = np.pi / 2

# extra(4) dims: cube_to_target xy (pickplace, meters across the workspace;
# lift feeds zeros), ring height (0..ring_height_max=0.05 in
# conf/env/pickplace.yaml), task id (already 0/1).
EXTRA_SCALE = np.array([0.3, 0.3, 0.05, 1.0])


def _load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def _qpos_norm(model) -> tuple[np.ndarray, np.ndarray]:
    """Joint-range midpoint and half-range per JOINT_NAMES entry."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    lo = model.jnt_range[ids, 0]
    hi = model.jnt_range[ids, 1]
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _qvel_scale(model, action_scale: float, n_substeps: int) -> float:
    """Peak commandable joint speed: one full-scale action step per tick."""
    control_hz = 1.0 / (model.opt.timestep * n_substeps)
    return max_joint_speed_rad_s(action_scale, control_hz)


def build_obs_norm(prev_actions_n: int, marker_include_rot: bool,
                   action_scale: float, n_substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """(center, scale) float32 arrays for the SO101BaseEnv obs layout
    (obs_dim_for): qpos, qvel, markers, marker_age, cube_pos, extra,
    prev_actions."""
    model = _load_model()
    qpos_center, qpos_scale = _qpos_norm(model)
    qvel_scale = _qvel_scale(model, action_scale, n_substeps)

    n_joints = len(JOINT_NAMES)
    center = [qpos_center, np.zeros(n_joints)]
    scale = [qpos_scale, np.full(n_joints, qvel_scale)]
    for _ in range(N_MARKERS):
        center.append(POS_CENTER)
        scale.append(POS_SCALE)
        if marker_include_rot:
            center.append(np.zeros(3))
            scale.append(np.full(3, ROT_SCALE))
    # marker age: [0, cap] maps to [-1, 1]
    center.append(np.full(N_MARKERS, MARKER_AGE_CAP_S / 2.0))
    scale.append(np.full(N_MARKERS, MARKER_AGE_CAP_S / 2.0))
    center.append(POS_CENTER)  # cube_pos
    scale.append(POS_SCALE)
    center.append(np.zeros(4))  # extra
    scale.append(EXTRA_SCALE)
    center.append(np.zeros(prev_actions_n * n_joints))  # already in [-1, 1]
    scale.append(np.ones(prev_actions_n * n_joints))

    center = np.concatenate(center).astype(np.float32)
    scale = np.concatenate(scale).astype(np.float32)
    expected = obs_dim_for(prev_actions_n, marker_include_rot)
    assert center.shape == (expected,), (center.shape, expected)
    return center, scale


def build_reach_obs_norm(prev_actions_n: int, n_waypoints: int,
                         action_scale: float, n_substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """(center, scale) for SO101ReachEnv._get_obs: qpos, qvel, ee_pos,
    waypoint one-hot, prev_actions."""
    model = _load_model()
    qpos_center, qpos_scale = _qpos_norm(model)
    qvel_scale = _qvel_scale(model, action_scale, n_substeps)

    n_joints = len(JOINT_NAMES)
    center = np.concatenate([qpos_center, np.zeros(n_joints), POS_CENTER,
                             np.zeros(n_waypoints),
                             np.zeros(prev_actions_n * n_joints)]).astype(np.float32)
    scale = np.concatenate([qpos_scale, np.full(n_joints, qvel_scale), POS_SCALE,
                            np.ones(n_waypoints),
                            np.ones(prev_actions_n * n_joints)]).astype(np.float32)
    return center, scale
