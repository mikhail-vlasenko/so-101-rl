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

from src.base_env import JOINT_NAMES, MARKER_AGE_CAP_S, N_MARKERS, obs_dim_for, priv_dim_for
from src.units import max_joint_speed_rad_s

REPO_ROOT = Path(__file__).resolve().parent.parent
# Joint ranges and the physics timestep live in so101.xml, which every scene
# includes; the plain scene is the canonical standalone way to load them.
SCENE_XML = REPO_ROOT / "so101" / "scene.xml"
# The nominal sponge box (privileged half-extent center) lives in the cube
# scenes; lift and pickplace share the same physical sponge geom.
CUBE_SCENE_XML = REPO_ROOT / "so101" / "scene_lift.xml"

# Box the camera-space world positions (markers, cube, reach ee) live in: arm
# base at the origin, reach ~0.35 m, most action low over the table. Scale
# reference only — ObsNorm never clips, values outside just normalize past ±1.
POS_CENTER = np.array([0.1, 0.0, 0.15])
POS_SCALE = np.array([0.35, 0.35, 0.25])

# Axis-angle rotation-vector components (marker_include_rot layout): bounded
# by pi, typically around 1 in magnitude.
ROT_SCALE = np.pi / 2

# √M upper triangle (src/shape_obs.SQRTM_UPPER_ORDER): eigenvalues are the
# half extents / √3, so the diagonal is centered on the nominal upright box's
# diag (h/√3, computed below) and spans roughly [hz, hx]/√3 as the box tips;
# off-diagonals are zero-centered cross terms of the same magnitude band.
SQRTM_DIAG_SCALE = 0.02
SQRTM_OFFDIAG_SCALE = 0.01

# extra(4) dims: cube_to_target xy (pickplace, meters across the workspace;
# lift feeds zeros), ring height (0..ring_height_max=0.05 in
# conf/env/pickplace.yaml), task id (already 0/1).
EXTRA_SCALE = np.array([0.3, 0.3, 0.05, 1.0])

# ---- Privileged tail (asymmetric critic; base_env.priv_dim_for) ----
# The DR-latent scales are fixed references sized to the *full* preset
# (conf/dr/full.yaml) at 2x its sigma, mapping the ±2σ bulk onto [-1, 1].
# Deliberately constants here rather than reads of the dr yaml: the transform
# must not change between curriculum stages, or swapping dr presets on a
# resume would invalidate the checkpoint.
# Cube velocity (free-joint qvel): linear ~carry speed, worst case a drop from
# lift height (~1.4 m/s free fall); angular ~tip/tumble, tens of rad/s when
# flicked.
CUBE_LINVEL_SCALE = 1.0
CUBE_ANGVEL_SCALE = 10.0
QPOS_BIAS_SCALE = 0.02          # rad; 2x full's obs_bias.qpos_sigma
MARKER_POS_BIAS_SCALE = 0.01    # m;   2x full's obs_bias.marker_pos_sigma
MARKER_ROT_BIAS_SCALE = 0.1     # rad; 2x full's obs_bias.marker_rot_sigma
LIVE_BIAS_SCALE = 0.006         # m;   2x full's obs_bias.live_sigma
PRECISE_BIAS_SCALE = 0.006      # m;   2x full's obs_bias.precise_sigma
PRECISE_ROT_BIAS_SCALE = 0.1    # rad; 2x full's obs_bias.precise_rot_sigma
COMMON_BIAS_SCALE = 0.005       # m;   2x full's obs_bias.marker_common_sigma
# Camera pipeline delay: [0, 0.05] s maps to [-1, 1], covering both the
# measured 42-52 ms range and the synchronous camera's 0.
CAM_DELAY_CENTER = 0.025
CAM_DELAY_SCALE = 0.025
CUBE_HALF_EXTENT_SCALE = 0.005  # m; full's cube_size_jitter/2 (half-extent jitter)


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


def _cube_nominal_half_extents() -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(CUBE_SCENE_XML))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert gid >= 0, f"Cube geom 'cube_geom' not found in {CUBE_SCENE_XML}"
    return model.geom_size[gid].copy()


def _sqrtm_norm() -> tuple[np.ndarray, np.ndarray]:
    """(center, scale) of a √M upper-triangle block (both the obs precise
    channel and the priv-tail true √M): diagonal centered on the nominal
    upright box's h/√3, off-diagonals zero-centered."""
    diag = _cube_nominal_half_extents() / np.sqrt(3.0)
    center = np.concatenate([diag, np.zeros(3)])
    scale = np.concatenate([np.full(3, SQRTM_DIAG_SCALE),
                            np.full(3, SQRTM_OFFDIAG_SCALE)])
    return center, scale


def build_obs_norm(prev_actions_n: int, marker_include_rot: bool,
                   action_scale: float, n_substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """(center, scale) float32 arrays for the full SO101BaseEnv obs layout:
    the actor block (obs_dim_for: qpos, qvel, markers, marker_age, live +
    live_age, center + sqrtM + precise_age, extra, prev_actions) followed by
    the privileged tail (priv_dim_for)."""
    model = _load_model()
    qpos_center, qpos_scale = _qpos_norm(model)
    qvel_scale = _qvel_scale(model, action_scale, n_substeps)
    sqrtm_center, sqrtm_scale = _sqrtm_norm()

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
    center.append(POS_CENTER)  # live centroid
    scale.append(POS_SCALE)
    # live age: [0, cap] maps to [-1, 1], same as marker age
    center.append(np.full(1, MARKER_AGE_CAP_S / 2.0))
    scale.append(np.full(1, MARKER_AGE_CAP_S / 2.0))
    center.append(POS_CENTER)  # precise body center
    scale.append(POS_SCALE)
    center.append(sqrtm_center)  # √M upper triangle
    scale.append(sqrtm_scale)
    center.append(np.full(1, MARKER_AGE_CAP_S / 2.0))  # precise age
    scale.append(np.full(1, MARKER_AGE_CAP_S / 2.0))
    center.append(np.zeros(4))  # extra
    scale.append(EXTRA_SCALE)
    center.append(np.zeros(prev_actions_n * n_joints))  # already in [-1, 1]
    scale.append(np.ones(prev_actions_n * n_joints))

    # Privileged tail (critic-only dims; layout in base_env._priv_tail).
    center.append(POS_CENTER)  # true cube pos
    scale.append(POS_SCALE)
    center.append(np.zeros(3))  # true cube rot (axis-angle)
    scale.append(np.full(3, ROT_SCALE))
    center.append(sqrtm_center)  # true √M upper triangle
    scale.append(sqrtm_scale)
    center.append(np.zeros(6))  # cube velocity (linear + angular)
    scale.append(np.array([CUBE_LINVEL_SCALE] * 3 + [CUBE_ANGVEL_SCALE] * 3))
    center.append(np.full(2, 0.5))  # jaw contact flags: {0, 1} -> {-1, 1}
    scale.append(np.full(2, 0.5))
    center.append(np.zeros(n_joints))  # qpos bias
    scale.append(np.full(n_joints, QPOS_BIAS_SCALE))
    center.append(np.zeros(N_MARKERS * 3))  # per-tag marker pos bias
    scale.append(np.full(N_MARKERS * 3, MARKER_POS_BIAS_SCALE))
    if marker_include_rot:
        center.append(np.zeros(N_MARKERS * 3))  # per-tag marker rot bias
        scale.append(np.full(N_MARKERS * 3, MARKER_ROT_BIAS_SCALE))
    center.append(np.zeros(3))  # live centroid bias
    scale.append(np.full(3, LIVE_BIAS_SCALE))
    center.append(np.zeros(3))  # precise center bias
    scale.append(np.full(3, PRECISE_BIAS_SCALE))
    center.append(np.zeros(3))  # precise rot-perturb (axis-angle)
    scale.append(np.full(3, PRECISE_ROT_BIAS_SCALE))
    center.append(np.zeros(3))  # common-mode pos bias
    scale.append(np.full(3, COMMON_BIAS_SCALE))
    center.append(np.full(1, CAM_DELAY_CENTER))  # camera pipeline delay
    scale.append(np.full(1, CAM_DELAY_SCALE))
    center.append(_cube_nominal_half_extents())  # sponge half extents
    scale.append(np.full(3, CUBE_HALF_EXTENT_SCALE))

    center = np.concatenate(center).astype(np.float32)
    scale = np.concatenate(scale).astype(np.float32)
    expected = obs_dim_for(prev_actions_n, marker_include_rot) + \
        priv_dim_for(marker_include_rot)
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
