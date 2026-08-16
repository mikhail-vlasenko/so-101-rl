"""Base gymnasium environment for SO-101 arm tasks.

Shared MuJoCo setup, contact detection, rendering, and reset/step skeleton.
Subclasses define task-specific config, reward, termination, and observations.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from src.camera_sim import CameraSim
from src.bps import (
    BPS_OBS_DIM,
    BPSConfig,
    BPSMeasurement,
    BPSObsState,
    load_bps_config,
)
from src.marker_noise import CameraIntrinsics, anisotropic_pos_noise, load_camera_intrinsics
from src.obs_history import ObsHistory
from src.robot_spec import EE_SITE_NAME, JOINT_NAMES
from src.servo_profile import ServoProfile
from src.shape_obs import (
    MARKER_AGE_CAP_S,
    STATIC_DWELL_S,
    VISIBLE_FRACTION_MIN,
    ObjectChannelDriver,
    ObjectObsState,
)
from src.sim_bps import (
    SyntheticBPSGenerator,
    SyntheticCloudConfig,
    clean_synthetic_cloud_config,
)
from src.surface_cloud import (
    transform_box_surface_points_world,
    unit_box_surface_points,
    visible_surface as cube_visible_surface,
    visible_surface_mask,
    visible_surface_summary,
)
from src.units import action_to_target
from real.marker_spec import ARM_TAG_TO_SITE, TAG_SIZE_MM

EE_OBJECT_DELTA_DIM = 3


def ee_object_delta(ee_pos: np.ndarray, object_center: np.ndarray) -> np.ndarray:
    """Deployable relative-position feature shared by sim and real rollouts."""
    return np.asarray(ee_pos) - np.asarray(object_center)


def state_dim_for(prev_actions_n: int, marker_include_rot: bool = False) -> int:
    """Width of one historical actor-state frame.

    The frame contains qpos/qvel, arm marker poses/ages, the fast live object
    centroid/age, task extras, previous actions, and the derived end-effector
    minus held live-centroid vector.  The current BPS block is deliberately
    outside this frame so history taps never duplicate it.

    Each marker contributes its world xyz (M=3); when marker_include_rot is true it
    also contributes a world rotation vector (axis-angle, +3 dims, M=6) — the same
    quantities the camera pipeline recovers for the physical AprilTags (real/vision/pose.py
    rvec/tvec, camera->world mapped). marker_age is each tag's seconds since the
    capture of the pose it is serving (capped at MARKER_AGE_CAP_S): pipeline
    latency when freshly detected, growing while the tag is undetected and its
    last pose is held.

    The live centroid is biased toward the visible surface and goes stale when
    either camera loses the object.  The precise visible surface is represented
    by :mod:`src.bps` and appended exactly once after all state taps.
    """
    marker_dim = 6 if marker_include_rot else 3
    return (12 + N_MARKERS * marker_dim + N_MARKERS + 8
            + prev_actions_n * 6 + EE_OBJECT_DELTA_DIM)


def obs_dim_for(prev_actions_n: int, marker_include_rot: bool = False,
                history_taps: tuple[int, ...] = (0,)) -> int:
    """Actor width for ``[state history | current BPS block]``."""
    return (len(history_taps) * state_dim_for(prev_actions_n, marker_include_rot)
            + BPS_OBS_DIM)


def legacy_tag_actor_dim_for(prev_actions_n: int,
                             marker_include_rot: bool = False) -> int:
    """Single-frame actor width of the last deployable tag-observation teacher."""
    marker_dim = 6 if marker_include_rot else 3
    return 12 + N_MARKERS * marker_dim + N_MARKERS + 11 + prev_actions_n * 6


def priv_dim_for(marker_include_rot: bool = False) -> int:
    """Privileged tail appended after the actor block (asymmetric critic):
    true cube pose (3+3, world pos + axis-angle), cube velocity (6,
    free-joint qvel), jaw contact flags (2), and the
    episode's sampled DR latents: qpos bias (6), per-tag marker pos bias
    (2*3), marker rot bias (2*3, only when the rot obs it perturbs is in the
    actor block), live centroid bias (3), precise center bias (3), common-mode
    pos bias (3), camera pipeline delay (1),
    sponge half extents (3). Contents built by
    SO101BaseEnv._priv_tail; normalization constants in src/obs_norm.py."""
    dim = 14 + 6 + N_MARKERS * 3 + 6 + 3 + 1 + 3
    if marker_include_rot:
        dim += N_MARKERS * 3
    return dim


# AprilTags glued to the arm (ids per real/marker_spec.py ROLES): "finger" on
# the bottom face of the fixed jaw, "wrist" on the right face of the wrist.
# Sites of the same names in so101.xml mirror their placement.
MARKER_SITE_NAMES = ["marker_finger", "marker_wrist"]
N_MARKERS = len(MARKER_SITE_NAMES)

# AprilTag id 1 on the sponge's largest face. The obs path no longer reads it
# (the cube channels are tag-free) — the site survives only as the GT anchor
# for legacy-teacher distillation (distill.teacher_obs=legacy_tag; delete with
# it, see TODO.md). MARKER_AGE_CAP_S itself lives in src/shape_obs.py (shared
# with the real pipeline) and is re-exported here for the marker path.
CUBE_TAG_SITE_NAME = "cube_tag"

# Fixed cameras in so101.xml standing in for the two physical webcams: tag_cam
# (main) watches the AprilTags, and together with tag_cam_aux drives the
# dual-view cube channels. A tag counts as visible when it projects inside the
# main camera frame (CameraModel.in_view — the real camera's field of view)
# *and* the angle between its outward normal (+z of the site frame) and the
# tag->camera ray is under MAX — past that the AprilTag detector loses the
# grazing view. NEAR marks a softer band: from NEAR up to MAX the view grazes
# and the detector flakes, so the DR dropout (marker_dropout_prob) drops those
# tags more often than tags comfortably facing the camera.
TAG_CAM_NAME = "tag_cam"
TAG_CAM_AUX_NAME = "tag_cam_aux"
MARKER_VIS_MAX_ANGLE_DEG = 70.0
MARKER_VIS_NEAR_ANGLE_DEG = 65.0
_MARKER_VIS_COS_MIN = np.cos(np.radians(MARKER_VIS_MAX_ANGLE_DEG))
_MARKER_VIS_COS_NEAR = np.cos(np.radians(MARKER_VIS_NEAR_ANGLE_DEG))

# Render-only tints for the marker sites: green when the tag is detected this
# frame, red when not (over-angle or dropped). Visual only — never read by obs.
MARKER_VISIBLE_RGBA = np.array([0.1, 0.8, 0.1, 1.0])
MARKER_HIDDEN_RGBA = np.array([0.9, 0.1, 0.1, 1.0])


def set_marker_render_colors(model, site_ids, detected: np.ndarray) -> None:
    """Tint marker sites green when currently detected and red otherwise."""
    detected = np.asarray(detected, dtype=bool)
    if detected.shape != (len(site_ids),):
        raise ValueError(
            f"detected must have shape ({len(site_ids)},), got {detected.shape}")
    for sid, seen in zip(site_ids, detected):
        model.site_rgba[sid] = MARKER_VISIBLE_RGBA if seen else MARKER_HIDDEN_RGBA


@dataclass(frozen=True)
class RuntimeEnvConfig:
    """Runtime observation/DR options shared across envs."""
    obs_noise: dict | None = None
    cam_latency: dict | None = None
    obs_bias: dict | None = None
    marker_dropout: dict | None = None
    marker_always_visible: bool = False
    marker_include_rot: bool = False
    prev_actions_n: int = 2
    cube_size_jitter: float = 0.0
    history_taps: tuple = (0,)
    bps_config: BPSConfig | None = None
    synthetic_cloud: SyntheticCloudConfig | None = None


def marker_world_poses(data, site_ids):
    """World poses of the marker sites: (pos (N,3), rot (N,3)).

    rot is an axis-angle rotation vector, matching the Rodrigues-vector
    convention of cv2.solvePnP in real/vision/pose.py.
    """
    pos = np.empty((len(site_ids), 3))
    rot = np.empty((len(site_ids), 3))
    quat = np.empty(4)
    for i, sid in enumerate(site_ids):
        pos[i] = data.site_xpos[sid]
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        mujoco.mju_quat2Vel(rot[i], quat, 1.0)
    return pos, rot


def marker_world_normals(data, site_ids):
    """Outward tag normals (N,3): +z of each marker site's world frame."""
    return np.stack([data.site_xmat[sid].reshape(3, 3)[:, 2] for sid in site_ids])


@dataclass(frozen=True)
class CameraModel:
    """The fixed tag camera as a calibrated pinhole: its world pose (cam_xpos /
    cam_xmat, MuJoCo convention — the camera looks down its local -z with +y up)
    plus the intrinsics from real/vision/camera_intrinsics.yaml.

    in_view() is the field-of-view test the marker-visibility pipeline uses in
    place of the old world-height proxy: a world point is projected through the
    real camera matrix and must land inside the image, exactly the bound a real
    detection has (a tag off the frame is simply never seen). Distortion is
    ignored — the coeffs are small and this is a coarse in/out gate softened by
    the grazing-angle band and the DR dropout.
    """
    pos: np.ndarray   # (3,) world position
    mat: np.ndarray   # (3,3) world rotation, columns = camera-frame axes in world
    intr: CameraIntrinsics

    def in_view(self, points):
        """Bool (N,) — each world point of `points` (N,3) projects inside the frame."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        # World -> camera frame: (R^T @ rel) per row is (rel @ R) for row vectors.
        cam = (pts - self.pos) @ self.mat
        depth = -cam[:, 2]                       # camera looks down -z; in front -> depth > 0
        in_front = depth > 0
        safe_depth = np.where(in_front, depth, 1.0)  # avoid 0-division for points behind
        # MuJoCo cam frame is +y up; the OpenCV pixel v axis points down, so flip y.
        u = self.intr.fx * (cam[:, 0] / safe_depth) + self.intr.cx
        v = self.intr.fy * (-cam[:, 1] / safe_depth) + self.intr.cy
        return (in_front & (u >= 0) & (u < self.intr.width)
                & (v >= 0) & (v < self.intr.height))


def tag_cam_world_pos(model, data):
    """World position of TAG_CAM_NAME. The camera hangs off a fixed mount body,
    so cam_xpos (filled by forward kinematics) is its constant world position;
    model.cam_pos is body-relative and would be wrong. Runs mj_kinematics so the
    caller need not have forwarded data first (the mount is qpos-independent)."""
    return tag_cam_model(model, data).pos


def tag_cam_model(model, data, name=TAG_CAM_NAME, camera="main"):
    """CameraModel for a fixed scene camera: its world pose (cam_xpos /
    cam_xmat, both filled by mj_camlight — the mounts are qpos-independent, so
    this needs no prior forward) plus that physical unit's calibrated
    intrinsics (real/vision/intrinsics.py names). Single source for the camera
    geometry the visibility checks consume; defaults to the main tag camera,
    pass (TAG_CAM_AUX_NAME, "aux") for the second unit."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    assert cam_id >= 0, f"Camera '{name}' not found in model"
    # cam_xpos/cam_xmat are filled by mj_camlight, which needs body frames from mj_kinematics.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    return CameraModel(pos=data.cam_xpos[cam_id].copy(),
                       mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                       intr=load_camera_intrinsics(camera))


def markers_visible(data, site_ids, cam):
    """Per-marker geometric visibility from the CameraModel `cam` as a bool array:
    inside the camera frame (cam.in_view) *and* plane angle under
    MARKER_VIS_MAX_ANGLE_DEG (see TAG_CAM_NAME doc). No stochastic dropout — used
    by tests and the real-arm twin rollout."""
    pos, _ = marker_world_poses(data, site_ids)
    normals = marker_world_normals(data, site_ids)
    return marker_dropout_prob(pos, normals, cam, p_near=0.0, p_far=0.0) < 1.0


def marker_dropout_prob(pos, normals, cam, p_near, p_far):
    """Per-marker probability the tag is undetected this frame (its held pose
    goes stale), from the tags' world positions (N,3) and outward normals (N,3)
    and the CameraModel `cam`.

    Projecting outside the camera frame (cam.in_view — the real FOV) or past
    MARKER_VIS_MAX_ANGLE_DEG the camera can't see the tag at all -> 1.0.
    In the near-boundary band [NEAR, MAX] the grazing view is flaky -> p_near.
    Comfortably facing the camera (angle < NEAR) -> p_far (rare detector miss).
    """
    in_view = cam.in_view(pos)
    prob = np.empty(len(pos))
    for i in range(len(pos)):
        to_cam = cam.pos - pos[i]
        cos = (normals[i] @ to_cam) / np.linalg.norm(to_cam)
        if not in_view[i] or cos < _MARKER_VIS_COS_MIN:
            prob[i] = 1.0
        elif cos < _MARKER_VIS_COS_NEAR:
            prob[i] = p_near
        else:
            prob[i] = p_far
    return prob


CUBE_SURFACE_UNIT_POINTS, CUBE_SURFACE_UNIT_NORMALS = unit_box_surface_points(3)


def cube_surface_points_world(data, cube_geom_id, half_extents):
    """(points (54, 3), normals (54, 3)) of the box-surface sample grid in
    world coordinates, from the cube geom's current world pose."""
    rotation = data.geom_xmat[cube_geom_id].reshape(3, 3)
    center = data.geom_xpos[cube_geom_id]
    points = center + (CUBE_SURFACE_UNIT_POINTS
                       * np.asarray(half_extents)) @ rotation.T
    return points, CUBE_SURFACE_UNIT_NORMALS @ rotation.T


class CamState(NamedTuple):
    """World-state snapshot recorded per physics substep for CameraSim: what a
    camera frame captured at that instant would see. Occlusion is resolved here,
    at capture time, where live MjData exists — the frame may be consumed ticks
    later, when the world has moved on."""
    marker_pos: np.ndarray     # (N_MARKERS, 3)
    marker_rot: np.ndarray     # (N_MARKERS, 3) axis-angle
    marker_normal: np.ndarray  # (N_MARKERS, 3) outward tag normals
    cube_center: np.ndarray    # (3,) GT box center
    cube_vis_frac: np.ndarray  # (2,) per-camera visible fraction (main, aux)
    cube_vis_centroid: np.ndarray  # (2, 3) per-camera visible-point centroid
    cube_seen: np.ndarray      # (2,) bool — any surface point visible per camera
    live_detected: bool
    bps_measurement: BPSMeasurement | None


class CamFrame(NamedTuple):
    """One processed detection: dropout, bias, and noise frozen at capture time.
    marker_pos/rot entries are only meaningful where detected, live only when
    live_detected — _ingest_frame folds the detected ones into the held state
    the obs serves. The synthetic BPS measurement was resolved against the
    captured world and is held here until the frame becomes available; the
    ingest-time static gate decides whether it refreshes the precise state."""
    marker_pos: np.ndarray  # (N_MARKERS, 3)
    marker_rot: np.ndarray  # (N_MARKERS, 3)
    detected: np.ndarray    # (N_MARKERS,) bool
    live: np.ndarray        # (3,) measured live centroid (visible-surface avg)
    live_gate: np.ndarray   # (3,) pre-noise live measurement for the static gate
    live_priv: np.ndarray   # (3,) privileged live: GT center, same bias/noise
    live_detected: bool
    vis_frac: np.ndarray    # (2,) per-camera visible fraction at capture
    bps_measurement: BPSMeasurement | None


# Resting orientations of the box: which body axis is vertical (with sign) and
# the matching half-extent index for the rest height. Quats are the 90-degree
# tilts (or identity/flip for z) that put that axis vertical; a uniform yaw is
# composed on top in sample_cube_orientation.
_S = np.sqrt(0.5)
_REST_FACES = {
    "x_down": (np.array([_S, 0.0, _S, 0.0]), 0),   # +x -> -z (legacy crutch pose)
    "x_up": (np.array([_S, 0.0, -_S, 0.0]), 0),
    "y_up": (np.array([_S, _S, 0.0, 0.0]), 1),
    "y_down": (np.array([_S, -_S, 0.0, 0.0]), 1),
    "z_up": (np.array([1.0, 0.0, 0.0, 0.0]), 2),   # flat on the largest face
    "z_down": (np.array([0.0, 1.0, 0.0, 0.0]), 2),
}
_SIDE_FACES = ("x_down", "x_up", "y_up", "y_down")
_FLAT_FACES = ("z_up", "z_down")


def sample_cube_orientation(rng, half_extents, smallest_face_only=False,
                            no_flat_spawns=False):
    """Spawn orientation for the sponge box: resting on any face + free yaw.

    half_extents (hx, hy, hz) must be strictly ordered hx > hy > hz (the
    6 x 4 x 2.5 cm box). By default the box rests on any of its six faces —
    flat on a largest hx*hy face included, since the tag-free object channels
    impose no facing constraint — chosen uniformly, with a uniform random yaw.
    Returns (quat wxyz, rest_half_z).

    Curriculum crutches (conf/config.yaml, no obs-dim change):
    - smallest_face_only: always stand on the smallest hy*hz face (x-axis
      vertical, tallest, easiest-to-grasp pose every episode).
    - no_flat_spawns: exclude the flat largest-face poses, i.e. the historic
      side-standing spawn distribution.
    """
    hx, hy, hz = half_extents
    assert hx > hy > hz, f"expected strictly ordered half extents, got {half_extents}"
    if smallest_face_only:
        face = "x_down"
    elif no_flat_spawns:
        face = _SIDE_FACES[rng.integers(len(_SIDE_FACES))]
    else:
        faces = _SIDE_FACES + _FLAT_FACES
        face = faces[rng.integers(len(faces))]
    tilt, rest_axis = _REST_FACES[face]
    yaw = rng.uniform(-np.pi, np.pi)
    yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, yaw_quat, tilt)
    return quat, float(half_extents[rest_axis])

FIXED_JAW_NAMES = [
    "fixed_jaw_box1", "fixed_jaw_box2", "fixed_jaw_box3",
    "fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6",
    "fixed_jaw_box7", "fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2",
    "fixed_jaw_sph_tip3",
]
MOVING_JAW_NAMES = [
    "moving_jaw_box1", "moving_jaw_box2", "moving_jaw_box3",
    "moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3",
]
class SO101ArmEnv(gym.Env):
    """Shared SO-101 arm machinery: model/joint setup, the clip ->
    action_to_target -> servo-profile -> substep drive (_apply_action),
    prev-actions bookkeeping, collision/floor checks, and rendering.

    Task envs own observation, reward, and reset on top: SO101BaseEnv layers
    the cube + AprilTag camera observation pipeline here; SO101ReachEnv builds
    its waypoint task directly on this class (no cube in its scene).
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    XML_PATH: str  # subclasses must set
    TASK_NAME: str

    def __init__(self, render_mode, slow_factor, xml_path, prev_actions_n, env_cfg):
        super().__init__()
        self.render_mode = render_mode
        self.slow_factor = slow_factor
        self.prev_actions_n = int(prev_actions_n)

        self.model = mujoco.MjModel.from_xml_path(xml_path or self.XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.n_joints = len(JOINT_NAMES)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in JOINT_NAMES]
        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert self.floor_geom_id >= 0, "Floor geom 'floor' not found in XML"
        self.arm_geom_ids = {i for i in range(self.model.ngeom) if self.model.geom_group[i] == 3}

        self.joint_low = self.model.jnt_range[self.joint_ids, 0]
        self.joint_high = self.model.jnt_range[self.joint_ids, 1]

        cfg = env_cfg
        self.action_scale = float(cfg["action_scale"])
        self.use_servo_profile = bool(cfg["use_servo_profile"])
        self.max_steps = int(cfg["max_steps"])
        self.n_substeps = int(cfg["n_substeps"])
        self._step_dt = self.model.opt.timestep * self.n_substeps

        self._prev_actions = np.zeros((self.prev_actions_n, self.n_joints), dtype=np.float32)

        # Firmware motion-profile model: ctrl carries the profiled setpoint,
        # not the raw tick target (see src/servo_profile.py).
        self._servo_profile = ServoProfile(self.n_joints)
        self._ctrl_target = None

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_joints,), dtype=np.float32)

        self.step_count = 0
        self.viewer = None

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_joint_pos(self):
        return self.data.qpos[self.joint_qposadr].copy()

    def _has_arm_collision(self):
        """Check if any arm/gripper geom has a contact (with environment or itself)."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.arm_geom_ids or c.geom2 in self.arm_geom_ids:
                return True
        return False

    def _has_floor_contact(self):
        """Check if any arm/gripper geom is in contact with the floor."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor_geom_id and g2 in self.arm_geom_ids:
                return True
            if g2 == self.floor_geom_id and g1 in self.arm_geom_ids:
                return True
        return False

    def _min_arm_floor_dist(self, distmax: float) -> float:
        """Min signed distance (m) between any arm geom and the floor, capped at distmax.
        Negative values mean penetration. distmax bounds the broadphase work — pairs
        further apart than distmax return distmax without doing exact collision math."""
        min_dist = distmax
        for gid in self.arm_geom_ids:
            d = mujoco.mj_geomDistance(self.model, self.data, gid,
                                       self.floor_geom_id, distmax, None)
            if d < min_dist:
                min_dist = d
        return min_dist

    def _arm_floor_contact_force(self):
        """Sum of normal contact force magnitudes (N) between any arm geom and the floor."""
        total = 0.0
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == self.floor_geom_id and g2 in self.arm_geom_ids) or \
                    (g2 == self.floor_geom_id and g1 in self.arm_geom_ids):
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                total += abs(wrench[0])
        return total

    def _write_arm_pose(self, qpos):
        """Write an arm pose plus the matching ctrl and actuator activation
        state (dyntype='filter' actuators drive toward act, so leaving it stale
        would snap the arm on the first step), then forward the model."""
        self.data.qpos[self.joint_qposadr] = qpos
        self.data.ctrl[:self.n_joints] = qpos
        if self.model.na > 0:
            self.data.act[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def _on_substep(self):
        """Hook after each physics substep of _apply_action (SO101BaseEnv
        records camera states here)."""

    def _apply_action(self, action):
        """Clip the raw policy action, record it in the prev-actions buffer,
        and drive the sim one control tick (action_to_target quantization, the
        servo profile when enabled, n_substeps physics steps). Returns the
        clipped action."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        if self.prev_actions_n > 0:
            if self.prev_actions_n > 1:
                self._prev_actions[:-1] = self._prev_actions[1:]
            self._prev_actions[-1] = action

        current = self.data.qpos[self.joint_qposadr].copy()
        target = action_to_target(current, action, self.action_scale,
                                  self.joint_low, self.joint_high)
        if self.use_servo_profile:
            setpoints = self._servo_profile.tick(self._ctrl_target, target,
                                                 self.n_substeps, self.model.opt.timestep)
            for k in range(self.n_substeps):
                self.data.ctrl[:self.n_joints] = setpoints[k]
                mujoco.mj_step(self.model, self.data)
                self._on_substep()
            self._ctrl_target = target
        else:
            self.data.ctrl[:self.n_joints] = target
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
                self._on_substep()
        return action

    def _render_human(self):
        import time
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        if self.slow_factor > 1:
            time.sleep(self.model.opt.timestep * self.n_substeps * (self.slow_factor - 1))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class SO101BaseEnv(SO101ArmEnv):
    """Base class for the cube-manipulation tasks: adds the AprilTag marker
    pipeline (hold-last-pose ages, DR noise/bias/latency/dropout), the
    tag-free dual-view cube channels (live centroid + static-gated BPS visible
    surface) and the cube-centric reset/step skeleton on
    top of SO101ArmEnv."""

    TASK_ID: float  # 0.0 = lift, 1.0 = pickplace

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1, xml_path=None,
                 cfg: RuntimeEnvConfig | None = None):
        assert cfg is not None, "RuntimeEnvConfig is required"
        super().__init__(render_mode=render_mode, slow_factor=slow_factor,
                         xml_path=xml_path, prev_actions_n=cfg.prev_actions_n,
                         env_cfg=env_cfg)
        # dict with keys qpos_sigma, marker_rot_sigma, tag_px_noise,
        # tag_depth_factor, live_sigma, precise_sigma; or
        # None. Marker position noise is anisotropic in the camera frame
        # (src/marker_noise.py): tag_px_noise with tag_depth_factor derive the
        # depth-vs-lateral split. The cube channels use isotropic sigmas —
        # live_sigma per frame on the triangulated centroid (two-view
        # triangulation has no solvePnP depth pathology) and precise_sigma per
        # refresh on the dense cloud center. BPS surface corruption is modeled
        # by SyntheticCloudConfig rather than by perturbing encoded distances.
        # The camera
        # re-anchor's common-mode error is per-episode only
        # (obs_bias.marker_common_sigma): the real pipeline EMAs the static
        # camera pose (real/rollout/marker_obs.py), leaving no meaningful per-frame
        # common jitter. No qvel key: the qvel obs is the backward difference of
        # consecutive qpos obs (matching the real pipeline), so its noise is
        # inherited from qpos_sigma, not configured.
        self.obs_noise = cfg.obs_noise
        # AprilTag detector dropout (DR): dict with keys "near"/"far" giving the
        # per-frame probability a geometrically-visible tag is missed (near-boundary
        # vs comfortably-facing), or None for geometric visibility only.
        self.marker_dropout = cfg.marker_dropout
        # Easy-mode crutch: feed every tag to the policy regardless of camera angle
        # or dropout (no held/stale poses). See conf/config.yaml:marker_always_visible.
        self.marker_always_visible = bool(cfg.marker_always_visible)
        # When false the marker rotation vectors are dropped from the obs (positions
        # only). Changes obs dim — see obs_dim_for / conf/config.yaml:marker_include_rot.
        self.marker_include_rot = bool(cfg.marker_include_rot)
        # Camera latency model (sim2real): dict with frame_ms, object_frame_ms,
        # marker_delay_ms for
        # AprilTag markers, live_delay_ms for the dual-SAM centroid,
        # bps_delay_ms for dense stereo, and jitter_ms (see conf/dr/full.yaml),
        # or None for a synchronous zero-latency camera. Stored raw here; the
        # CameraSim is built below once the control period is known.
        self.cam_latency = cfg.cam_latency
        # dict with keys qpos_sigma, marker_pos_sigma, marker_rot_sigma,
        # live_sigma, precise_sigma, marker_common_sigma; or
        # None. The per-tag marker biases are sampled independently (each tag's
        # own glue/pose-estimate error). The cube channels get per-episode
        # biases of their own: live_sigma the systematic visible-surface/
        # segmentation offset and precise_sigma the dense-cloud center offset.
        # marker_common_sigma adds a separate
        # shared shift applied to every camera-derived position (the camera
        # re-anchor / table calibration offset, correlated across tags and the
        # cube channels alike) — see _common_pos_bias.
        self.obs_bias = cfg.obs_bias
        self._qpos_bias = np.zeros(len(JOINT_NAMES))
        self._marker_pos_bias = np.zeros((N_MARKERS, 3))
        self._marker_rot_bias = np.zeros((N_MARKERS, 3))
        self._live_bias = np.zeros(3)
        self._precise_bias = np.zeros(3)
        # Common-mode shift shared by every tag (arm markers + cube) for the whole
        # episode: the real pipeline re-anchors the camera from one table tag
        # (real/rollout/marker_obs.py), so its calibration/detection error moves all tags
        # together, not independently. Drawn in reset from obs_bias.
        self._common_pos_bias = np.zeros(3)
        self.bps_config = cfg.bps_config or load_bps_config()
        cloud_config = cfg.synthetic_cloud or clean_synthetic_cloud_config()
        self._bps_generator = SyntheticBPSGenerator(self.bps_config, cloud_config)
        self._live_surface_count = CUBE_SURFACE_UNIT_POINTS.shape[0]
        self._combined_surface_unit_points = np.concatenate(
            (CUBE_SURFACE_UNIT_POINTS, self._bps_generator.unit_points))
        self._combined_surface_unit_normals = np.concatenate(
            (CUBE_SURFACE_UNIT_NORMALS, self._bps_generator.unit_normals))

        # The actor sees tapped state frames followed by one current BPS block.
        self.state_dim = state_dim_for(self.prev_actions_n, self.marker_include_rot)
        self.obs_dim = obs_dim_for(self.prev_actions_n, self.marker_include_rot,
                                   cfg.history_taps)
        self.priv_dim = priv_dim_for(self.marker_include_rot)
        # Lag-tap history (src/obs_history.py): the served observation is the
        # state frame at each configured tick lag. The current BPS block and
        # privileged tail are appended once after the history.
        self._history = ObsHistory(cfg.history_taps, self.state_dim)

        self.marker_site_ids = []
        for name in MARKER_SITE_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            assert sid >= 0, f"Marker site '{name}' not found in XML"
            self.marker_site_ids.append(sid)
        # Full pinhole cameras (world pose + per-unit calibrated intrinsics):
        # tag_cam (main) gates the arm markers' FOV/angle visibility and,
        # together with tag_cam_aux, drives the dual-view cube channels
        # (cube_visible_surface per camera, main first).
        self.tag_cam = tag_cam_model(self.model, self.data)
        self.tag_cam_pos = self.tag_cam.pos
        self.tag_cam_aux = tag_cam_model(self.model, self.data,
                                         TAG_CAM_AUX_NAME, "aux")
        self.cube_cams = (self.tag_cam, self.tag_cam_aux)
        # Camera geometry for the anisotropic (camera-frame) marker noise: focal
        # length from the solved intrinsics and each obs tag's printed edge length,
        # both the same calibration the real solvePnP pipeline uses (src/marker_noise.py,
        # real/marker_spec.py). All arm tags happen to be 20 mm, but map through
        # marker_spec so the obs slots stay the single source of truth.
        self._focal_px = self.tag_cam.intr.focal_px
        site_to_tag = {site: tag for tag, site in ARM_TAG_TO_SITE.items()}
        self._marker_tag_sizes = np.array(
            [TAG_SIZE_MM[site_to_tag[name]] / 1000.0 for name in MARKER_SITE_NAMES])

        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        assert self.cube_geom_id >= 0, "Cube geom 'cube_geom' not found in XML"
        self.cube_body_id = int(self.model.geom_bodyid[self.cube_geom_id])
        # Legacy tag site: not on the obs path (the cube channels are
        # tag-free); kept as the GT anchor for legacy-teacher distillation and
        # doubling as the live-channel render tint target.
        self.cube_tag_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                                  CUBE_TAG_SITE_NAME)
        assert self.cube_tag_site_id >= 0, \
            f"Cube tag site '{CUBE_TAG_SITE_NAME}' not found in XML"

        self.fixed_jaw_geom_ids = set()
        self.moving_jaw_geom_ids = set()
        for name in FIXED_JAW_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"Gripper geom '{name}' not found in XML"
            self.fixed_jaw_geom_ids.add(gid)
        for name in MOVING_JAW_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"Gripper geom '{name}' not found in XML"
            self.moving_jaw_geom_ids.add(gid)
        self.gripper_geom_ids = self.fixed_jaw_geom_ids | self.moving_jaw_geom_ids

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_idx = self.model.jnt_qposadr[cube_joint_id]
        self.cube_dofadr = self.model.jnt_dofadr[cube_joint_id]
        # Sponge box half extents. The nominal value is the XML box; the actual
        # per-episode extents (self.cube_half_extents) are resampled each reset
        # when cube_size_jitter > 0 (DR sponge resize). The per-episode resting
        # half-height (depends on which face it stands on) is set in reset() as
        # self.cube_rest_half_z.
        self.cube_nominal_half_extents = self.model.geom_size[self.cube_geom_id].copy()
        self.cube_half_extents = self.cube_nominal_half_extents.copy()
        # Half-extent jitter: each full side length varies +/- cube_size_jitter,
        # i.e. each half-extent varies +/- cube_size_jitter/2. See conf/dr/*.yaml.
        self.cube_size_half_jitter = float(cfg.cube_size_jitter) / 2.0

        task_cfg = env_cfg
        self.cube_low = np.array(task_cfg["cube_low"])
        self.cube_high = np.array(task_cfg["cube_high"])
        self.cube_smallest_face_only = bool(task_cfg["cube_smallest_face_only"])
        self.cube_no_flat_spawns = bool(task_cfg["cube_no_flat_spawns"])
        self.floor_contact_penalty = float(task_cfg["floor_contact_penalty"])

        self.gripper_idx = JOINT_NAMES.index("gripper")

        # Cube-drag metric: cube center within DRAG_HEIGHT_TOL of resting height
        # AND lateral speed above DRAG_SPEED_THRESH (m/s).
        self.DRAG_HEIGHT_TOL = 0.005
        self.DRAG_SPEED_THRESH = 0.01

        # Marker/cube obs come from simulated camera frames (src/camera_sim.py);
        # qpos/qvel are encoder-path and stay fresh. _cam_frame is the frame the
        # policy currently sees; _prev_qpos_obs feeds the differenced qvel obs.
        if self.cam_latency is None:
            self._camera = CameraSim.synchronous(self._step_dt,
                                                 self.model.opt.timestep)
        else:
            self._camera = CameraSim(
                frame_s=float(self.cam_latency["frame_ms"]) * 1e-3,
                object_frame_s=(
                    float(self.cam_latency["object_frame_ms"]) * 1e-3),
                marker_delay_range_s=(
                    float(self.cam_latency["marker_delay_ms"][0]) * 1e-3,
                    float(self.cam_latency["marker_delay_ms"][1]) * 1e-3),
                live_delay_range_s=(
                    float(self.cam_latency["live_delay_ms"][0]) * 1e-3,
                    float(self.cam_latency["live_delay_ms"][1]) * 1e-3),
                bps_delay_range_s=(
                    float(self.cam_latency["bps_delay_ms"][0]) * 1e-3,
                    float(self.cam_latency["bps_delay_ms"][1]) * 1e-3),
                jitter_s=float(self.cam_latency["jitter_ms"]) * 1e-3,
                control_dt=self._step_dt, substep_s=self.model.opt.timestep)
        self._cam_frame: CamFrame | None = None
        self._live_detected = False
        # Hold-last-pose marker state: the obs serves each tag's most recent
        # detection plus its age; a last-capture time of -inf means never seen
        # this episode (zero pose, age pinned at MARKER_AGE_CAP_S).
        self._held_marker_pos = np.zeros((N_MARKERS, 3))
        self._held_marker_rot = np.zeros((N_MARKERS, 3))
        self._marker_last_capture_t = np.full(N_MARKERS, -np.inf)
        # The live driver owns the shared hold/age/static gate. BPSObsState owns
        # the one current/held dense-surface block outside observation history.
        self._obj = ObjectChannelDriver()
        # Capture-time twin of the real ObjectSource static gate. It decides
        # whether a selected frame starts dense work; the delivery-time driver
        # above still owns the policy-facing live state and precise eligibility.
        self._capture_obj = ObjectChannelDriver()
        self._bps_state = BPSObsState()
        # Dense processing starts from a live frame only when that frame passes
        # the static/visibility gate. Its result arrives later on the BPS stream.
        self._bps_eligible_capture_times: set[float] = set()
        self._episode_start_t = 0.0
        # Privileged (always-fresh) mirror of the held state: every frame
        # overwrites it regardless of detection, so it serves exactly what a
        # marker_always_visible=true policy sees — same frames, same
        # noise/bias/latency, no dropout, no hold-last-pose and no static
        # gate (live = GT center + the frame's bias/noise). Read only by privileged_obs() (the distillation
        # teacher view, src/distill.py); never fed to this env's own policy.
        # Needs no reset beyond the fresh objects: _ingest_frame overwrites
        # all of it unconditionally, starting with the reset frame.
        self._held_marker_pos_priv = np.zeros((N_MARKERS, 3))
        self._held_marker_rot_priv = np.zeros((N_MARKERS, 3))
        self._marker_last_capture_t_priv = np.full(N_MARKERS, -np.inf)
        self._obj_state_priv = ObjectObsState()
        self._prev_qpos_obs = None
        self._last_encoder_obs = None

        self._parse_config(task_cfg)

        obs_high = np.full(self.obs_dim + self.priv_dim,
                           np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def _parse_config(self, cfg):
        """Override for task-specific config."""

    def _set_marker_render_colors(self, detected):
        """Tint each marker site green when detected this frame, red otherwise.
        Visual only (site rgba) — no effect on physics or observations."""
        set_marker_render_colors(self.model, self.marker_site_ids, detected)

    def _capture_camera_state(self, capture_object: bool = True,
                              gate_dense: bool = False,
                              capture_t: float | None = None):
        """CamState snapshot of the current MjData — recorded per substep so
        CameraSim can capture frames at any past instant of the tick.

        Marker-only frames skip all sponge geometry. Object frames batch live
        and dense samples into one visibility pass when the capture-time static
        gate can start a dense job. Direct callers get an unconditional object
        capture; scheduled callers pass ``gate_dense=True`` and ``capture_t``.
        """
        if gate_dense and capture_t is None:
            raise ValueError("capture_t is required when gate_dense is true")
        marker_pos, marker_rot = marker_world_poses(self.data, self.marker_site_ids)
        vis_frac = np.zeros(2)
        vis_centroid = np.zeros((2, 3))
        seen = np.zeros(2, dtype=bool)
        cube_center = self.data.geom_xpos[self.cube_geom_id].copy()
        if not capture_object:
            return CamState(
                marker_pos=marker_pos, marker_rot=marker_rot,
                marker_normal=marker_world_normals(
                    self.data, self.marker_site_ids),
                cube_center=cube_center,
                cube_vis_frac=vis_frac, cube_vis_centroid=vis_centroid,
                cube_seen=seen, live_detected=False, bps_measurement=None)

        dense_candidate = (
            not gate_dense
            or self.marker_always_visible
            or self._capture_obj.static_now()
        )
        dense_visible = dense_candidate and not \
            self._bps_generator.whole_view_lost(self.np_random)
        if dense_visible:
            points, normals = transform_box_surface_points_world(
                self.data, self.cube_geom_id, self.cube_half_extents,
                self._combined_surface_unit_points,
                self._combined_surface_unit_normals)
        else:
            points, normals = cube_surface_points_world(
                self.data, self.cube_geom_id, self.cube_half_extents)

        live_points = points[:self._live_surface_count]
        live_normals = normals[:self._live_surface_count]
        dense_masks = []
        for i, cam in enumerate(self.cube_cams):
            visible, _ = visible_surface_mask(
                self.model, self.data, cam, self.cube_body_id, points, normals)
            live_visible = visible[:self._live_surface_count]
            live_facing = int(np.count_nonzero(np.einsum(
                "ij,ij->i", live_normals, cam.pos - live_points) > 0.0))
            frac, centroid = visible_surface_summary(
                live_points, live_visible, live_facing)
            vis_frac[i] = frac
            if centroid is not None:
                vis_centroid[i] = centroid
                seen[i] = True
            if dense_visible:
                dense_masks.append(visible[self._live_surface_count:])

        if self.marker_always_visible:
            live_detected = True
        else:
            if self.marker_dropout is None:
                p_near = p_far = 0.0
            else:
                p_near = self.marker_dropout["near"]
                p_far = self.marker_dropout["far"]
            if not seen.all():
                live_prob = 1.0
            elif vis_frac.min() < VISIBLE_FRACTION_MIN:
                live_prob = p_near
            else:
                live_prob = p_far
            live_detected = bool(self.np_random.random() >= live_prob)

        live_gate = (cube_center if self.marker_always_visible
                     else vis_centroid.mean(axis=0))
        dense_eligible = True
        if gate_dense:
            if live_detected:
                self._capture_obj.ingest_live(
                    capture_t, live_gate, gate_point=live_gate)
            dense_eligible = (
                self.marker_always_visible
                or (live_detected and self._capture_obj.gate_open(vis_frac))
            )

        bps_capture = None
        if dense_visible and dense_eligible:
            dense_points = points[self._live_surface_count:]
            bps_capture = self._bps_generator.capture_visible(
                dense_points, tuple(dense_masks), self.np_random)
        return CamState(marker_pos=marker_pos, marker_rot=marker_rot,
                        marker_normal=marker_world_normals(self.data, self.marker_site_ids),
                        cube_center=cube_center,
                        cube_vis_frac=vis_frac, cube_vis_centroid=vis_centroid,
                        cube_seen=seen, live_detected=live_detected,
                        bps_measurement=(None if bps_capture is None
                                         else bps_capture.measurement))

    def _process_frame(self, state: CamState,
                       capture_object: bool = True) -> CamFrame:
        """One simulated detection of a captured world state: roll the per-frame
        dropout (geometric visibility + DR), apply bias and per-frame noise.
        Runs exactly once per frame, at capture time — a real frame is detected
        once, so consuming it twice re-reads the same values. Undetected
        markers keep garbage pose entries and an undetected live channel a
        garbage centroid; only the detected flags gate what _ingest_frame
        folds into the held obs state."""
        if self.marker_always_visible:
            detected = np.ones(N_MARKERS, dtype=bool)
        else:
            if self.marker_dropout is None:
                p_near = p_far = 0.0
            else:
                p_near = self.marker_dropout["near"]
                p_far = self.marker_dropout["far"]
            prob = marker_dropout_prob(state.marker_pos, state.marker_normal,
                                       self.tag_cam, p_near, p_far)
            detected = self.np_random.random(N_MARKERS) >= prob
        live_detected = bool(capture_object and state.live_detected)

        marker_pos = state.marker_pos.copy()
        marker_rot = state.marker_rot.copy()
        # Measured live centroid: the two views' visible-surface centroids
        # averaged — reproducing the real triangulated point's bias toward the
        # visible surface instead of pretending the live channel sees the true
        # center. The privileged live is the GT center under the same
        # bias/noise (what the marker_always_visible crutch serves).
        live_meas = (state.cube_vis_centroid.mean(axis=0)
                     if capture_object else np.zeros(3))
        live_priv = state.cube_center.copy()
        bps_measurement = state.bps_measurement if capture_object else None
        live_err = np.zeros(3)
        if self.obs_bias is not None:
            # _common_pos_bias hits the arm markers and both cube channels
            # (shared camera re-anchor error); the per-channel biases stay
            # independent.
            marker_pos = marker_pos + self._marker_pos_bias + self._common_pos_bias
            marker_rot = marker_rot + self._marker_rot_bias
            if capture_object:
                live_err = live_err + self._live_bias + self._common_pos_bias
            if bps_measurement is not None:
                bps_measurement = BPSMeasurement(
                    distances=bps_measurement.distances,
                    center_base=(bps_measurement.center_base + self._precise_bias
                                 + self._common_pos_bias),
                    valid_fraction=bps_measurement.valid_fraction,
                )
        if self.obs_noise is not None:
            rng = self.np_random
            px = self.obs_noise["tag_px_noise"]
            depth_factor = self.obs_noise["tag_depth_factor"]
            # Per-tag anisotropic position noise: small in the image plane, large
            # along each tag's own camera ray (solvePnP range from apparent size).
            # The ray uses the TRUE pose (state.*), so its depth axis is the real
            # line of sight rather than a circular function of the noise.
            for i in range(N_MARKERS):
                marker_pos[i] += anisotropic_pos_noise(
                    rng, state.marker_pos[i], self.tag_cam_pos,
                    self._marker_tag_sizes[i], self._focal_px, px, depth_factor)
            marker_rot = marker_rot + rng.normal(0, self.obs_noise["marker_rot_sigma"],
                                                 size=marker_rot.shape)
            if capture_object:
                live_err = live_err + rng.normal(
                    0, self.obs_noise["live_sigma"], size=3)
            if bps_measurement is not None:
                bps_measurement = BPSMeasurement(
                    distances=bps_measurement.distances,
                    center_base=(bps_measurement.center_base + rng.normal(
                        0, self.obs_noise["precise_sigma"], size=3)),
                    valid_fraction=bps_measurement.valid_fraction,
                )
        if self.marker_always_visible and capture_object:
            # Easy-mode crutch: the live channel reads the (noisy) true center,
            # sidestepping the visible-surface bias and any occlusion.
            live_meas = live_priv
        live = live_meas + live_err
        return CamFrame(marker_pos=marker_pos, marker_rot=marker_rot,
                        detected=detected, live=live,
                        # The static gate judges the PRE-noise measurement:
                        # the injected live_sigma covers in-motion/sync error,
                        # not the sub-mm jitter a real static track shows —
                        # gating on the noisy value would (wrongly) never pass
                        # under DR while the real gate passes routinely.
                        live_gate=live_meas,
                        live_priv=live_priv + live_err,
                        live_detected=live_detected,
                        vis_frac=state.cube_vis_frac.copy(),
                        bps_measurement=bps_measurement)

    def _ingest_marker_frame(self, capture_t: float, frame: CamFrame) -> None:
        """Publish the AprilTag result for one captured frame."""
        self._cam_frame = frame
        det = frame.detected
        self._held_marker_pos[det] = frame.marker_pos[det]
        self._held_marker_rot[det] = frame.marker_rot[det]
        self._marker_last_capture_t[det] = capture_t
        self._set_marker_render_colors(det)

        self._held_marker_pos_priv[:] = frame.marker_pos
        self._held_marker_rot_priv[:] = frame.marker_rot
        self._marker_last_capture_t_priv[:] = capture_t

    def _ingest_live_frame(self, capture_t: float, frame: CamFrame) -> None:
        """Publish the dual-SAM centroid and decide whether dense work starts."""
        eligible = False
        if frame.live_detected:
            self._obj.ingest_live(capture_t, frame.live, gate_point=frame.live_gate)
            eligible = self.marker_always_visible or self._obj.gate_open(frame.vis_frac)
        elif self.marker_always_visible:
            # Crutch with undetectable live geometry still serves noisy GT.
            self._obj.state.ingest_live(capture_t, frame.live)
            eligible = True
        if eligible:
            self._bps_eligible_capture_times.add(capture_t)

        self._live_detected = frame.live_detected
        # Live-channel indicator on the legacy tag site: green while both
        # cameras measure the object, red while the live channel is stale.
        # Visual only — never read by obs.
        self.model.site_rgba[self.cube_tag_site_id] = \
            MARKER_VISIBLE_RGBA if frame.live_detected else MARKER_HIDDEN_RGBA

        self._obj_state_priv.ingest_live(capture_t, frame.live_priv)

    def _ingest_bps_frame(self, capture_t: float, frame: CamFrame) -> None:
        """Publish dense stereo if its earlier live result opened the gate."""
        eligible = capture_t in self._bps_eligible_capture_times
        # Marker/live consume every camera frame, while the bounded dense
        # worker selects only a subset. Once a selected result arrives, older
        # eligibility records can never produce a BPS delivery.
        self._bps_eligible_capture_times = {
            eligible_t for eligible_t in self._bps_eligible_capture_times
            if eligible_t > capture_t
        }
        # Reset reconstructs already-in-flight frames from the static
        # pre-episode scene. Their live-stage gate ran before episode time, so
        # evaluate it against the seeded static history when the result lands.
        if capture_t <= self._episode_start_t and frame.live_detected:
            eligible = self._obj.gate_open(frame.vis_frac)
        if self.marker_always_visible or eligible:
            self._bps_state.ingest(capture_t, frame.bps_measurement)

    def _ingest_frame(self, capture_t, frame: CamFrame):
        """Synchronously publish all modalities; used by direct contract tests."""
        self._ingest_marker_frame(capture_t, frame)
        self._ingest_live_frame(capture_t, frame)
        self._ingest_bps_frame(capture_t, frame)

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3].copy()

    def _obs_extra(self, cube_pos):
        """Return task-specific obs dimensions appended after [qpos, qvel, markers, cube].

        Receives the held (possibly noisy/stale) LIVE centroid — the carrying
        phase needs "where is it now", not the static-refreshed center — so
        derived quantities stay consistent with what the agent sees.
        """
        raise NotImplementedError

    def _encoder_obs(self):
        """(qpos, qvel) for this control tick: qpos on the encoder path (fresh,
        biased+noisy), qvel its backward difference like
        real/rollout_common.ArmLoop. Advances _prev_qpos_obs and caches the
        result in _last_encoder_obs so a paired privileged_obs() reuses the
        identical read — the student and teacher views must differ only in the
        tag channels. Call exactly once per served policy observation."""
        qpos = self.data.qpos[self.joint_qposadr].copy()
        if self.obs_bias is not None:
            qpos = qpos + self._qpos_bias
        if self.obs_noise is not None:
            qpos = qpos + self.np_random.normal(0, self.obs_noise["qpos_sigma"],
                                                size=qpos.shape)
        # Real qvel is the backward difference of consecutive encoder reads over
        # the control tick, so the obs inherits its half-tick lag, smoothing, and
        # noise (sqrt(2)*qpos_sigma/dt) from qpos. Zero on the first obs after
        # reset — ArmLoop.boot does the same.
        if self._prev_qpos_obs is None:
            qvel = np.zeros_like(qpos)
        else:
            qvel = (qpos - self._prev_qpos_obs) / self._step_dt
        self._prev_qpos_obs = qpos
        self._last_encoder_obs = (qpos, qvel)
        return qpos, qvel

    def _tag_obs(self, held_pos, held_rot, marker_last_t):
        """(markers, marker_age) from a held-tag source: the arm markers' obs
        slice and their age channels. Shared by the student view
        (_serve_obs, hold-last-pose state) and the privileged teacher view
        (privileged_obs, always-fresh mirror)."""
        if self.marker_include_rot:
            # hstack interleaves per marker: [pos_finger, rot_finger, pos_wrist, rot_wrist]
            markers = np.hstack([held_pos, held_rot]).flatten()
        else:
            # positions only: [pos_finger, pos_wrist]
            markers = held_pos.flatten()
        # Clip below at 0: the frame schedule's float slack (CameraSim._EPS)
        # can put a capture an epsilon after the obs instant.
        marker_age = np.clip(self.data.time - marker_last_t, 0.0, MARKER_AGE_CAP_S)
        return markers, marker_age

    def _priv_tail(self):
        """Privileged tail appended after the actor block (asymmetric critic,
        priv_dim_for): ground truth plus the episode's sampled DR latents. Only
        the value function reads these dims — the actor slices them off before
        its first layer (src/networks.TakeFirst) — so the "policy sees no GT"
        contract holds structurally, and the critic is discarded at deployment
        (real rollouts pad these dims with zeros that are never read). Latents
        whose DR knob is off stay at their zero init — truthfully "no error".
        marker_dropout near/far are run constants, not per-episode samples, so
        they carry no information and are not included."""
        return self._build_priv_tail(self._live_bias, self._precise_bias)

    def _build_priv_tail(self, live_bias, precise_bias):
        """Build the shared tail layout with the supplied object-bias slots."""
        cube_quat = self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7]
        cube_rot = np.empty(3)
        mujoco.mju_quat2Vel(cube_rot, cube_quat, 1.0)
        # Free-joint qvel: linear (world frame) + angular (body frame).
        cube_vel = self.data.qvel[self.cube_dofadr:self.cube_dofadr + 6]
        fixed_contact, moving_contact = self._jaw_contact_flags()
        parts = [self._get_cube_pos(), cube_rot, cube_vel,
                 [float(fixed_contact), float(moving_contact)],
                 self._qpos_bias, self._marker_pos_bias.flatten()]
        if self.marker_include_rot:
            parts.append(self._marker_rot_bias.flatten())
        parts.extend([live_bias, precise_bias, self._common_pos_bias,
                      [self._camera.pipeline_delay_s], self.cube_half_extents])
        return np.concatenate(parts)

    def _compute_state_obs(self):
        """Build one actor-state frame (the part repeated through history).

        qpos/qvel take the encoder path (fresh; qvel differenced like
        real/rollout_common.ArmLoop); markers serve the held per-tag
        detections with their ages; the live object channel uses the shared
        hold/age state. The final derived vector is the current kinematic EE
        position minus that same held live centroid, so it remains deployable
        and inherits object-channel staleness/noise. The BPS block and
        privileged tail are separate."""
        qpos, qvel = self._encoder_obs()
        markers, marker_age = self._tag_obs(
            self._held_marker_pos, self._held_marker_rot,
            self._marker_last_capture_t)
        live, live_age = self._obj.serve(self.data.time)
        return np.concatenate([
            qpos, qvel, markers, marker_age,
            live, [live_age], self._obs_extra(live),
            self._prev_actions.flatten(),
            ee_object_delta(self._get_ee_pos(), live),
        ]).astype(np.float32)

    def _serve_obs(self, reset: bool):
        """Return ``[state taps | current BPS | privileged tail]``.

        On reset the boot state seeds every history slot; BPS and the tail are
        each served once at the current instant.
        """
        state = self._compute_state_obs()
        tapped = self._history.reset(state) if reset else self._history.push(state)
        return np.concatenate([
            tapped,
            self._bps_state.serve(self.data.time).flat(),
            self._priv_tail(),
        ]).astype(np.float32)

    def privileged_obs(self):
        """The observation a marker_always_visible=true policy would see on the
        current state: every tag pose fresh (the always-updated privileged
        mirror, no hold-last-pose and no dropout), sharing the paired
        policy observation's encoder read, task extras, and prev-actions so the two
        views differ only in the tag channels. The distillation teacher view
        (src/distill.py). Call once, immediately after the tick's policy observation."""
        assert self._last_encoder_obs is not None, \
            "privileged_obs() needs a policy observation earlier this tick"
        qpos, qvel = self._last_encoder_obs
        markers, marker_age = self._tag_obs(
            self._held_marker_pos_priv, self._held_marker_rot_priv,
            self._marker_last_capture_t_priv)
        live, live_age = self._obj_state_priv.serve(self.data.time)
        return np.concatenate([qpos, qvel, markers, marker_age,
                               live, [live_age],
                               self._obs_extra(live),
                               self._prev_actions.flatten(),
                               ee_object_delta(self._get_ee_pos(), live),
                               self._bps_state.serve(self.data.time).flat(),
                               self._priv_tail()]).astype(np.float32)

    def legacy_tag_obs(self):
        """Fresh single-frame observation for tag-policy distillation only.

        This is the one supported migration bridge from the last sponge-tag
        teacher. The deployed BPS actor never calls it, and the sponge tag
        remains evaluation-only everywhere else.
        """
        assert self._last_encoder_obs is not None, \
            "legacy_tag_obs() needs a policy observation earlier this tick"
        qpos, qvel = self._last_encoder_obs
        markers, marker_age = self._tag_obs(
            self._held_marker_pos_priv, self._held_marker_rot_priv,
            self._marker_last_capture_t_priv)
        (cube_tag_pos,), (cube_tag_rot,) = marker_world_poses(
            self.data, [self.cube_tag_site_id])
        cube_age = float(np.clip(
            self.data.time - np.max(self._marker_last_capture_t_priv),
            0.0, MARKER_AGE_CAP_S))
        return np.concatenate([
            qpos, qvel, markers, marker_age,
            cube_tag_pos, cube_tag_rot, [cube_age],
            self._obs_extra(cube_tag_pos), self._prev_actions.flatten(),
            self._legacy_tag_priv_tail(),
        ]).astype(np.float32)

    def _legacy_tag_priv_tail(self):
        """Exact critic-tail semantics of the final tag-observation teacher.

        The retired cube-tag position/rotation bias slots are zero because the
        migration bridge reads the evaluation tag directly. Keeping those slots
        distinct from the new live/BPS center biases prevents an old teacher's
        critic from receiving unrelated latent values.
        """
        return self._build_priv_tail(np.zeros(3), np.zeros(3))

    def _jaw_contact_flags(self):
        """(fixed, moving): whether each gripper jaw currently touches the cube."""
        fixed_contact = False
        moving_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.cube_geom_id:
                other = g2
            elif g2 == self.cube_geom_id:
                other = g1
            else:
                continue
            if other in self.fixed_jaw_geom_ids:
                fixed_contact = True
            if other in self.moving_jaw_geom_ids:
                moving_contact = True
        return fixed_contact, moving_contact

    def _n_jaw_contacts(self):
        """Number of distinct gripper jaws (0, 1, or 2) currently touching the cube."""
        fixed_contact, moving_contact = self._jaw_contact_flags()
        return int(fixed_contact) + int(moving_contact)

    def _has_gripper_contact(self):
        """Check if cube_geom is in contact with both jaws simultaneously."""
        return self._n_jaw_contacts() == 2

    def _arm_cube_contact_force(self):
        """Sum of normal contact force magnitudes (N) between any arm geom and the cube."""
        total = 0.0
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.cube_geom_id:
                other = c.geom2
            elif c.geom2 == self.cube_geom_id:
                other = c.geom1
            else:
                continue
            if other not in self.arm_geom_ids:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            total += abs(wrench[0])
        return total

    def _cube_angular_speed(self):
        """Magnitude of the cube's angular velocity (rad/s) — how fast it tips/rolls."""
        w = self.data.qvel[self.cube_dofadr + 3:self.cube_dofadr + 6]
        return float(np.linalg.norm(w))

    def _detect_grasp(self):
        """Grasp = cube close to EE + gripper closing + contact."""
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        dist = np.linalg.norm(ee_pos - cube_pos)
        gripper_val = self.data.qpos[self.joint_ids[self.gripper_idx]]
        return (dist < 0.05
                and gripper_val < 0.3
                and self._has_gripper_contact())

    def _gripper_closedness(self):
        """Fraction the gripper is closed, in [0, 1]. 0 = fully open, 1 = fully closed.
        Normalized against the gripper joint's range from the model (single source)."""
        gripper_val = self.data.qpos[self.joint_ids[self.gripper_idx]]
        lo = self.joint_low[self.gripper_idx]
        hi = self.joint_high[self.gripper_idx]
        return float(np.clip((hi - gripper_val) / (hi - lo), 0.0, 1.0))

    def _sample_cube_pos(self):
        """Return random cube xy position. Override for rejection sampling."""
        return self.np_random.uniform(self.cube_low, self.cube_high)

    def _cube_arm_contact(self):
        """Any contact between the cube and an arm/gripper geom."""
        arm_ids = self.arm_geom_ids | self.gripper_geom_ids
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.cube_geom_id and c.geom2 in arm_ids:
                return True
            if c.geom2 == self.cube_geom_id and c.geom1 in arm_ids:
                return True
        return False

    def _sample_collision_free_arm_pose(self):
        """Sample random arm position, rejecting arm-environment collisions."""
        attempt = 0
        while True:
            joint_pos = self.np_random.uniform(self.joint_low, self.joint_high)
            self._write_arm_pose(joint_pos)
            if not self._has_arm_collision():
                return joint_pos
            attempt += 1
            if attempt % 10 == 0:
                print(f"WARNING: {attempt} arm position samples rejected (collision)")

    def _sample_visible_cube_spawn(self, max_attempts):
        """Sample the cube spawn pose, rejecting any the two cameras cannot
        comfortably see — matching the real protocol of placing the sponge in
        both cameras' clear view. Requires the arm (and any task scenery)
        already placed: a candidate must reach at least VISIBLE_FRACTION_MIN
        visible surface in BOTH views (so the live channel detects immediately
        and the boot precise refresh can fire) and not touch the arm. Writes
        the accepted pose into qpos (data left forwarded) and returns
        (cube_pos, cube_quat). Returns None when no candidate passes within
        max_attempts."""
        for _ in range(max_attempts):
            cube_xy = self._sample_cube_pos()
            cube_quat, self.cube_rest_half_z = sample_cube_orientation(
                self.np_random, self.cube_half_extents,
                smallest_face_only=self.cube_smallest_face_only,
                no_flat_spawns=self.cube_no_flat_spawns)
            cube_pos = np.array([cube_xy[0], cube_xy[1], self.cube_rest_half_z])
            self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
            self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = cube_quat
            mujoco.mj_forward(self.model, self.data)
            points, normals = cube_surface_points_world(
                self.data, self.cube_geom_id, self.cube_half_extents)
            if any(cube_visible_surface(self.model, self.data, cam,
                                        self.cube_body_id, points, normals)[0]
                   < VISIBLE_FRACTION_MIN for cam in self.cube_cams):
                continue
            if self._cube_arm_contact():
                continue
            return cube_pos, cube_quat
        return None

    def _set_cube_half_extents(self, half):
        """Resize the sponge box: write the geom half-extents and slide the
        cube_tag site back onto the (now moved) top face — the tag is glued to
        the center of the largest face, so its z offset equals the z half-extent.
        Keeps self.cube_half_extents in sync for the orientation/rest-height
        sampling that reads it."""
        self.model.geom_size[self.cube_geom_id] = half
        self.model.site_pos[self.cube_tag_site_id][2] = half[2]
        self.cube_half_extents = np.asarray(half, dtype=np.float64).copy()

    def _sample_cube_half_extents(self):
        """Per-episode sponge half-extents (DR resize): each axis jittered by a
        uniform +/- cube_size_half_jitter around the nominal box, reject-sampling
        any draw that breaks the strict hx > hy > hz face ordering that
        sample_cube_orientation and the tag-on-largest-face convention require."""
        nominal = self.cube_nominal_half_extents
        if self.cube_size_half_jitter == 0.0:
            return nominal.copy()
        j = self.cube_size_half_jitter
        while True:
            half = nominal + self.np_random.uniform(-j, j, size=3)
            if half[0] > half[1] > half[2]:
                return half

    def _randomize_scene(self):
        """Hook for task scenery randomization (e.g. ring height). Runs before
        the arm and cube are placed, so their rejection sampling (collision,
        tag visibility) sees the final scene geometry."""

    def _on_reset(self, cube_pos):
        """Hook for task-specific reset state. Called after common reset."""

    def reset(self, *, seed=None, options=None):
        """options may pin the initial state instead of sampling it (used by
        sysid/replay_rollout.py to restart a sim episode from a recorded real
        pose): "qpos" (6,) sets the arm joints, "cube_pos" (3,) + "cube_quat"
        (4, wxyz — must come together) set the cube. Pinned states skip the
        corresponding sampling (including the cube's tag-visible-spawn
        rejection) but keep every other reset step identical."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._prev_qpos_obs = None
        self._last_encoder_obs = None
        self._prev_actions[:] = 0.0

        options = options or {}
        assert ("cube_pos" in options) == ("cube_quat" in options), \
            "cube_pos and cube_quat must be overridden together"

        cube_pinned = "cube_pos" in options
        qpos_pinned = "qpos" in options

        # Sponge-size DR: resample the box extents before any orientation/spawn
        # sampling reads self.cube_half_extents. Skipped when the cube pose is
        # pinned (e.g. sysid replay), which must use the modeled nominal sponge.
        self._set_cube_half_extents(
            self.cube_nominal_half_extents if cube_pinned
            else self._sample_cube_half_extents())

        if cube_pinned:
            self._randomize_scene()
            cube_pos = np.asarray(options["cube_pos"], dtype=np.float64)
            cube_quat = np.asarray(options["cube_quat"], dtype=np.float64)
            # Spawn height = resting half height (cube spawns at rest).
            self.cube_rest_half_z = float(cube_pos[2])
            self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
            self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = cube_quat
            if qpos_pinned:
                joint_pos = np.asarray(options["qpos"], dtype=np.float64)
                self._write_arm_pose(joint_pos)
            else:
                joint_pos = self._sample_collision_free_arm_pose()
        elif qpos_pinned:
            self._randomize_scene()
            # Park the cube out of reach while setting the pinned arm pose.
            self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = (1.0, 1.0, 0.05)
            joint_pos = np.asarray(options["qpos"], dtype=np.float64)
            self._write_arm_pose(joint_pos)
            sampled = self._sample_visible_cube_spawn(max_attempts=100)
            if sampled is None:
                raise AssertionError(
                    "no camera-visible cube spawn found in 100 attempts; "
                    "check the spawn box against the tag camera")
            cube_pos, cube_quat = sampled
        else:
            # If the sampled arm fully blocks camera-tag visibility, restart the
            # reset sampling from scratch with a new scene/arm realization.
            for reset_attempt in range(100):
                self._randomize_scene()
                self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = (1.0, 1.0, 0.05)
                joint_pos = self._sample_collision_free_arm_pose()
                sampled = self._sample_visible_cube_spawn(max_attempts=100)
                if sampled is not None:
                    cube_pos, cube_quat = sampled
                    break
                if (reset_attempt + 1) % 10 == 0:
                    print(
                        "WARNING: "
                        f"{reset_attempt + 1} reset resamples rejected "
                        "(no visible cube spawn for sampled arm pose)"
                    )
            else:
                raise AssertionError(
                    "no camera-visible cube spawn found after 200 reset resamples "
                    "(100 cube attempts each); check the spawn box and tag camera")

        self._on_reset(cube_pos)
        self._servo_profile.reset(joint_pos)
        self._ctrl_target = joint_pos.copy()
        self.step_count = 0
        self._max_cube_height = cube_pos[2]
        self._grasp_steps = 0
        self._floor_contact_steps = 0
        self._cube_drag_steps = 0
        self._markers_hidden_total = 0
        self._live_hidden_total = 0
        self._precise_age_sum = 0.0
        self._prev_cube_xy = cube_pos[:2].copy()

        # Sample obs biases AFTER physics randomization so toggling obs_bias does
        # not change the distribution of cube/arm initial states for a given seed.
        if self.obs_bias is not None:
            rng = self.np_random
            self._qpos_bias = rng.normal(0, self.obs_bias["qpos_sigma"], size=self.n_joints)
            self._marker_pos_bias = rng.normal(0, self.obs_bias["marker_pos_sigma"],
                                               size=(N_MARKERS, 3))
            self._marker_rot_bias = rng.normal(0, self.obs_bias["marker_rot_sigma"],
                                               size=(N_MARKERS, 3))
            self._live_bias = rng.normal(0, self.obs_bias["live_sigma"], size=3)
            self._precise_bias = rng.normal(0, self.obs_bias["precise_sigma"], size=3)
            # Shared across all camera-derived positions (see _common_pos_bias
            # init): one draw the whole episode, added to every marker and both
            # cube channels in _process_frame.
            self._common_pos_bias = rng.normal(0, self.obs_bias["marker_common_sigma"], size=3)

        self._held_marker_pos[:] = 0.0
        self._held_marker_rot[:] = 0.0
        self._marker_last_capture_t[:] = -np.inf
        self._obj = ObjectChannelDriver()
        self._capture_obj = ObjectChannelDriver()
        self._bps_state = BPSObsState()
        self._obj_state_priv = ObjectObsState()
        self._bps_eligible_capture_times.clear()
        self._episode_start_t = self.data.time
        reset_state = self._capture_camera_state()
        if reset_state.live_detected:
            capture_gate = (reset_state.cube_center
                            if self.marker_always_visible
                            else reset_state.cube_vis_centroid.mean(axis=0))
            self._capture_obj.seed_static(
                self.data.time - STATIC_DWELL_S, capture_gate)
            self._capture_obj.ingest_live(
                self.data.time, capture_gate, gate_point=capture_gate)
        deliveries = self._camera.reset(
            self.np_random, self.data.time, reset_state, reset_state,
            self._process_frame)
        marker_capture_t, marker_frame = deliveries.marker[0]
        live_capture_t, live_frame = deliveries.live[0]
        bps_capture_t, bps_frame = deliveries.bps[0]
        self._ingest_marker_frame(marker_capture_t, marker_frame)
        # The pre-episode world is static (the CameraSim reset contract), so
        # the boot detection proves a full static dwell: seed the static-gate
        # history one dwell back with the same measurement, letting the boot
        # frame refresh the precise channel immediately — the real rollout's
        # settle-before-start warmup does the same thing physically.
        if live_frame.live_detected:
            self._obj.seed_static(
                live_capture_t - STATIC_DWELL_S, live_frame.live_gate)
        self._ingest_live_frame(live_capture_t, live_frame)
        self._ingest_bps_frame(bps_capture_t, bps_frame)

        return self._serve_obs(reset=True), {}

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        """Return (reward, terminated, info) for the current step."""
        raise NotImplementedError

    def _on_episode_end(self, info):
        """Hook to add task-specific info at episode end. Mutate info dict."""

    def _on_substep(self):
        # Snapshotting resolves occlusion against both cameras (raycasts, the
        # dominant cost of a substep), so only do it for the substeps the frame
        # schedule will actually capture from.
        if self._camera.needs_state(self.data.time, self.np_random):
            self._camera.record(
                self.data.time,
                self._capture_camera_state(
                    capture_object=self._camera.record_object,
                    gate_dense=self._camera.record_object,
                    capture_t=self._camera.record_capture_t))

    def step(self, action):
        self.step_count += 1
        self._apply_action(action)

        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)
        grasped = self._detect_grasp()
        if grasped:
            self._grasp_steps += 1
        floor_contact = self._has_floor_contact() if self.floor_contact_penalty else False
        if floor_contact:
            self._floor_contact_steps += 1
        self._max_cube_height = max(self._max_cube_height, cube_pos[2])

        cube_xy_speed = np.linalg.norm(cube_pos[:2] - self._prev_cube_xy) / self._step_dt
        self._prev_cube_xy = cube_pos[:2].copy()
        if cube_pos[2] < self.cube_rest_half_z + self.DRAG_HEIGHT_TOL and cube_xy_speed > self.DRAG_SPEED_THRESH:
            self._cube_drag_steps += 1

        reward, terminated, info = self._compute_step(
            ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact,
        )
        info["task_name"] = self.TASK_NAME

        # Advance every camera-derived stream to this observation instant.
        # Marker/live reuse every capture/noise sample; BPS receives the
        # separately bounded dense-worker subset after its longer delay.
        deliveries = self._camera.observe(self.data.time, self._process_frame)
        for capture_t, frame in deliveries.marker:
            self._ingest_marker_frame(capture_t, frame)
        for capture_t, frame in deliveries.live:
            self._ingest_live_frame(capture_t, frame)
        for capture_t, frame in deliveries.bps:
            self._ingest_bps_frame(capture_t, frame)
        self._markers_hidden_total += N_MARKERS - int(self._cam_frame.detected.sum())
        self._live_hidden_total += not self._live_detected
        self._precise_age_sum += self._bps_state.serve(self.data.time).age_s

        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            info["max_cube_height"] = self._max_cube_height
            # Grasp is the rung between "approached the cube" and "lifted it":
            # while success_rate is still zero it is the only signal that says
            # whether the policy is on the path at all.
            info["ever_grasped"] = float(self._grasp_steps > 0)
            info["grasp_ratio"] = self._grasp_steps / self.step_count
            if self.floor_contact_penalty:
                info["floor_contact_ratio"] = self._floor_contact_steps / self.step_count
            info["cube_drag_ratio"] = self._cube_drag_steps / self.step_count
            # Fraction of marker-steps hidden from the camera (0 = always visible).
            info["marker_hidden_ratio"] = \
                self._markers_hidden_total / (self.step_count * N_MARKERS)
            # Fraction of steps the live channel had no both-view measurement,
            # and the mean served precise-channel age — the two health signals
            # of the live-plus-BPS cube observation.
            info["live_hidden_ratio"] = self._live_hidden_total / self.step_count
            info["precise_age_mean"] = self._precise_age_sum / self.step_count
            self._on_episode_end(info)

        if self.render_mode == "human":
            self._render_human()

        return self._serve_obs(reset=False), float(reward), terminated, truncated, info
