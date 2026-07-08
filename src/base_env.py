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
from src.marker_noise import anisotropic_pos_noise, load_focal_px
from src.servo_profile import ServoProfile
from src.units import action_to_target
from real.marker_spec import ARM_TAG_TO_SITE, CUBE_TAG_ID, TAG_SIZE_MM


def obs_dim_for(prev_actions_n: int, marker_include_rot: bool = False) -> int:
    """OBS_DIM = qpos(6) + qvel(6) + markers(2*M) + marker_age(2) + cube_tag(6)
    + cube_age(1) + extra(4) + prev_actions(N*6).

    Each marker contributes its world xyz (M=3); when marker_include_rot is true it
    also contributes a world rotation vector (axis-angle, +3 dims, M=6) — the same
    quantities the camera pipeline recovers for the physical AprilTags (real/pose.py
    rvec/tvec, camera->world mapped). marker_age is each tag's seconds since the
    capture of the pose it is serving (capped at MARKER_AGE_CAP_S): pipeline
    latency when freshly detected, growing while the tag is undetected and its
    last pose is held.

    The cube is observed the same way: cube_tag is the world pose (xyz +
    axis-angle rotation vector, always both) of the AprilTag on the sponge's
    largest face — the raw tag pose, never inferred back to a sponge center —
    and cube_age is its own age channel under the identical hold-last-pose
    convention.
    """
    marker_dim = 6 if marker_include_rot else 3
    # 12 = qpos(6) + qvel(6); 11 = cube_tag(3+3) + cube_age(1) + extra(4)
    return 12 + N_MARKERS * marker_dim + N_MARKERS + 11 + prev_actions_n * 6


# AprilTags glued to the arm (ids per real/marker_spec.py ROLES): "finger" on
# the bottom face of the fixed jaw, "wrist" on the right face of the wrist.
# Sites of the same names in so101.xml mirror their placement.
MARKER_SITE_NAMES = ["marker_finger", "marker_wrist"]
N_MARKERS = len(MARKER_SITE_NAMES)

# AprilTag id 1 on the sponge's largest face (real/marker_spec.py ROLES);
# the site of the same name in the scene XMLs mirrors it.
CUBE_TAG_SITE_NAME = "cube_tag"

# Undetected tags keep their last measured pose in the obs while their age
# channel grows, capped here; a tag never detected this episode reads an
# all-zero pose with age pinned at the cap. Shared with the real pipeline
# (real/marker_obs.py) — the identical convention on both sides.
MARKER_AGE_CAP_S = 1.0

# Fixed camera in so101.xml standing in for the physical webcam that watches
# the tags. A tag counts as visible when the angle between its outward normal
# (+z of the site frame) and the tag->camera ray is under this — past that the
# AprilTag detector loses the grazing view. Deliberately a plane-angle check
# only: no FOV or occlusion test. NEAR marks a softer band: from NEAR up to MAX
# the view grazes and the detector flakes, so the DR dropout (marker_dropout_prob)
# drops those tags more often than tags comfortably facing the camera.
TAG_CAM_NAME = "tag_cam"
MARKER_VIS_MAX_ANGLE_DEG = 70.0
MARKER_VIS_NEAR_ANGLE_DEG = 65.0
_MARKER_VIS_COS_MIN = np.cos(np.radians(MARKER_VIS_MAX_ANGLE_DEG))
_MARKER_VIS_COS_NEAR = np.cos(np.radians(MARKER_VIS_NEAR_ANGLE_DEG))
# Field-of-view proxy: the plane-angle check above models grazing, not the camera
# frame bounds. On the real rig the webcam clips tags near this height off the
# ground, so a tag whose world z exceeds it has left the top of the image and
# counts as undetected regardless of facing angle.
MARKER_VIS_MAX_HEIGHT_M = 0.23

# Render-only tints for the marker sites: green when the tag is detected this
# frame, red when not (over-angle or dropped). Visual only — never read by obs.
MARKER_VISIBLE_RGBA = np.array([0.1, 0.8, 0.1, 1.0])
MARKER_HIDDEN_RGBA = np.array([0.9, 0.1, 0.1, 1.0])


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


def marker_world_poses(data, site_ids):
    """World poses of the marker sites: (pos (N,3), rot (N,3)).

    rot is an axis-angle rotation vector, matching the Rodrigues-vector
    convention of cv2.solvePnP in real/pose.py.
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


def tag_cam_world_pos(model, data):
    """World position of TAG_CAM_NAME. The camera hangs off a fixed mount body,
    so cam_xpos (filled by forward kinematics) is its constant world position;
    model.cam_pos is body-relative and would be wrong. Runs mj_kinematics so the
    caller need not have forwarded data first (the mount is qpos-independent)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, TAG_CAM_NAME)
    assert cam_id >= 0, f"Camera '{TAG_CAM_NAME}' not found in model"
    # cam_xpos is filled by mj_camlight, which needs body frames from mj_kinematics.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    return data.cam_xpos[cam_id].copy()


def markers_visible(data, site_ids, cam_pos):
    """Per-marker geometric visibility from cam_pos as a bool array: plane angle
    under MARKER_VIS_MAX_ANGLE_DEG *and* world height under MARKER_VIS_MAX_HEIGHT_M
    (see TAG_CAM_NAME doc). No stochastic dropout — used by tests and the real-arm
    twin rollout."""
    pos, _ = marker_world_poses(data, site_ids)
    normals = marker_world_normals(data, site_ids)
    return marker_dropout_prob(pos, normals, cam_pos, p_near=0.0, p_far=0.0) < 1.0


def marker_dropout_prob(pos, normals, cam_pos, p_near, p_far):
    """Per-marker probability the tag is undetected this frame (its held pose
    goes stale), from the tags' world positions (N,3) and outward normals (N,3).

    Above MARKER_VIS_MAX_HEIGHT_M (out the top of frame) or past
    MARKER_VIS_MAX_ANGLE_DEG the camera can't see the tag at all -> 1.0.
    In the near-boundary band [NEAR, MAX] the grazing view is flaky -> p_near.
    Comfortably facing the camera (angle < NEAR) -> p_far (rare detector miss).
    """
    prob = np.empty(len(pos))
    for i in range(len(pos)):
        to_cam = cam_pos - pos[i]
        cos = (normals[i] @ to_cam) / np.linalg.norm(to_cam)
        if pos[i][2] > MARKER_VIS_MAX_HEIGHT_M or cos < _MARKER_VIS_COS_MIN:
            prob[i] = 1.0
        elif cos < _MARKER_VIS_COS_NEAR:
            prob[i] = p_near
        else:
            prob[i] = p_far
    return prob


# The camera's visible stand-in geoms (tag_cam_body/lens in so101.xml) sit
# within ~4 cm of the camera position, so the occlusion ray stops this far
# short of the camera to keep them from self-occluding every tag. Nothing else
# in the scene lives that close to the camera.
_CAM_GEOM_CLEARANCE_M = 0.06


def cube_tag_sample_points(data, site_id, half_extents):
    """The tag centre plus its four square corners, in world coordinates.

    Corners are the site box's ±x/±y half-extents mapped through the site's
    world rotation — the printed AprilTag's quad, the geometry the detector must
    see whole. z (tag thickness) is ignored: corners lie in the tag plane.
    """
    xmat = data.site_xmat[site_id].reshape(3, 3)
    center = data.site_xpos[site_id]
    hx, hy = half_extents[0], half_extents[1]
    pts = [center]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            pts.append(center + sx * hx * xmat[:, 0] + sy * hy * xmat[:, 1])
    return pts


def cube_tag_occluded(model, data, site_id, cam_pos, cube_body_id):
    """True when a geom blocks the cube tag's view of the camera at its centre
    or any of its four corners.

    Casts an mj_ray from each sample point (see cube_tag_sample_points) toward
    the camera against all geoms (static included — ring walls and floor block;
    so do the arm's visual meshes, the shapes the camera actually sees). The
    cube's own body is excluded so a tag facing away stays the angle check's
    job, not a self-occlusion. Any blocked ray hides the tag: a real AprilTag
    decode fails when part of its quad is occluded, not just its centre.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    for pt in cube_tag_sample_points(data, site_id, model.site_size[site_id]):
        vec = cam_pos - pt
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(model, data, pt.astype(np.float64), vec / dist,
                            None, 1, cube_body_id, geomid)
        if 0.0 <= hit < dist - _CAM_GEOM_CLEARANCE_M:
            return True
    return False


def cube_tag_visible(model, data, site_id, cam_pos, cube_body_id):
    """Geometric visibility of the cube tag from cam_pos: the markers_visible
    checks (plane angle under MARKER_VIS_MAX_ANGLE_DEG, height under
    MARKER_VIS_MAX_HEIGHT_M) plus an unobstructed ray to the camera. No
    stochastic dropout — used by tests and the real-arm rollout's FK branch."""
    pos, _ = marker_world_poses(data, [site_id])
    normal = data.site_xmat[site_id].reshape(3, 3)[:, 2]
    if marker_dropout_prob(pos, normal[None], cam_pos, p_near=0.0, p_far=0.0)[0] >= 1.0:
        return False
    return not cube_tag_occluded(model, data, site_id, cam_pos, cube_body_id)


class CamState(NamedTuple):
    """World-state snapshot recorded per physics substep for CameraSim: what a
    camera frame captured at that instant would see. Occlusion is resolved here,
    at capture time, where live MjData exists — the frame may be consumed ticks
    later, when the world has moved on."""
    marker_pos: np.ndarray     # (N_MARKERS, 3)
    marker_rot: np.ndarray     # (N_MARKERS, 3) axis-angle
    marker_normal: np.ndarray  # (N_MARKERS, 3) outward tag normals
    cube_tag_pos: np.ndarray   # (3,)
    cube_tag_rot: np.ndarray   # (3,) axis-angle
    cube_tag_normal: np.ndarray  # (3,) outward tag normal
    cube_tag_occluded: bool


class CamFrame(NamedTuple):
    """One processed detection: dropout, bias, and noise frozen at capture time.
    marker_pos/rot entries are only meaningful where detected — _ingest_frame
    folds the detected ones into the held per-tag state the obs serves."""
    marker_pos: np.ndarray  # (N_MARKERS, 3)
    marker_rot: np.ndarray  # (N_MARKERS, 3)
    detected: np.ndarray    # (N_MARKERS,) bool
    cube_tag_pos: np.ndarray  # (3,)
    cube_tag_rot: np.ndarray  # (3,)
    cube_detected: bool


def sample_cube_orientation(rng, half_extents, smallest_face_only=False):
    """Spawn orientation for the sponge box: standing on a non-largest face.

    half_extents (hx, hy, hz) must be strictly ordered hx > hy > hz (the
    6 x 4 x 2.5 cm box). The box stands either on its hy*hz face (x-axis up,
    2*hx tall) or its hx*hz face (y-axis up, 2*hy tall), never on the largest
    hx*hy face, with a uniform random yaw. Returns (quat wxyz, rest_half_z).

    smallest_face_only (simplified-starts curriculum crutch): always stand on
    the smallest hy*hz face (x-axis vertical) so the box presents the same
    upright, easy-to-grasp pose every episode. See conf/config.yaml.
    """
    hx, hy, hz = half_extents
    assert hx > hy > hz, f"expected strictly ordered half extents, got {half_extents}"
    s = np.sqrt(0.5)
    if smallest_face_only or rng.uniform() < 0.5:
        tilt = np.array([s, 0.0, s, 0.0])  # 90 deg about y: x-axis vertical
        rest_half_z = hx
    else:
        tilt = np.array([s, s, 0.0, 0.0])  # 90 deg about x: y-axis vertical
        rest_half_z = hy
    yaw = rng.uniform(-np.pi, np.pi)
    yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, yaw_quat, tilt)
    return quat, float(rest_half_z)

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
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
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
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
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
    """Base class for the cube-manipulation tasks: adds the cube + AprilTag
    camera observation pipeline (markers, cube tag, hold-last-pose ages, DR
    noise/bias/latency/dropout) and the cube-centric reset/step skeleton on
    top of SO101ArmEnv."""

    TASK_ID: float  # 0.0 = lift, 1.0 = pickplace

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1, xml_path=None,
                 cfg: RuntimeEnvConfig | None = None):
        assert cfg is not None, "RuntimeEnvConfig is required"
        super().__init__(render_mode=render_mode, slow_factor=slow_factor,
                         xml_path=xml_path, prev_actions_n=cfg.prev_actions_n,
                         env_cfg=env_cfg)
        # dict with keys qpos_sigma, marker_rot_sigma, tag_px_noise,
        # cube_px_noise, tag_depth_factor; or None. Marker/cube position noise is
        # anisotropic in the camera frame (src/marker_noise.py) instead of a
        # single sigma: tag_px_noise (arm markers) / cube_px_noise (cube tag)
        # with tag_depth_factor derive the depth-vs-lateral split. The
        # camera re-anchor's common-mode error is per-episode only
        # (obs_bias.marker_common_sigma): the real pipeline EMAs the static
        # camera pose (real/marker_obs.py), leaving no meaningful per-frame
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
        # Camera latency model (sim2real): dict with keys frame_ms, delay_ms
        # [lo, hi], jitter_ms (see conf/dr/full.yaml), or None for a synchronous
        # zero-latency camera. Stored raw here; the CameraSim is built below
        # once the control period is known.
        self.cam_latency = cfg.cam_latency
        # dict with keys qpos_sigma, marker_pos_sigma, marker_rot_sigma,
        # cube_sigma, marker_common_sigma; or None. The per-tag marker biases are
        # sampled independently (each tag's own glue/pose-estimate error); the cube
        # tag's pos bias uses cube_sigma and its rot bias reuses marker_rot_sigma
        # (same AprilTag pipeline). marker_common_sigma adds a separate shared shift
        # applied to every tag (the camera re-anchor / table calibration offset,
        # correlated across tags) — see _common_pos_bias.
        self.obs_bias = cfg.obs_bias
        self._qpos_bias = np.zeros(len(JOINT_NAMES))
        self._marker_pos_bias = np.zeros((N_MARKERS, 3))
        self._marker_rot_bias = np.zeros((N_MARKERS, 3))
        self._cube_bias = np.zeros(3)
        self._cube_rot_bias = np.zeros(3)
        # Common-mode shift shared by every tag (arm markers + cube) for the whole
        # episode: the real pipeline re-anchors the camera from one table tag
        # (real/marker_obs.py), so its calibration/detection error moves all tags
        # together, not independently. Drawn in reset from obs_bias.
        self._common_pos_bias = np.zeros(3)
        self.obs_dim = obs_dim_for(self.prev_actions_n, self.marker_include_rot)

        self.marker_site_ids = []
        for name in MARKER_SITE_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            assert sid >= 0, f"Marker site '{name}' not found in XML"
            self.marker_site_ids.append(sid)
        self.tag_cam_pos = tag_cam_world_pos(self.model, self.data)
        # Camera geometry for the anisotropic (camera-frame) marker noise: focal
        # length from the solved intrinsics and each obs tag's printed edge length,
        # both the same calibration the real solvePnP pipeline uses (src/marker_noise.py,
        # real/marker_spec.py). All arm/cube tags happen to be 20 mm, but map through
        # marker_spec so the obs slots stay the single source of truth.
        self._focal_px = load_focal_px()
        site_to_tag = {site: tag for tag, site in ARM_TAG_TO_SITE.items()}
        self._marker_tag_sizes = np.array(
            [TAG_SIZE_MM[site_to_tag[name]] / 1000.0 for name in MARKER_SITE_NAMES])
        self._cube_tag_size = TAG_SIZE_MM[CUBE_TAG_ID] / 1000.0

        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        assert self.cube_geom_id >= 0, "Cube geom 'cube_geom' not found in XML"
        self.cube_body_id = int(self.model.geom_bodyid[self.cube_geom_id])
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
            self._camera = CameraSim.synchronous(self._step_dt)
        else:
            self._camera = CameraSim(
                frame_s=float(self.cam_latency["frame_ms"]) * 1e-3,
                delay_range_s=(float(self.cam_latency["delay_ms"][0]) * 1e-3,
                               float(self.cam_latency["delay_ms"][1]) * 1e-3),
                jitter_s=float(self.cam_latency["jitter_ms"]) * 1e-3,
                control_dt=self._step_dt)
        self._cam_frame: CamFrame | None = None
        # Hold-last-pose marker state: the obs serves each tag's most recent
        # detection plus its age; a last-capture time of -inf means never seen
        # this episode (zero pose, age pinned at MARKER_AGE_CAP_S). The cube
        # tag holds the identical state.
        self._held_marker_pos = np.zeros((N_MARKERS, 3))
        self._held_marker_rot = np.zeros((N_MARKERS, 3))
        self._marker_last_capture_t = np.full(N_MARKERS, -np.inf)
        self._held_cube_tag_pos = np.zeros(3)
        self._held_cube_tag_rot = np.zeros(3)
        self._cube_last_capture_t = -np.inf
        # Privileged (always-fresh) mirror of the held-tag state: every frame
        # overwrites it regardless of detection, so it serves exactly what a
        # marker_always_visible=true policy sees — same frames, same
        # noise/bias/latency, no dropout and no hold-last-pose. Read only by
        # privileged_obs() (the distillation teacher view, src/distill.py);
        # never fed to this env's own policy. Needs no reset: _ingest_frame
        # overwrites all of it unconditionally, starting with the reset frame.
        self._held_marker_pos_priv = np.zeros((N_MARKERS, 3))
        self._held_marker_rot_priv = np.zeros((N_MARKERS, 3))
        self._marker_last_capture_t_priv = np.full(N_MARKERS, -np.inf)
        self._held_cube_tag_pos_priv = np.zeros(3)
        self._held_cube_tag_rot_priv = np.zeros(3)
        self._cube_last_capture_t_priv = -np.inf
        self._prev_qpos_obs = None
        self._last_encoder_obs = None

        self._parse_config(task_cfg)

        obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def _parse_config(self, cfg):
        """Override for task-specific config."""

    def _set_marker_render_colors(self, detected):
        """Tint each marker site green when detected this frame, red otherwise.
        Visual only (site rgba) — no effect on physics or observations."""
        for sid, det in zip(self.marker_site_ids, detected):
            self.model.site_rgba[sid] = MARKER_VISIBLE_RGBA if det else MARKER_HIDDEN_RGBA

    def _capture_camera_state(self):
        """CamState snapshot of the current MjData — recorded per substep so
        CameraSim can capture frames at any past instant of the tick."""
        marker_pos, marker_rot = marker_world_poses(self.data, self.marker_site_ids)
        (cube_tag_pos,), (cube_tag_rot,) = marker_world_poses(
            self.data, [self.cube_tag_site_id])
        cube_tag_normal = self.data.site_xmat[self.cube_tag_site_id].reshape(3, 3)[:, 2]
        return CamState(marker_pos=marker_pos, marker_rot=marker_rot,
                        marker_normal=marker_world_normals(self.data, self.marker_site_ids),
                        cube_tag_pos=cube_tag_pos, cube_tag_rot=cube_tag_rot,
                        cube_tag_normal=cube_tag_normal.copy(),
                        cube_tag_occluded=cube_tag_occluded(
                            self.model, self.data, self.cube_tag_site_id,
                            self.tag_cam_pos, self.cube_body_id))

    def _process_frame(self, state: CamState) -> CamFrame:
        """One simulated detection of a captured world state: roll the per-frame
        dropout (geometric visibility + DR), apply bias and per-frame noise.
        Runs exactly once per frame, at capture time — a real frame is detected
        once, so consuming it twice re-reads the same values. Undetected tags
        keep garbage pose entries; only the detected flags gate what
        _ingest_frame folds into the held obs state."""
        if self.marker_always_visible:
            detected = np.ones(N_MARKERS, dtype=bool)
            cube_detected = True
        else:
            if self.marker_dropout is None:
                p_near = p_far = 0.0
            else:
                p_near = self.marker_dropout["near"]
                p_far = self.marker_dropout["far"]
            prob = marker_dropout_prob(state.marker_pos, state.marker_normal,
                                       self.tag_cam_pos, p_near, p_far)
            detected = self.np_random.random(N_MARKERS) >= prob
            # The cube tag shares the arm tags' angle/height/dropout geometry
            # plus the capture-time occlusion test (a blocked ray is a certain
            # miss regardless of facing angle).
            if state.cube_tag_occluded:
                cube_prob = 1.0
            else:
                cube_prob = marker_dropout_prob(state.cube_tag_pos[None],
                                                state.cube_tag_normal[None],
                                                self.tag_cam_pos, p_near, p_far)[0]
            cube_detected = bool(self.np_random.random() >= cube_prob)

        marker_pos = state.marker_pos.copy()
        marker_rot = state.marker_rot.copy()
        cube_tag_pos = state.cube_tag_pos.copy()
        cube_tag_rot = state.cube_tag_rot.copy()
        if self.obs_bias is not None:
            # _common_pos_bias hits both arm markers and the cube (shared camera
            # re-anchor error); the per-tag biases stay independent.
            marker_pos = marker_pos + self._marker_pos_bias + self._common_pos_bias
            marker_rot = marker_rot + self._marker_rot_bias
            cube_tag_pos = cube_tag_pos + self._cube_bias + self._common_pos_bias
            cube_tag_rot = cube_tag_rot + self._cube_rot_bias
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
            cube_tag_pos = cube_tag_pos + anisotropic_pos_noise(
                rng, state.cube_tag_pos, self.tag_cam_pos,
                self._cube_tag_size, self._focal_px,
                self.obs_noise["cube_px_noise"], depth_factor)
            marker_rot = marker_rot + rng.normal(0, self.obs_noise["marker_rot_sigma"],
                                                 size=marker_rot.shape)
            cube_tag_rot = cube_tag_rot + rng.normal(0, self.obs_noise["marker_rot_sigma"],
                                                     size=cube_tag_rot.shape)
        return CamFrame(marker_pos=marker_pos, marker_rot=marker_rot,
                        detected=detected, cube_tag_pos=cube_tag_pos,
                        cube_tag_rot=cube_tag_rot, cube_detected=cube_detected)

    def _ingest_frame(self, capture_t, frame: CamFrame):
        """Consume one camera frame: it becomes the current frame (render tint,
        hidden metrics), and each detected tag's measurement — arm markers and
        cube tag alike — overwrites the held pose the obs serves, the same
        per-frame update the real capture thread applies (real/marker_obs.py)."""
        self._cam_frame = frame
        det = frame.detected
        self._held_marker_pos[det] = frame.marker_pos[det]
        self._held_marker_rot[det] = frame.marker_rot[det]
        self._marker_last_capture_t[det] = capture_t
        self._set_marker_render_colors(det)
        if frame.cube_detected:
            self._held_cube_tag_pos = frame.cube_tag_pos
            self._held_cube_tag_rot = frame.cube_tag_rot
            self._cube_last_capture_t = capture_t
        self.model.site_rgba[self.cube_tag_site_id] = \
            MARKER_VISIBLE_RGBA if frame.cube_detected else MARKER_HIDDEN_RGBA

        # Privileged mirror: fold every tag in unconditionally — no dropout,
        # no hold-last-pose — so privileged_obs() serves exactly what a
        # marker_always_visible=true policy would see on this same frame (same
        # capture noise/bias/latency, ages that only reflect pipeline delay).
        self._held_marker_pos_priv[:] = frame.marker_pos
        self._held_marker_rot_priv[:] = frame.marker_rot
        self._marker_last_capture_t_priv[:] = capture_t
        self._held_cube_tag_pos_priv = frame.cube_tag_pos.copy()
        self._held_cube_tag_rot_priv = frame.cube_tag_rot.copy()
        self._cube_last_capture_t_priv = capture_t

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3].copy()

    def _obs_extra(self, cube_pos):
        """Return task-specific obs dimensions appended after [qpos, qvel, markers, cube].

        Receives the held (possibly noisy/stale) cube tag position so derived
        quantities stay consistent with what the agent sees.
        """
        raise NotImplementedError

    def _encoder_obs(self):
        """(qpos, qvel) for this control tick: qpos on the encoder path (fresh,
        biased+noisy), qvel its backward difference like
        real/rollout_common.ArmLoop. Advances _prev_qpos_obs and caches the
        result in _last_encoder_obs so a paired privileged_obs() reuses the
        identical read — the student and teacher views must differ only in the
        tag channels. Call exactly once per tick (from _compute_obs)."""
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

    def _tag_obs(self, held_pos, held_rot, marker_last_t, cube_last_t):
        """(markers, marker_age, cube_age) from a held-tag source: the arm
        markers' obs slice and both age channels. Shared by the student view
        (_compute_obs, hold-last-pose state) and the privileged teacher view
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
        cube_age = np.clip(self.data.time - cube_last_t, 0.0, MARKER_AGE_CAP_S)
        return markers, marker_age, cube_age

    def _compute_obs(self):
        """Build the policy observation. qpos/qvel take the encoder path (fresh;
        qvel differenced like real/rollout_common.ArmLoop); markers and the
        cube tag serve the held per-tag detections with their ages."""
        qpos, qvel = self._encoder_obs()
        markers, marker_age, cube_age = self._tag_obs(
            self._held_marker_pos, self._held_marker_rot,
            self._marker_last_capture_t, self._cube_last_capture_t)
        return np.concatenate([qpos, qvel, markers, marker_age,
                               self._held_cube_tag_pos, self._held_cube_tag_rot,
                               [cube_age],
                               self._obs_extra(self._held_cube_tag_pos),
                               self._prev_actions.flatten()]).astype(np.float32)

    def privileged_obs(self):
        """The observation a marker_always_visible=true policy would see on the
        current state: every tag pose fresh (the always-updated privileged
        mirror, no hold-last-pose and no dropout), sharing the paired
        _compute_obs's encoder read, task extras, and prev-actions so the two
        views differ only in the tag channels. The distillation teacher view
        (src/distill.py). Call once, immediately after the tick's _compute_obs."""
        assert self._last_encoder_obs is not None, \
            "privileged_obs() needs a _compute_obs earlier this tick"
        qpos, qvel = self._last_encoder_obs
        markers, marker_age, cube_age = self._tag_obs(
            self._held_marker_pos_priv, self._held_marker_rot_priv,
            self._marker_last_capture_t_priv, self._cube_last_capture_t_priv)
        return np.concatenate([qpos, qvel, markers, marker_age,
                               self._held_cube_tag_pos_priv, self._held_cube_tag_rot_priv,
                               [cube_age],
                               self._obs_extra(self._held_cube_tag_pos_priv),
                               self._prev_actions.flatten()]).astype(np.float32)

    def _n_jaw_contacts(self):
        """Number of distinct gripper jaws (0, 1, or 2) currently touching the cube."""
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
        """Sample the cube spawn pose, rejecting any the camera cannot actually
        see the tag of — matching the real protocol of placing the sponge where
        the camera sees its tag. Requires the arm (and any task scenery) already
        placed: a candidate must face the camera within the strict (non-grazing)
        angle band, have an unobstructed ray past the arm and scenery, and not
        touch the arm. Writes the accepted pose into qpos (data left forwarded)
        and returns (cube_pos, cube_quat). Returns None when no candidate
        passes within max_attempts."""
        for _ in range(max_attempts):
            cube_xy = self._sample_cube_pos()
            cube_quat, self.cube_rest_half_z = sample_cube_orientation(
                self.np_random, self.cube_half_extents,
                smallest_face_only=self.cube_smallest_face_only)
            cube_pos = np.array([cube_xy[0], cube_xy[1], self.cube_rest_half_z])
            self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
            self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = cube_quat
            mujoco.mj_forward(self.model, self.data)
            tag_pos, _ = marker_world_poses(self.data, [self.cube_tag_site_id])
            normal = self.data.site_xmat[self.cube_tag_site_id].reshape(3, 3)[:, 2]
            # Strict band: p_near=1.0 rejects the flaky 65-70° grazing views,
            # so the first camera frame comfortably detects.
            if marker_dropout_prob(tag_pos, normal[None], self.tag_cam_pos,
                                   p_near=1.0, p_far=0.0)[0] >= 1.0:
                continue
            if cube_tag_occluded(self.model, self.data, self.cube_tag_site_id,
                                 self.tag_cam_pos, self.cube_body_id):
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
        self._floor_contact_steps = 0
        self._cube_drag_steps = 0
        self._markers_hidden_total = 0
        self._cube_hidden_total = 0
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
            self._cube_bias = rng.normal(0, self.obs_bias["cube_sigma"], size=3)
            self._cube_rot_bias = rng.normal(0, self.obs_bias["marker_rot_sigma"], size=3)
            # Shared across all tags (see _common_pos_bias init): one draw the whole
            # episode, added to every marker and the cube in _process_frame.
            self._common_pos_bias = rng.normal(0, self.obs_bias["marker_common_sigma"], size=3)

        self._held_marker_pos[:] = 0.0
        self._held_marker_rot[:] = 0.0
        self._marker_last_capture_t[:] = -np.inf
        self._held_cube_tag_pos = np.zeros(3)
        self._held_cube_tag_rot = np.zeros(3)
        self._cube_last_capture_t = -np.inf
        capture_t, frame = self._camera.reset(self.np_random, self.data.time,
                                              self._capture_camera_state(),
                                              self._process_frame)
        self._ingest_frame(capture_t, frame)

        return self._compute_obs(), {}

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        """Return (reward, terminated, info) for the current step."""
        raise NotImplementedError

    def _on_episode_end(self, info):
        """Hook to add task-specific info at episode end. Mutate info dict."""

    def _on_substep(self):
        self._camera.record(self.data.time, self._capture_camera_state())

    def step(self, action):
        self.step_count += 1
        self._apply_action(action)

        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)
        grasped = self._detect_grasp()
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

        # Advance the camera to the obs instant and ingest every frame that
        # became available this tick (dropout/noise were frozen in at capture
        # time): each detection — arm markers and cube tag alike — refreshes
        # its held pose. Losing a tag freezes its pose while its age channel
        # grows — there is deliberately no reward term for it.
        for capture_t, frame in self._camera.observe(self.data.time, self.np_random,
                                                     self._process_frame):
            self._ingest_frame(capture_t, frame)
        self._markers_hidden_total += N_MARKERS - int(self._cam_frame.detected.sum())
        self._cube_hidden_total += not self._cam_frame.cube_detected

        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            info["max_cube_height"] = self._max_cube_height
            if self.floor_contact_penalty:
                info["floor_contact_ratio"] = self._floor_contact_steps / self.step_count
            info["cube_drag_ratio"] = self._cube_drag_steps / self.step_count
            # Fraction of marker-steps hidden from the camera (0 = always visible).
            info["marker_hidden_ratio"] = \
                self._markers_hidden_total / (self.step_count * N_MARKERS)
            info["cube_hidden_ratio"] = self._cube_hidden_total / self.step_count
            self._on_episode_end(info)

        if self.render_mode == "human":
            self._render_human()

        return self._compute_obs(), float(reward), terminated, truncated, info
