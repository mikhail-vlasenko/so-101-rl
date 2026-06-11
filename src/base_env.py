"""Base gymnasium environment for SO-101 arm tasks.

Shared MuJoCo setup, contact detection, rendering, and reset/step skeleton.
Subclasses define task-specific config, reward, termination, and observations.
"""

from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from src.servo_profile import ServoProfile
from src.units import action_to_target


def obs_dim_for(prev_actions_n: int) -> int:
    """OBS_DIM = qpos(6) + qvel(6) + markers(2*6) + cube_pos(3) + extra(4) + prev_actions(N*6).

    Each marker contributes its world xyz plus a world rotation vector
    (axis-angle, 3 dims) — the same quantities the camera pipeline recovers
    for the physical AprilTags (real/pose.py rvec/tvec, camera->world mapped).
    """
    return 31 + prev_actions_n * 6


# AprilTags glued to the arm (ids per real/marker_spec.py ROLES): "finger" on
# the bottom face of the fixed jaw, "wrist" on the right face of the wrist.
# Sites of the same names in so101.xml mirror their placement.
MARKER_SITE_NAMES = ["marker_finger", "marker_wrist"]
N_MARKERS = len(MARKER_SITE_NAMES)


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


def sample_cube_orientation(rng, half_extents):
    """Spawn orientation for the sponge box: standing on a non-largest face.

    half_extents (hx, hy, hz) must be strictly ordered hx > hy > hz (the
    3 x 2 x 1.5 cm box). The box stands either on its hy*hz face (x-axis up,
    2*hx tall) or its hx*hz face (y-axis up, 2*hy tall), never on the largest
    hx*hy face, with a uniform random yaw. Returns (quat wxyz, rest_half_z).
    """
    hx, hy, hz = half_extents
    assert hx > hy > hz, f"expected strictly ordered half extents, got {half_extents}"
    s = np.sqrt(0.5)
    if rng.uniform() < 0.5:
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


class SO101BaseEnv(gym.Env):
    """Base class for SO-101 arm tasks with shared MuJoCo setup."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    XML_PATH: str  # subclasses must set
    TASK_ID: float  # 0.0 = lift, 1.0 = pickplace
    TASK_NAME: str  # "lift" or "pickplace"

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1, xml_path=None,
                 obs_noise=None, obs_latency=0, obs_bias=None, prev_actions_n=2):
        super().__init__()
        self.render_mode = render_mode
        self.slow_factor = slow_factor
        # dict with keys qpos_sigma, qvel_sigma, marker_pos_sigma,
        # marker_rot_sigma, cube_sigma; or None
        self.obs_noise = obs_noise
        self.obs_latency = int(obs_latency)  # frames; agent sees obs from N steps ago
        self._obs_history: deque = deque()
        # dict with keys qpos_sigma, marker_pos_sigma, marker_rot_sigma,
        # cube_sigma; or None. Marker biases are sampled independently per
        # marker (uncorrelated: each tag has its own glue/pose-estimate error).
        self.obs_bias = obs_bias
        self._qpos_bias = np.zeros(len(JOINT_NAMES))
        self._marker_pos_bias = np.zeros((N_MARKERS, 3))
        self._marker_rot_bias = np.zeros((N_MARKERS, 3))
        self._cube_bias = np.zeros(3)
        self.prev_actions_n = int(prev_actions_n)
        self.obs_dim = obs_dim_for(self.prev_actions_n)

        self.model = mujoco.MjModel.from_xml_path(xml_path or self.XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.n_joints = len(JOINT_NAMES)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in JOINT_NAMES]
        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.marker_site_ids = []
        for name in MARKER_SITE_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            assert sid >= 0, f"Marker site '{name}' not found in XML"
            self.marker_site_ids.append(sid)

        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

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

        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert self.floor_geom_id >= 0, "Floor geom 'floor' not found in XML"
        self.arm_geom_ids = {i for i in range(self.model.ngeom) if self.model.geom_group[i] == 3}

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_idx = self.model.jnt_qposadr[cube_joint_id]
        self.cube_dofadr = self.model.jnt_dofadr[cube_joint_id]
        # Sponge box half extents; per-episode resting half-height (depends on
        # which face it stands on) is set in reset() as self.cube_rest_half_z.
        self.cube_half_extents = self.model.geom_size[self.cube_geom_id].copy()

        self.joint_low = self.model.jnt_range[self.joint_ids, 0]
        self.joint_high = self.model.jnt_range[self.joint_ids, 1]

        # Common config
        cfg = env_cfg
        self.action_scale = float(cfg["action_scale"])
        self.max_steps = int(cfg["max_steps"])
        self.n_substeps = int(cfg["n_substeps"])
        self.cube_low = np.array(cfg["cube_low"])
        self.cube_high = np.array(cfg["cube_high"])
        self.floor_contact_penalty = float(cfg["floor_contact_penalty"])

        self.gripper_idx = JOINT_NAMES.index("gripper")

        self._prev_actions = np.zeros((self.prev_actions_n, self.n_joints), dtype=np.float32)

        # Firmware motion-profile model: ctrl carries the profiled setpoint,
        # not the raw tick target (see src/servo_profile.py).
        self._servo_profile = ServoProfile(self.n_joints)
        self._ctrl_target = None

        # Cube-drag metric: cube center within DRAG_HEIGHT_TOL of resting height
        # AND lateral speed above DRAG_SPEED_THRESH (m/s).
        self._step_dt = self.model.opt.timestep * self.n_substeps
        self.DRAG_HEIGHT_TOL = 0.005
        self.DRAG_SPEED_THRESH = 0.01

        self._parse_config(cfg)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_joints,), dtype=np.float32)
        obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.step_count = 0
        self.viewer = None

    def _parse_config(self, cfg):
        """Override for task-specific config."""

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3].copy()

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_joint_pos(self):
        return self.data.qpos[self.joint_qposadr].copy()

    def _obs_extra(self, cube_pos):
        """Return task-specific obs dimensions appended after [qpos, qvel, markers, cube].

        Receives the (possibly noisy) cube_pos so derived quantities stay consistent
        with the cube_pos visible to the agent.
        """
        raise NotImplementedError

    def _compute_obs(self):
        qpos = self.data.qpos[self.joint_qposadr].copy()
        qvel = self.data.qvel[self.joint_dofadr].copy()
        marker_pos, marker_rot = marker_world_poses(self.data, self.marker_site_ids)
        cube_pos = self._get_cube_pos()

        if self.obs_bias is not None:
            qpos = qpos + self._qpos_bias
            marker_pos = marker_pos + self._marker_pos_bias
            marker_rot = marker_rot + self._marker_rot_bias
            cube_pos = cube_pos + self._cube_bias

        if self.obs_noise is not None:
            rng = self.np_random
            qpos = qpos + rng.normal(0, self.obs_noise["qpos_sigma"], size=qpos.shape)
            qvel = qvel + rng.normal(0, self.obs_noise["qvel_sigma"], size=qvel.shape)
            marker_pos = marker_pos + rng.normal(0, self.obs_noise["marker_pos_sigma"],
                                                 size=marker_pos.shape)
            marker_rot = marker_rot + rng.normal(0, self.obs_noise["marker_rot_sigma"],
                                                 size=marker_rot.shape)
            cube_pos = cube_pos + rng.normal(0, self.obs_noise["cube_sigma"], size=cube_pos.shape)

        # hstack interleaves per marker: [pos_finger, rot_finger, pos_wrist, rot_wrist]
        markers = np.hstack([marker_pos, marker_rot]).flatten()
        return np.concatenate([qpos, qvel, markers, cube_pos, self._obs_extra(cube_pos),
                               self._prev_actions.flatten()]).astype(np.float32)

    def _get_obs(self):
        raw = self._compute_obs()
        if self.obs_latency == 0:
            return raw
        self._obs_history.append(raw)
        while len(self._obs_history) > self.obs_latency + 1:
            self._obs_history.popleft()
        return self._obs_history[0]

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

    def _on_reset(self, cube_pos):
        """Hook for task-specific reset state. Called after common reset."""

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._obs_history.clear()
        self._prev_actions[:] = 0.0

        cube_xy = self._sample_cube_pos()
        cube_quat, self.cube_rest_half_z = sample_cube_orientation(
            self.np_random, self.cube_half_extents)
        cube_pos = np.array([cube_xy[0], cube_xy[1], self.cube_rest_half_z])
        self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
        self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = cube_quat

        self._on_reset(cube_pos)

        # Sample random arm position, rejecting any arm-environment collisions
        attempt = 0
        while True:
            joint_pos = self.np_random.uniform(self.joint_low, self.joint_high)
            self.data.qpos[self.joint_qposadr] = joint_pos
            self.data.ctrl[:self.n_joints] = joint_pos
            # If actuators carry activation state (e.g. dyntype="filter"),
            # initialize it at the joint position so the controller doesn't
            # snap the arm from 0 toward joint_pos on the first step.
            if self.model.na > 0:
                self.data.act[:] = joint_pos
            mujoco.mj_forward(self.model, self.data)
            if not self._has_arm_collision():
                break
            attempt += 1
            if attempt % 10 == 0:
                print(f"WARNING: {attempt} arm position samples rejected (collision)")
        self._servo_profile.reset(joint_pos)
        self._ctrl_target = joint_pos.copy()
        self.step_count = 0
        self._max_cube_height = cube_pos[2]
        self._floor_contact_steps = 0
        self._cube_drag_steps = 0
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

        return self._get_obs(), {}

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        """Return (reward, terminated, info) for the current step."""
        raise NotImplementedError

    def _on_episode_end(self, info):
        """Hook to add task-specific info at episode end. Mutate info dict."""

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        if self.prev_actions_n > 0:
            if self.prev_actions_n > 1:
                self._prev_actions[:-1] = self._prev_actions[1:]
            self._prev_actions[-1] = action

        current = self.data.qpos[self.joint_qposadr].copy()
        target = action_to_target(current, action, self.action_scale,
                                  self.joint_low, self.joint_high)
        setpoints = self._servo_profile.tick(self._ctrl_target, target,
                                             self.n_substeps, self.model.opt.timestep)
        for k in range(self.n_substeps):
            self.data.ctrl[:self.n_joints] = setpoints[k]
            mujoco.mj_step(self.model, self.data)
        self._ctrl_target = target

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

        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            info["max_cube_height"] = self._max_cube_height
            if self.floor_contact_penalty:
                info["floor_contact_ratio"] = self._floor_contact_steps / self.step_count
            info["cube_drag_ratio"] = self._cube_drag_steps / self.step_count
            self._on_episode_end(info)

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), float(reward), terminated, truncated, info

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
