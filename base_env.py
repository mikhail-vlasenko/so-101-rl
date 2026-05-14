"""Base gymnasium environment for SO-101 arm tasks.

Shared MuJoCo setup, contact detection, rendering, and reset/step skeleton.
Subclasses define task-specific config, reward, termination, and observations.
"""

from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def _obs_dim_for(prev_actions_n: int) -> int:
    """OBS_DIM = qpos(6) + qvel(6) + ee_pos(3) + cube_pos(3) + extra(4) + prev_actions(N*6)."""
    return 22 + prev_actions_n * 6

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
        self.obs_noise = obs_noise  # dict with keys qpos_sigma, qvel_sigma, ee_sigma, cube_sigma; or None
        self.obs_latency = int(obs_latency)  # frames; agent sees obs from N steps ago
        self._obs_history: deque = deque()
        self.obs_bias = obs_bias  # dict with keys qpos_sigma, ee_sigma, cube_sigma; or None
        self._qpos_bias = np.zeros(len(JOINT_NAMES))
        self._ee_bias = np.zeros(3)
        self._cube_bias = np.zeros(3)
        self.prev_actions_n = int(prev_actions_n)
        self.obs_dim = _obs_dim_for(self.prev_actions_n)

        self.model = mujoco.MjModel.from_xml_path(xml_path or self.XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.n_joints = len(JOINT_NAMES)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in JOINT_NAMES]
        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

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
        self.cube_half_z = float(self.model.geom_size[self.cube_geom_id, 2])

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
        """Return task-specific obs dimensions appended after [qpos, qvel, ee, cube].

        Receives the (possibly noisy) cube_pos so derived quantities stay consistent
        with the cube_pos visible to the agent.
        """
        raise NotImplementedError

    def _compute_obs(self):
        qpos = self.data.qpos[self.joint_qposadr].copy()
        qvel = self.data.qvel[self.joint_dofadr].copy()
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()

        if self.obs_bias is not None:
            qpos = qpos + self._qpos_bias
            ee_pos = ee_pos + self._ee_bias
            cube_pos = cube_pos + self._cube_bias

        if self.obs_noise is not None:
            rng = self.np_random
            qpos = qpos + rng.normal(0, self.obs_noise["qpos_sigma"], size=qpos.shape)
            qvel = qvel + rng.normal(0, self.obs_noise["qvel_sigma"], size=qvel.shape)
            ee_pos = ee_pos + rng.normal(0, self.obs_noise["ee_sigma"], size=ee_pos.shape)
            cube_pos = cube_pos + rng.normal(0, self.obs_noise["cube_sigma"], size=cube_pos.shape)

        return np.concatenate([qpos, qvel, ee_pos, cube_pos, self._obs_extra(cube_pos),
                               self._prev_actions.flatten()]).astype(np.float32)

    def _get_obs(self):
        raw = self._compute_obs()
        if self.obs_latency == 0:
            return raw
        self._obs_history.append(raw)
        while len(self._obs_history) > self.obs_latency + 1:
            self._obs_history.popleft()
        return self._obs_history[0]

    def _has_gripper_contact(self):
        """Check if cube_geom is in contact with both jaws simultaneously."""
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
            if fixed_contact and moving_contact:
                return True
        return False

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

    def _detect_grasp(self):
        """Grasp = cube close to EE + gripper closing + contact."""
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        dist = np.linalg.norm(ee_pos - cube_pos)
        gripper_val = self.data.qpos[self.joint_ids[self.gripper_idx]]
        return (dist < 0.05
                and gripper_val < 0.3
                and self._has_gripper_contact())

    def _sample_cube_pos(self):
        """Return random cube position. Override for rejection sampling."""
        return self.np_random.uniform(self.cube_low, self.cube_high)

    def _on_reset(self, cube_pos):
        """Hook for task-specific reset state. Called after common reset."""

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._obs_history.clear()
        self._prev_actions[:] = 0.0

        cube_pos = self._sample_cube_pos()
        self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
        self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = [1, 0, 0, 0]

        self._on_reset(cube_pos)

        # Sample random arm position, rejecting any arm-environment collisions
        attempt = 0
        while True:
            joint_pos = self.np_random.uniform(self.joint_low, self.joint_high)
            self.data.qpos[self.joint_qposadr] = joint_pos
            self.data.ctrl[:self.n_joints] = joint_pos
            mujoco.mj_forward(self.model, self.data)
            if not self._has_arm_collision():
                break
            attempt += 1
            if attempt % 10 == 0:
                print(f"WARNING: {attempt} arm position samples rejected (collision)")
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
            self._ee_bias = rng.normal(0, self.obs_bias["ee_sigma"], size=3)
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
        target = current + action * self.action_scale
        target = np.clip(target, self.joint_low, self.joint_high)
        self.data.ctrl[:self.n_joints] = target

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

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
        if cube_pos[2] < self.cube_half_z + self.DRAG_HEIGHT_TOL and cube_xy_speed > self.DRAG_SPEED_THRESH:
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
