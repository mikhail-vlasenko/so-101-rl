"""Gymnasium environment: SO-101 arm reach task."""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class SO101ReachEnv(gym.Env):
    """Move the SO-101 end-effector to a random target position."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1):
        super().__init__()
        self.render_mode = render_mode
        self.slow_factor = slow_factor

        # Load model
        self.model = mujoco.MjModel.from_xml_path("so101/scene_reach.xml")
        self.data = mujoco.MjData(self.model)

        # Joint and site IDs
        self.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        self.actuator_names = self.joint_names  # same names
        self.n_joints = len(self.joint_names)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in self.actuator_names]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")

        # Joint limits (for clipping)
        self.joint_low = self.model.jnt_range[self.joint_ids, 0]
        self.joint_high = self.model.jnt_range[self.joint_ids, 1]

        cfg = env_cfg or {}
        self.action_scale = float(cfg["action_scale"])
        self.max_steps = int(cfg["max_steps"])
        self.success_threshold = float(cfg["success_threshold"])
        self.success_bonus = float(cfg["success_bonus"])
        self.target_low = np.array(cfg["target_low"])
        self.target_high = np.array(cfg["target_high"])

        # Action: delta joint positions (scaled to [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_joints,), dtype=np.float32)

        # Observation: joint_pos(6) + joint_vel(6) + ee_pos(3) + target_pos(3) = 18
        obs_high = np.full(18, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.step_count = 0

        # Viewer (lazy init)
        self.viewer = None

    def _get_obs(self):
        qpos = self.data.qpos[: self.n_joints].copy()
        qvel = self.data.qvel[: self.n_joints].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self.data.mocap_pos[0].copy()
        return np.concatenate([qpos, qvel, ee_pos, target_pos]).astype(np.float32)

    def _sample_target(self):
        return self.np_random.uniform(self.target_low, self.target_high)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize target
        target = self._sample_target()
        self.data.mocap_pos[0] = target

        # Small random initial joint positions
        noise = self.np_random.uniform(-0.1, 0.1, size=self.n_joints)
        self.data.qpos[: self.n_joints] = np.clip(noise, self.joint_low, self.joint_high)
        self.data.ctrl[: self.n_joints] = self.data.qpos[: self.n_joints]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Apply delta to current joint targets
        current = self.data.ctrl[: self.n_joints].copy()
        target = current + action * self.action_scale
        target = np.clip(target, self.joint_low, self.joint_high)

        # Keep gripper open
        gripper_idx = self.joint_names.index("gripper")
        target[gripper_idx] = self.joint_high[gripper_idx]

        self.data.ctrl[: self.n_joints] = target

        # Step simulation (multiple substeps for stability)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Compute reward
        ee_pos = self.data.site_xpos[self.ee_site_id]
        target_pos = self.data.mocap_pos[0]
        dist = np.linalg.norm(ee_pos - target_pos)
        reward = -dist

        terminated = False
        if dist < self.success_threshold:
            reward += self.success_bonus
            terminated = True

        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), float(reward), terminated, truncated, {"distance": dist}

    def _render_human(self):
        import time
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        if self.slow_factor > 1:
            time.sleep(self.model.opt.timestep * 10 * (self.slow_factor - 1))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
