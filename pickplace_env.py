"""Gymnasium environment: SO-101 arm pick-and-place task."""

import enum

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class Phase(enum.IntEnum):
    REACH = 0
    GRASP = 1
    LIFT = 2
    PLACE = 3
    RETURN = 4


class SO101PickPlaceEnv(gym.Env):
    """Pick up a cube, place it at a target, and return to passive pose."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1):
        super().__init__()
        self.render_mode = render_mode
        self.slow_factor = slow_factor

        self.model = mujoco.MjModel.from_xml_path("so101/scene_pickplace.xml")
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        self.n_joints = len(self.joint_names)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.joint_names]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        # Cube geom ID
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # Gripper geom IDs (fixed + moving jaw)
        self.gripper_geom_ids = set()
        for name in ["fixed_jaw_box1", "fixed_jaw_box2", "fixed_jaw_box3",
                      "fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6",
                      "fixed_jaw_box7", "fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2",
                      "fixed_jaw_sph_tip3",
                      "moving_jaw_box1", "moving_jaw_box2", "moving_jaw_box3",
                      "moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.gripper_geom_ids.add(gid)

        # Cube freejoint qpos index
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_idx = self.model.jnt_qposadr[cube_joint_id]

        # Joint limits
        self.joint_low = self.model.jnt_range[self.joint_ids, 0]
        self.joint_high = self.model.jnt_range[self.joint_ids, 1]

        # Config
        cfg = env_cfg
        self.action_scale = float(cfg["action_scale"])
        self.max_steps = int(cfg["max_steps"])
        self.n_substeps = int(cfg["n_substeps"])
        self.lift_height = float(cfg["lift_height"])
        self.cube_low = np.array(cfg["cube_low"])
        self.cube_high = np.array(cfg["cube_high"])
        self.place_target = np.array(cfg["place_target"])
        self.passive_pose = np.array(cfg["passive_pose"])

        # Gripper joint index
        self.gripper_idx = self.joint_names.index("gripper")

        # Spaces: 6 joint pos + 6 joint vel + 3 ee pos + 3 cube pos = 18
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_joints,), dtype=np.float32)
        obs_high = np.full(18, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.step_count = 0
        self.phase = Phase.REACH
        self.viewer = None

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3].copy()

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_joint_pos(self):
        return self.data.qpos[:self.n_joints].copy()

    def _get_obs(self):
        qpos = self.data.qpos[:self.n_joints].copy()
        qvel = self.data.qvel[:self.n_joints].copy()
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        return np.concatenate([qpos, qvel, ee_pos, cube_pos]).astype(np.float32)

    def _has_gripper_contact(self):
        """Check if cube_geom is in contact with any gripper geom."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.cube_geom_id and g2 in self.gripper_geom_ids:
                return True
            if g2 == self.cube_geom_id and g1 in self.gripper_geom_ids:
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Random cube position
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
        # Cube orientation: identity quaternion
        self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = [1, 0, 0, 0]

        # Small random arm joint noise, gripper open
        noise = self.np_random.uniform(-0.1, 0.1, size=self.n_joints)
        noise[self.gripper_idx] = self.joint_high[self.gripper_idx]  # gripper open
        self.data.qpos[:self.n_joints] = np.clip(noise, self.joint_low, self.joint_high)
        self.data.ctrl[:self.n_joints] = self.data.qpos[:self.n_joints]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.phase = Phase.REACH
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Delta joint position control (gripper is agent-controlled)
        current = self.data.ctrl[:self.n_joints].copy()
        target = current + action * self.action_scale
        target = np.clip(target, self.joint_low, self.joint_high)
        self.data.ctrl[:self.n_joints] = target

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # State
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        joint_pos = self._get_joint_pos()
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)
        grasped = self._detect_grasp()

        # Phase transitions
        self._update_phase(ee_cube_dist, grasped, cube_pos, joint_pos)

        # Reward
        reward = self._compute_reward(ee_pos, cube_pos, joint_pos, ee_cube_dist, grasped)

        # Termination
        terminated = False
        if self.phase == Phase.RETURN:
            joint_dist = np.linalg.norm(joint_pos - self.passive_pose)
            if joint_dist < 0.3:
                reward += 10.0
                terminated = True

        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), float(reward), terminated, truncated, {
            "phase": self.phase.name,
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
        }

    def _update_phase(self, ee_cube_dist, grasped, cube_pos, joint_pos):
        if self.phase == Phase.REACH:
            if ee_cube_dist < 0.03:
                self.phase = Phase.GRASP
        elif self.phase == Phase.GRASP:
            if grasped:
                self.phase = Phase.LIFT
            elif ee_cube_dist > 0.08:
                self.phase = Phase.REACH
        elif self.phase == Phase.LIFT:
            if not grasped:
                self.phase = Phase.REACH
            elif cube_pos[2] > self.lift_height:
                self.phase = Phase.PLACE
        elif self.phase == Phase.PLACE:
            if not grasped:
                self.phase = Phase.REACH
            else:
                xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
                if xy_dist < 0.03 and cube_pos[2] < 0.05:
                    self.phase = Phase.RETURN

    def _compute_reward(self, ee_pos, cube_pos, joint_pos, ee_cube_dist, grasped):
        reward = -0.01  # time penalty

        if self.phase == Phase.REACH:
            reward += -ee_cube_dist
            if ee_cube_dist < 0.03:
                reward += 1.0

        elif self.phase == Phase.GRASP:
            # Reward for closing gripper near cube
            gripper_val = self.data.qpos[self.joint_ids[self.gripper_idx]]
            reward += -ee_cube_dist
            reward += 0.5 * (1.0 - gripper_val)  # reward closing
            if grasped:
                reward += 5.0

        elif self.phase == Phase.LIFT:
            height = cube_pos[2]
            reward += 2.0 * min(height / self.lift_height, 1.0)
            if height > self.lift_height:
                reward += 3.0

        elif self.phase == Phase.PLACE:
            xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
            reward += -xy_dist
            if xy_dist < 0.03:
                reward += 2.0
                # Reward for lowering
                reward += 1.0 * max(0, 1.0 - cube_pos[2] / 0.05)

        elif self.phase == Phase.RETURN:
            joint_dist = np.linalg.norm(joint_pos - self.passive_pose)
            reward += -joint_dist
            if joint_dist < 0.5:
                reward += 2.0

        return reward

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
