"""Gymnasium environment: SO-101 arm cube lifting task.

Simpler than pick-and-place — agent learns to grasp and lift a cube.
Terminates when cube reaches target height.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


# Reward constants
TIME_PENALTY = -0.05
EE_CUBE_COEFF = -0.5
GRASP_HOLD_REWARD = 0.05
HEIGHT_PROGRESS_COEFF = 200.0


class SO101LiftEnv(gym.Env):
    """Grasp a cube and lift it to a target height."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    XML_PATH = "so101/scene_lift.xml"

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1, xml_path=None):
        super().__init__()
        self.render_mode = render_mode
        self.slow_factor = slow_factor

        self.model = mujoco.MjModel.from_xml_path(xml_path or self.XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        self.n_joints = len(self.joint_names)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.joint_names]
        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        # Cube geom ID
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # Gripper geom IDs per jaw
        fixed_jaw_names = [
            "fixed_jaw_box1", "fixed_jaw_box2", "fixed_jaw_box3",
            "fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6",
            "fixed_jaw_box7", "fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2",
            "fixed_jaw_sph_tip3",
        ]
        moving_jaw_names = [
            "moving_jaw_box1", "moving_jaw_box2", "moving_jaw_box3",
            "moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3",
        ]
        self.fixed_jaw_geom_ids = set()
        self.moving_jaw_geom_ids = set()
        for name in fixed_jaw_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"Gripper geom '{name}' not found in XML"
            self.fixed_jaw_geom_ids.add(gid)
        for name in moving_jaw_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"Gripper geom '{name}' not found in XML"
            self.moving_jaw_geom_ids.add(gid)
        self.gripper_geom_ids = self.fixed_jaw_geom_ids | self.moving_jaw_geom_ids

        # Floor geom ID + all arm/gripper collision geom IDs (group 3)
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert self.floor_geom_id >= 0, "Floor geom 'floor' not found in XML"
        self.arm_geom_ids = {i for i in range(self.model.ngeom) if self.model.geom_group[i] == 3}

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
        self.cube_low = np.array(cfg["cube_low"])
        self.cube_high = np.array(cfg["cube_high"])
        self.floor_contact_penalty = float(cfg["floor_contact_penalty"])
        self.target_height = float(cfg["target_height"])

        self.gripper_idx = self.joint_names.index("gripper")

        # Spaces: 6 joint pos + 6 joint vel + 3 ee pos + 3 cube pos = 18
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_joints,), dtype=np.float32)
        obs_high = np.full(18, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.step_count = 0
        self.viewer = None

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3].copy()

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_obs(self):
        qpos = self.data.qpos[self.joint_qposadr].copy()
        qvel = self.data.qvel[self.joint_dofadr].copy()
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        return np.concatenate([qpos, qvel, ee_pos, cube_pos]).astype(np.float32)

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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Random cube position
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        self.data.qpos[self.cube_qpos_idx:self.cube_qpos_idx + 3] = cube_pos
        self.data.qpos[self.cube_qpos_idx + 3:self.cube_qpos_idx + 7] = [1, 0, 0, 0]

        # Small random arm joint noise, gripper open
        noise = self.np_random.uniform(-0.1, 0.1, size=self.n_joints)
        noise[self.gripper_idx] = self.joint_high[self.gripper_idx]
        self.data.qpos[self.joint_qposadr] = np.clip(noise, self.joint_low, self.joint_high)
        self.data.ctrl[:self.n_joints] = self.data.qpos[self.joint_qposadr]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self._prev_cube_z = cube_pos[2]
        self._max_cube_height = cube_pos[2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Delta joint position control
        current = self.data.ctrl[:self.n_joints].copy()
        target = current + action * self.action_scale
        target = np.clip(target, self.joint_low, self.joint_high)
        self.data.ctrl[:self.n_joints] = target

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # State
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)
        grasped = self._detect_grasp()
        floor_contact = self._has_floor_contact() if self.floor_contact_penalty else False
        self._max_cube_height = max(self._max_cube_height, cube_pos[2])

        # Reward
        reward = TIME_PENALTY
        reward += EE_CUBE_COEFF * ee_cube_dist

        if grasped:
            reward += GRASP_HOLD_REWARD

        height_delta = cube_pos[2] - self._prev_cube_z
        reward += HEIGHT_PROGRESS_COEFF * height_delta
        self._prev_cube_z = cube_pos[2]

        if floor_contact:
            reward += self.floor_contact_penalty

        # Termination: cube reached target height
        terminated = cube_pos[2] >= self.target_height
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self._render_human()

        info = {
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
            "cube_height": cube_pos[2],
        }
        if floor_contact:
            info["floor_contact"] = floor_contact
        if terminated or truncated:
            info["max_cube_height"] = self._max_cube_height

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
