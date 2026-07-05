"""SO-101 fixed-waypoint reaching env.

Safety-first sanity task for the sim-to-real pipeline. The agent must drive the
arm to one of N fixed joint-space waypoints (selected at reset, encoded as a
one-hot in the observation) and dwell there for `dwell_steps` consecutive steps.

No cube. EE is required to stay above `ee_z_min` so the gripper never approaches
the table during real-world deployment. Built on SO101ArmEnv (shared arm/drive
machinery); the cube + camera obs pipeline of SO101BaseEnv does not apply here.
"""

import numpy as np
import mujoco
from gymnasium import spaces

from src.base_env import SO101ArmEnv


class SO101ReachEnv(SO101ArmEnv):
    """Reach one of N fixed joint-space waypoints and dwell."""

    XML_PATH = "so101/scene.xml"
    TASK_NAME = "reach"

    def __init__(self, render_mode=None, env_cfg=None, slow_factor=1, xml_path=None,
                 obs_noise=None, cam_latency=None, obs_bias=None, marker_dropout=None,
                 marker_always_visible=False, marker_include_rot=False, prev_actions_n=2):
        # not used by reach env (kept for train.py signature)
        del obs_noise, cam_latency, obs_bias, marker_dropout, marker_always_visible
        del marker_include_rot
        super().__init__(render_mode=render_mode, slow_factor=slow_factor,
                         xml_path=xml_path, prev_actions_n=prev_actions_n,
                         env_cfg=env_cfg)

        self.joint_center = 0.5 * (self.joint_low + self.joint_high)
        self.joint_half_range = 0.5 * (self.joint_high - self.joint_low)

        cfg = env_cfg
        self.ee_z_min = float(cfg["ee_z_min"])
        self.tolerance = float(cfg["tolerance"])
        self.dwell_steps = int(cfg["dwell_steps"])
        self.limit_margin = float(cfg["limit_margin"])

        self.waypoints = np.array(cfg["waypoints"], dtype=np.float64)
        assert self.waypoints.ndim == 2 and self.waypoints.shape[1] == self.n_joints, (
            f"waypoints must be (N, {self.n_joints}); got {self.waypoints.shape}"
        )
        self.n_waypoints = self.waypoints.shape[0]
        self._validate_waypoints_safe()

        # Reward coefficients
        self.progress_coeff = float(cfg["progress_coeff"])
        self.action_penalty_coeff = float(cfg["action_penalty_coeff"])
        self.in_target_bonus = float(cfg["in_target_bonus"])
        self.dwell_complete_bonus = float(cfg["dwell_complete_bonus"])
        self.ee_below_penalty = float(cfg["ee_below_penalty"])
        self.joint_limit_penalty_coeff = float(cfg["joint_limit_penalty_coeff"])
        self.time_penalty = float(cfg["time_penalty"])

        # obs = [qpos(n), qvel(n), ee_pos(3), waypoint_onehot(N), prev_actions(prev_actions_n * n)]
        obs_dim = 2 * self.n_joints + 3 + self.n_waypoints + self.prev_actions_n * self.n_joints
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def _ee_z_at(self, qpos):
        """Forward-kinematics-only EE z for a given joint config."""
        saved_qpos = self.data.qpos.copy()
        saved_ctrl = self.data.ctrl.copy()
        self.data.qpos[self.joint_qposadr] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        ee_z = float(self.data.site_xpos[self.ee_site_id][2])
        self.data.qpos[:] = saved_qpos
        self.data.ctrl[:] = saved_ctrl
        mujoco.mj_kinematics(self.model, self.data)
        return ee_z

    def _joint_limit_excess(self, qpos):
        """Per-joint distance into the limit-penalty zone (zero outside). Shape (n_joints,)."""
        norm_dist = np.abs(qpos - self.joint_center) / self.joint_half_range
        return np.maximum(0.0, norm_dist - self.limit_margin)

    def _validate_waypoints_safe(self):
        """Crash on init if any configured waypoint is unsafe (EE too low,
        out of range, or inside the joint-limit penalty zone — the policy must
        not be asked to reach a target that the limit penalty discourages)."""
        for i, wp in enumerate(self.waypoints):
            assert np.all(wp >= self.joint_low) and np.all(wp <= self.joint_high), (
                f"Waypoint {i} {wp} out of joint range [{self.joint_low}, {self.joint_high}]"
            )
            ee_z = self._ee_z_at(wp)
            assert ee_z >= self.ee_z_min, (
                f"Waypoint {i} {wp} has EE z={ee_z:.3f} < ee_z_min={self.ee_z_min}"
            )
            excess = self._joint_limit_excess(wp)
            assert np.all(excess == 0.0), (
                f"Waypoint {i} {wp} is inside the joint-limit penalty zone "
                f"(margin={self.limit_margin}); per-joint excess={excess.tolist()}"
            )

    def _waypoint_onehot(self):
        oh = np.zeros(self.n_waypoints, dtype=np.float32)
        oh[self.waypoint_idx] = 1.0
        return oh

    def _get_obs(self):
        qpos = self.data.qpos[self.joint_qposadr].copy()
        qvel = self.data.qvel[self.joint_dofadr].copy()
        ee_pos = self._get_ee_pos()
        return np.concatenate([qpos, qvel, ee_pos, self._waypoint_onehot(),
                               self._prev_actions.flatten()]).astype(np.float32)

    def _sample_safe_init(self):
        """Sample a joint pose with EE z >= ee_z_min and no arm collision."""
        for attempt in range(200):
            q = self.np_random.uniform(self.joint_low, self.joint_high)
            self._write_arm_pose(q)
            if (not self._has_arm_collision()) and self._get_ee_pos()[2] >= self.ee_z_min:
                return q
        raise RuntimeError(
            "Could not sample a safe initial pose in 200 attempts "
            "(EE above ee_z_min and no collision). Check joint limits / scene."
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._prev_actions[:] = 0.0

        self.waypoint_idx = int(self.np_random.integers(self.n_waypoints))
        self.target_qpos = self.waypoints[self.waypoint_idx].copy()

        q0 = self._sample_safe_init()
        self._servo_profile.reset(q0)
        self._ctrl_target = q0.copy()

        self.step_count = 0
        self.dwell_count = 0
        self._prev_dist = float(np.linalg.norm(q0 - self.target_qpos))
        self._ee_below_steps = 0
        self._sum_limit_excess = 0.0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = self._apply_action(action)

        q = self._get_joint_pos()
        ee_pos = self._get_ee_pos()
        dist = float(np.linalg.norm(q - self.target_qpos))
        in_target = dist < self.tolerance

        if in_target:
            self.dwell_count += 1
        else:
            self.dwell_count = 0

        reward = self.time_penalty
        reward += self.progress_coeff * (self._prev_dist - dist)
        reward -= self.action_penalty_coeff * float(np.sum(action ** 2))
        if in_target:
            reward += self.in_target_bonus

        ee_below = ee_pos[2] < self.ee_z_min
        if ee_below:
            reward += self.ee_below_penalty
            self._ee_below_steps += 1

        limit_excess = self._joint_limit_excess(q)
        excess_sq = float(np.sum(limit_excess ** 2))
        reward -= self.joint_limit_penalty_coeff * excess_sq
        self._sum_limit_excess += excess_sq

        terminated = self.dwell_count >= self.dwell_steps
        if terminated:
            reward += self.dwell_complete_bonus

        truncated = self.step_count >= self.max_steps

        self._prev_dist = dist

        info = {
            "waypoint_idx": self.waypoint_idx,
            "joint_dist": dist,
            "in_target": in_target,
            "dwell_count": self.dwell_count,
            "ee_z": float(ee_pos[2]),
            "limit_excess_sq": excess_sq,
            "task_name": self.TASK_NAME,
        }
        if terminated or truncated:
            info["reached"] = bool(terminated)
            info["ee_below_ratio"] = self._ee_below_steps / self.step_count
            info["mean_limit_excess_sq"] = self._sum_limit_excess / self.step_count

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), float(reward), terminated, truncated, info
