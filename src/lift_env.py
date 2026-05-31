"""Gymnasium environment: SO-101 arm cube lifting task.

Simpler than pick-and-place — agent learns to grasp and lift a cube.
Terminates when cube reaches target height.
"""

import numpy as np
import mujoco

from src.base_env import SO101BaseEnv


# Reward constants
TIME_PENALTY = -0.05
EE_CUBE_COEFF = -0.5
GRASP_HOLD_REWARD = 0.05
HEIGHT_PROGRESS_COEFF = 200.0   # credited only while grasped
CONTACT_FORCE_COEFF = -0.002    # per N of arm↔cube normal force, pre-grasp only
CUBE_MOTION_COEFF = -2.0        # per m/s of cube speed, pre-grasp only


class SO101LiftEnv(SO101BaseEnv):
    """Grasp a cube and lift it to a target height."""

    XML_PATH = "so101/scene_lift.xml"
    TASK_ID = 0.0
    TASK_NAME = "lift"

    def _parse_config(self, cfg):
        self.target_height = float(cfg["target_height"])
        self.floor_proximity_thresh = float(cfg["floor_proximity_thresh"])
        self.floor_proximity_penalty = float(cfg["floor_proximity_penalty"])

    def _obs_extra(self, cube_pos):
        return [0.0, 0.0, 0.0, self.TASK_ID]

    def _on_reset(self, cube_pos):
        self._prev_cube_pos = cube_pos.copy()

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

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        reward = TIME_PENALTY
        reward += EE_CUBE_COEFF * ee_cube_dist

        if grasped:
            reward += GRASP_HOLD_REWARD
            height_delta = cube_pos[2] - self._prev_cube_pos[2]
            reward += HEIGHT_PROGRESS_COEFF * height_delta
        else:
            cube_speed = np.linalg.norm(cube_pos - self._prev_cube_pos) / self._step_dt
            reward += CUBE_MOTION_COEFF * cube_speed
            reward += CONTACT_FORCE_COEFF * self._arm_cube_contact_force()

        self._prev_cube_pos = cube_pos.copy()

        if floor_contact:
            reward += self.floor_contact_penalty

        if self._min_arm_floor_dist(self.floor_proximity_thresh) < self.floor_proximity_thresh:
            reward += self.floor_proximity_penalty

        terminated = cube_pos[2] >= self.target_height

        info = {
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
            "cube_height": cube_pos[2],
        }
        return reward, terminated, info
