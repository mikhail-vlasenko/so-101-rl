"""Gymnasium environment: SO-101 arm cube lifting task.

Simpler than pick-and-place — agent learns to grasp and lift a cube.
Terminates when cube reaches target height.
"""

import numpy as np

from base_env import SO101BaseEnv


# Reward constants
TIME_PENALTY = -0.05
EE_CUBE_COEFF = -0.5
GRASP_HOLD_REWARD = 0.05
HEIGHT_PROGRESS_COEFF = 200.0


class SO101LiftEnv(SO101BaseEnv):
    """Grasp a cube and lift it to a target height."""

    XML_PATH = "so101/scene_lift.xml"
    TASK_ID = 0.0
    TASK_NAME = "lift"

    def _parse_config(self, cfg):
        self.target_height = float(cfg["target_height"])

    def _obs_extra(self):
        return [0.0, self.TASK_ID]

    def _on_reset(self, cube_pos):
        self._prev_cube_z = cube_pos[2]

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        reward = TIME_PENALTY
        reward += EE_CUBE_COEFF * ee_cube_dist

        if grasped:
            reward += GRASP_HOLD_REWARD

        height_delta = cube_pos[2] - self._prev_cube_z
        reward += HEIGHT_PROGRESS_COEFF * height_delta
        self._prev_cube_z = cube_pos[2]

        if floor_contact:
            reward += self.floor_contact_penalty

        terminated = cube_pos[2] >= self.target_height

        info = {
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
            "cube_height": cube_pos[2],
        }
        return reward, terminated, info
