"""Gymnasium environment: SO-101 arm cube lifting task.

Simpler than pick-and-place — agent learns to grasp and lift a cube.
Terminates when cube reaches target height.
"""

import numpy as np

from src.base_env import SO101BaseEnv


# Reward constants
TIME_PENALTY = -0.05
EE_CUBE_COEFF = -0.5
GRASP_HOLD_REWARD = 0.15         # a static grasp must strictly beat the pre-grasp shaping rungs
HEIGHT_PROGRESS_COEFF = 200.0    # credited only while grasped
# Contact-quality bridge reach -> grasp (the gradient out of the local optima).
# Both rungs are gated on real cube↔jaw contact so the bonus can't be farmed by
# shoving the sponge with a closed gripper near it (which is what proximity-only
# shaping produced). Only horizontal flinging is penalized; gentle grasp contact
# is free.
JAW_CONTACT_REWARD = 0.05        # one+ gripper jaw touching the cube, pre-grasp
GRIPPER_CLOSE_COEFF = 0.05       # per unit closedness, only once BOTH jaws straddle the cube
CUBE_MOTION_COEFF = -1.0         # per m/s of horizontal cube speed past the deadzone, pre-grasp
CUBE_MOTION_DEADZONE = 0.05      # m/s; cube jitter below this isn't penalized


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

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        reward = TIME_PENALTY
        reward += EE_CUBE_COEFF * ee_cube_dist

        if grasped:
            reward += GRASP_HOLD_REWARD
            height_delta = cube_pos[2] - self._prev_cube_pos[2]
            reward += HEIGHT_PROGRESS_COEFF * height_delta
        else:
            horiz_speed = np.linalg.norm((cube_pos - self._prev_cube_pos)[:2]) / self._step_dt
            reward += CUBE_MOTION_COEFF * max(0.0, horiz_speed - CUBE_MOTION_DEADZONE)
            n_jaw = self._n_jaw_contacts()
            if n_jaw >= 1:
                reward += JAW_CONTACT_REWARD
            if n_jaw == 2:
                reward += GRIPPER_CLOSE_COEFF * self._gripper_closedness()

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
