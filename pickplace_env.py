"""Gymnasium environment: SO-101 arm pick-and-place task.

Phases: REACH → PLACE → RETURN.
"""

import enum

import numpy as np
import mujoco

from base_env import SO101BaseEnv


# Reward constants
TIME_PENALTY = -0.05
JOINT_PASSIVE_COEFF = -0.01
EE_CUBE_COEFF = -0.5
XY_PROGRESS_COEFF = 200.0
HEIGHT_MULT_MAX = 2.0
HEIGHT_MULT_CEILING = 0.05
GRASP_HOLD_REWARD = 0.05
PLACE_BONUS = GRASP_HOLD_REWARD * 2
RETURN_BONUS = 10.0
RETURN_THRESHOLD = 0.3


class Phase(enum.IntEnum):
    REACH = 0
    PLACE = 1
    RETURN = 2


class SO101PickPlaceEnv(SO101BaseEnv):
    """Pick up a cube, place it at a target, and return to passive pose."""

    XML_PATH = "so101/scene_pickplace.xml"
    TASK_ID = 1.0
    TASK_NAME = "pickplace"

    def _parse_config(self, cfg):
        self.place_target = np.array(cfg["place_target"])
        self.place_target_radius = float(cfg["place_target_radius"])
        self.place_target_height = float(cfg["place_target_height"])
        self.ring_center = np.array(cfg["ring_center"])
        self.ring_exclusion_radius = float(cfg["ring_exclusion_radius"])
        self.passive_pose = np.array(cfg["passive_pose"])
        self.ring_height_max = float(cfg["ring_height_max"])

        # Ring body ID for randomizing height
        self.ring_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ring")
        assert self.ring_body_id >= 0, "Ring body not found in XML"
        self.ring_wall_height = 0.1  # must match geom half-height in scene XML (0.05 * 2)
        self.ring_height = self.ring_height_max

    def _obs_extra(self):
        return [self.ring_height, self.TASK_ID]

    def _sample_cube_pos(self):
        while True:
            cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
            dist_to_ring = np.linalg.norm(cube_pos[:2] - self.ring_center)
            if dist_to_ring > self.ring_exclusion_radius:
                return cube_pos

    def _on_reset(self, cube_pos):
        # Random ring height: sink the ring body into the floor
        self.ring_height = self.np_random.uniform(0.0, self.ring_height_max)
        ring_body_pos = self.model.body_pos[self.ring_body_id].copy()
        ring_body_pos[2] = self.ring_height - self.ring_wall_height
        self.model.body_pos[self.ring_body_id] = ring_body_pos

        self.phase = Phase.REACH
        self.max_phase = Phase.REACH
        self._bonuses_collected = set()
        self._prev_xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
        self._xy_progress_total = 0.0
        self._xy_regress_total = 0.0

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        joint_pos = self._get_joint_pos()

        # Phase transitions
        self._update_phase(grasped, cube_pos)
        self.max_phase = max(self.max_phase, self.phase)

        # Reward
        reward = self._compute_reward(ee_pos, cube_pos, joint_pos, ee_cube_dist, grasped, floor_contact)

        # Termination
        terminated = False
        if self.phase == Phase.RETURN:
            joint_dist = np.linalg.norm(joint_pos - self.passive_pose)
            if joint_dist < RETURN_THRESHOLD:
                reward += RETURN_BONUS
                terminated = True

        info = {
            "phase": self.phase.name,
            "max_phase": int(self.max_phase),
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
        }
        if self.floor_contact_penalty:
            info["floor_contact"] = floor_contact

        return reward, terminated, info

    def _on_episode_end(self, info):
        info["xy_progress"] = self._xy_progress_total
        info["xy_regress"] = self._xy_regress_total

    def _update_phase(self, grasped, cube_pos):
        if self.phase == Phase.REACH:
            if grasped:
                self.phase = Phase.PLACE
        elif self.phase == Phase.PLACE:
            if not grasped:
                self.phase = Phase.REACH
            else:
                xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
                if xy_dist < self.place_target_radius and cube_pos[2] < self.place_target_height:
                    self.phase = Phase.RETURN

    def _bonus(self, name, amount):
        """Give a bonus only once per episode."""
        if name in self._bonuses_collected:
            return 0.0
        self._bonuses_collected.add(name)
        return amount

    def _compute_reward(self, ee_pos, cube_pos, joint_pos, ee_cube_dist, grasped, floor_contact):
        reward = TIME_PENALTY

        if floor_contact:
            reward += self.floor_contact_penalty

        joint_dist = np.linalg.norm(joint_pos - self.passive_pose)
        reward += JOINT_PASSIVE_COEFF * joint_dist

        # XY progress reward: positive for moving toward target, 2x negative for moving away
        xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
        xy_delta = self._prev_xy_dist - xy_dist  # positive = moved closer
        self._prev_xy_dist = xy_dist

        # Height multiplier: 1.0 at ground, 2.0 at HEIGHT_MULT_CEILING
        height_frac = np.clip(cube_pos[2] / HEIGHT_MULT_CEILING, 0.0, 1.0)
        height_mult = 1.0 + (HEIGHT_MULT_MAX - 1.0) * height_frac

        if xy_delta >= 0:
            xy_reward = XY_PROGRESS_COEFF * xy_delta * height_mult
            self._xy_progress_total += xy_reward
        else:
            xy_reward = XY_PROGRESS_COEFF * HEIGHT_MULT_MAX * xy_delta * height_mult
            self._xy_regress_total += xy_reward

        if self.phase == Phase.REACH:
            if grasped:
                reward += GRASP_HOLD_REWARD
            reward += EE_CUBE_COEFF * ee_cube_dist + xy_reward
        elif self.phase == Phase.PLACE:
            if grasped:
                reward += GRASP_HOLD_REWARD
            reward += xy_reward
        elif self.phase == Phase.RETURN:
            reward += PLACE_BONUS

        return reward
