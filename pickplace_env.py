"""Gymnasium environment: SO-101 arm pick-and-place task."""

import numpy as np
import mujoco

from base_env import SO101BaseEnv


# Reward constants
TIME_PENALTY = -0.05
EE_CUBE_COEFF = -0.5
EE_CUBE_MIN_DIST = 0.025  # cube half-edge + margin; closer than this gets no extra reward
XY_PROGRESS_COEFF = 200.0
GRASP_HOLD_REWARD = 0.05
PLACE_BONUS = 10.0


class SO101PickPlaceEnv(SO101BaseEnv):
    """Pick up a cube and place it at a target location."""

    XML_PATH = "so101/scene_pickplace.xml"
    TASK_ID = 1.0
    TASK_NAME = "pickplace"

    def _parse_config(self, cfg):
        self.place_target = np.array(cfg["place_target"])
        self.place_target_radius = float(cfg["place_target_radius"])
        self.place_target_height = float(cfg["place_target_height"])
        self.ring_center = np.array(cfg["ring_center"])
        self.ring_exclusion_radius = float(cfg["ring_exclusion_radius"])
        self.ring_height_min = float(cfg["ring_height_min"])
        self.ring_height_max = float(cfg["ring_height_max"])
        self.ring_contact_penalty = float(cfg["ring_contact_penalty"])

        # Ring body ID for randomizing height
        self.ring_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ring")
        assert self.ring_body_id >= 0, "Ring body not found in XML"
        self.ring_wall_height = 0.1  # must match geom half-height in scene XML (0.05 * 2)
        self.ring_height = self.ring_height_max

        self.ring_geom_ids = set()
        for i in range(6):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ring_wall_{i}")
            assert gid >= 0, f"Ring wall geom 'ring_wall_{i}' not found in XML"
            self.ring_geom_ids.add(gid)

    def _obs_extra(self, cube_pos):
        cube_to_target = cube_pos[:2] - self.place_target[:2]
        return [cube_to_target[0], cube_to_target[1], self.ring_height, self.TASK_ID]

    def _sample_cube_pos(self):
        while True:
            cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
            dist_to_ring = np.linalg.norm(cube_pos[:2] - self.ring_center)
            if dist_to_ring > self.ring_exclusion_radius:
                return cube_pos

    def _on_reset(self, cube_pos):
        # Random ring height: sink the ring body into the floor
        self.ring_height = self.np_random.uniform(self.ring_height_min, self.ring_height_max)
        ring_body_pos = self.model.body_pos[self.ring_body_id].copy()
        ring_body_pos[2] = self.ring_height - self.ring_wall_height
        self.model.body_pos[self.ring_body_id] = ring_body_pos

        self._prev_xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
        self._xy_progress_total = 0.0
        self._xy_regress_total = 0.0
        self._ring_contact_steps = 0

    def _has_ring_contact(self):
        """Check if any arm/gripper geom is in contact with a ring wall."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.ring_geom_ids and g2 in self.arm_geom_ids:
                return True
            if g2 in self.ring_geom_ids and g1 in self.arm_geom_ids:
                return True
        return False

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        ring_contact = self._has_ring_contact()
        if ring_contact:
            self._ring_contact_steps += 1
        reward = self._compute_reward(ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact, ring_contact)

        # Terminate with bonus when cube is placed in the target region
        xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
        placed = xy_dist < self.place_target_radius and cube_pos[2] < self.place_target_height
        terminated = False
        if placed:
            reward += PLACE_BONUS
            terminated = True

        info = {
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
            "placed": placed,
        }
        return reward, terminated, info

    def _on_episode_end(self, info):
        info["xy_progress"] = self._xy_progress_total
        info["xy_regress"] = self._xy_regress_total
        info["ring_contact_ratio"] = self._ring_contact_steps / self.step_count

    def _compute_reward(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact, ring_contact):
        reward = TIME_PENALTY

        if floor_contact:
            reward += self.floor_contact_penalty

        if ring_contact:
            reward += self.ring_contact_penalty

        # XY progress reward only counts while grasped — forces "carry, not push"
        xy_dist = np.linalg.norm(cube_pos[:2] - self.place_target[:2])
        xy_delta = self._prev_xy_dist - xy_dist  # positive = moved closer
        self._prev_xy_dist = xy_dist

        if grasped:
            xy_reward = XY_PROGRESS_COEFF * xy_delta
            if xy_delta >= 0:
                self._xy_progress_total += xy_reward
            else:
                self._xy_regress_total += xy_reward
            reward += xy_reward
            reward += GRASP_HOLD_REWARD
        else:
            # Approach reward, clipped so the agent isn't rewarded for crashing into the cube
            reward += EE_CUBE_COEFF * max(ee_cube_dist, EE_CUBE_MIN_DIST)

        return reward
