"""Gymnasium environment: SO-101 arm cube lifting task.

Simpler than pick-and-place — agent learns to grasp and lift a cube.
Terminates when cube reaches target height.

Shaping penalizes horizontal object speed only during active cube-table contact,
regardless of grasp. Contact and travel are accumulated at physics substeps so
lifting off or landing within a control tick cannot charge airborne transport.
The deadzone applies per substep; integrated excess travel is divided by the
control timestep to retain the coefficient's per-m/s-per-control-tick scale.
Vertical progress and unsupported horizontal transport are free of this term.
The stage-independent time cost is configured separately to allow careful approaches.
Distance, jaw contact and closure are bootstrap shaping only (shaping=none).
In light/full, positive rewards require a proper grasp.
"""

import numpy as np

from src.base_env import SO101BaseEnv


# Reward constants
GRASP_HOLD_REWARD = 0.15         # a static grasp must strictly beat the pre-grasp shaping rungs
HEIGHT_PROGRESS_COEFF = 200.0    # credited only while grasped
# Terminal bonus for a grasped lift to target_height. Holding just under the
# target nets at most +0.14/step with the default time cost (before ee-distance),
# while crossing the last millimeter pays 200 * 0.001 = 0.2 once and ends the
# episode — so without this bonus the optimal policy hovers instead of
# finishing. It must beat the discounted hold annuity: at gamma=0.99 the
# infinite-horizon upper bound is 0.14/(1-0.99) = 14. Re-check if gamma,
# time_penalty or GRASP_HOLD_REWARD changes.
LIFT_BONUS = 15.0


class SO101LiftEnv(SO101BaseEnv):
    """Grasp a cube and lift it to a target height."""

    XML_PATH = "so101/scene_lift.xml"
    TASK_ID = 0.0
    TASK_NAME = "lift"

    def _parse_config(self, cfg):
        self.target_height = float(cfg["target_height"])
        self.time_penalty = float(cfg["time_penalty"])
        assert self.time_penalty <= 0.0, "time_penalty must be nonpositive"
        self.ee_cube_coeff = float(cfg["ee_cube_coeff"])
        assert self.ee_cube_coeff <= 0.0, "ee_cube_coeff must be nonpositive"
        self.jaw_contact_reward = float(cfg["jaw_contact_reward"])
        self.gripper_close_coeff = float(cfg["gripper_close_coeff"])
        assert self.jaw_contact_reward >= 0.0, "jaw_contact_reward must be nonnegative"
        assert self.gripper_close_coeff >= 0.0, "gripper_close_coeff must be nonnegative"
        # The shaping group owns these costs; none disables them for learning
        # to grasp from scratch. Table travel is still measured when its cost is zero.
        self.floor_proximity_thresh = float(cfg["floor_proximity_thresh"])
        self.floor_proximity_penalty = float(cfg["floor_proximity_penalty"])
        self.floor_force_coeff = float(cfg["floor_force_coeff"])
        self.poke_force_coeff = float(cfg["poke_force_coeff"])
        self.cube_tip_coeff = float(cfg["cube_tip_coeff"])
        self.cube_motion_coeff = float(cfg["cube_motion_coeff"])
        self.cube_motion_deadzone = float(cfg["cube_motion_deadzone"])
        assert self.cube_motion_coeff <= 0.0, "cube_motion_coeff must be nonpositive"
        assert self.cube_motion_deadzone >= 0.0, "cube_motion_deadzone must be nonnegative"

    def _obs_extra(self, cube_pos):
        return [0.0, 0.0, 0.0, self.TASK_ID]

    def _on_reset(self, cube_pos):
        self._prev_cube_pos = cube_pos.copy()
        self._motion_prev_cube_xy = cube_pos[:2].copy()
        self._table_slide_distance = 0.0
        self._table_slide_excess_distance = 0.0
        self._episode_table_slide_distance = 0.0
        self._episode_motion_penalty = 0.0

    def _cube_has_table_contact(self):
        """Active cube-floor contact, not arm-floor contact or a height proxy."""
        for contact in self.data.contact:
            if ((contact.geom1 == self.cube_geom_id and contact.geom2 == self.floor_geom_id)
                    or (contact.geom2 == self.cube_geom_id and contact.geom1 == self.floor_geom_id)):
                if contact.efc_address >= 0 and contact.dist <= 0.0:
                    return True
        return False

    def _record_table_motion(self):
        cube_xy = self._get_cube_pos()[:2]
        if self._cube_has_table_contact():
            distance = float(np.linalg.norm(cube_xy - self._motion_prev_cube_xy))
            self._table_slide_distance += distance
            self._table_slide_excess_distance += max(
                0.0, distance - self.cube_motion_deadzone * self.model.opt.timestep)
        self._motion_prev_cube_xy = cube_xy.copy()

    def _on_substep(self):
        # MuJoCo's contacts describe the solve for the just-integrated substep.
        self._record_table_motion()
        super()._on_substep()

    def _on_episode_end(self, info):
        info["episode_table_slide_distance_m"] = self._episode_table_slide_distance
        info["episode_motion_penalty"] = self._episode_motion_penalty

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        reward = self.time_penalty
        reward += self.ee_cube_coeff * ee_cube_dist

        motion_penalty = self.cube_motion_coeff * self._table_slide_excess_distance / self._step_dt
        reward += motion_penalty
        table_slide_distance = self._table_slide_distance
        self._episode_motion_penalty += motion_penalty
        self._episode_table_slide_distance += table_slide_distance
        self._table_slide_distance = 0.0
        self._table_slide_excess_distance = 0.0

        if grasped:
            reward += GRASP_HOLD_REWARD
            height_delta = cube_pos[2] - self._prev_cube_pos[2]
            reward += HEIGHT_PROGRESS_COEFF * height_delta
        else:
            if self.jaw_contact_reward or self.gripper_close_coeff:
                n_jaw = self._n_jaw_contacts()
                if n_jaw >= 1:
                    reward += self.jaw_contact_reward
                # Closure credit requires load on opposing faces, not a corner pinch.
                if self.gripper_close_coeff and n_jaw == 2 and self._has_opposed_gripper_contact():
                    reward += self.gripper_close_coeff * self._gripper_closedness()
            # Gentle approach: penalize hard pokes and rolling the sponge over,
            # pre-grasp only (after grasp the cube rides with the gripper).
            if self.poke_force_coeff:
                reward += self.poke_force_coeff * self._arm_cube_contact_force()
            if self.cube_tip_coeff:
                reward += self.cube_tip_coeff * self._cube_angular_speed()

        self._prev_cube_pos = cube_pos.copy()

        if floor_contact:
            reward += self.floor_contact_penalty

        if self.floor_proximity_penalty and \
                self._min_arm_floor_dist(self.floor_proximity_thresh) < self.floor_proximity_thresh:
            reward += self.floor_proximity_penalty

        # The contact/proximity penalties above are binary, so once the jaws are
        # at the floor (unavoidable for a 1.5 cm sponge) pressing harder is free —
        # policies learn to lean on the floor as a height reference, up to ~20 N
        # in sim and a sustained servo push on the real arm. Penalizing the
        # contact force restores the gradient; applies grasped or not (the
        # observed presses happen during the grasp itself).
        if self.floor_force_coeff:
            reward += self.floor_force_coeff * self._arm_floor_contact_force()

        # Success = a genuine grasped lift to target height. Requiring the grasp
        # (not just the cube center crossing target_height) stops a violent flick
        # or a knock-up from counting as a lift, and keeps mean episode length an
        # honest "time to a clean first-try lift" signal.
        terminated = grasped and cube_pos[2] >= self.target_height
        if terminated:
            reward += LIFT_BONUS

        info = {
            "ee_cube_dist": ee_cube_dist,
            "grasped": grasped,
            "cube_height": cube_pos[2],
            "lift_success": terminated,
            "motion_penalty": motion_penalty,
            "table_slide_distance_m": table_slide_distance,
        }
        return reward, terminated, info
