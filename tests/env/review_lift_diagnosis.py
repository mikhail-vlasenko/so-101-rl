"""Manual, read-only policy diagnosis; explicitly run with pytest -s.

Required env vars: LIFT_DIAG_MODEL, LIFT_DIAG_VARIANT (normal/pose_clean/half_step),
LIFT_DIAG_OUTPUT. Uses the continuation's saved config and fixed 128 validation
starts. pose_clean removes pose noise/bias, not latency, occlusion, surface bias,
cloud noise or physical randomization. half_step halves the configured action
scale, including the shared raw clamp. These are inference ablations, not retraining.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from src.bps import validate_checkpoint_bps
from src.lift_env import (
    SO101LiftEnv, GRASP_HOLD_REWARD, HEIGHT_PROGRESS_COEFF,
    LIFT_BONUS,
)
from src.train import make_lr_schedule, runtime_cfg_from_hydra


def historical_config():
    cfg = OmegaConf.load("outputs/table_slide_continue_30m_20260916/.hydra/config.yaml")
    # This audit reproduces the old reward, before distance became stage-owned.
    cfg.lift_env.ee_cube_coeff = -0.5
    cfg.lift_env.jaw_contact_reward = 0.05
    cfg.lift_env.gripper_close_coeff = 0.05
    return cfg


def test_reward_and_schedule_budget():
    cfg = historical_config()
    # At 10 cm, distance still charges five times the reduced explicit time cost.
    assert abs(cfg.lift_env.ee_cube_coeff * 0.10) == pytest.approx(5 * abs(cfg.lift_env.time_penalty))
    assert cfg.train.gamma**15 == pytest.approx(0.8600583546)
    # Increasing gamma alone would invalidate the finish-vs-hold safety margin.
    assert (GRASP_HOLD_REWARD + cfg.lift_env.time_penalty) / (1 - .997) > LIFT_BONUS
    # A step-linear schedule scarcely advances under the current wall-time cap.
    completed_steps = 3599136
    progress = 1 - completed_steps / cfg.train.total_timesteps
    assert progress > .90
    schedule = make_lr_schedule("linear", 1e-4, 1e-5)
    assert schedule(progress) > .9e-4
    # Merely switching the existing config to linear would INCREASE the LR:
    # its inherited lr_min is 3e-4 while this refinement's initial LR is 1e-4.
    assert cfg.train.lr_min > cfg.train.learning_rate


class RewardAuditEnv(SO101LiftEnv):
    """Reconstruct component totals and assert their sum against production reward."""

    def _compute_step(self, ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact):
        terms = {
            "time": self.time_penalty,
            "distance": self.ee_cube_coeff * ee_cube_dist,
            "motion": self.cube_motion_coeff * self._table_slide_excess_distance / self._step_dt,
            "hold": GRASP_HOLD_REWARD if grasped else 0.0,
            "height": HEIGHT_PROGRESS_COEFF * (cube_pos[2] - self._prev_cube_pos[2]) if grasped else 0.0,
            "contact": 0.0, "poke": 0.0, "tip": 0.0,
            "floor_binary": self.floor_contact_penalty if floor_contact else 0.0,
            "floor_force": self.floor_force_coeff * self._arm_floor_contact_force(),
            "success": LIFT_BONUS if grasped and cube_pos[2] >= self.target_height else 0.0,
        }
        if not grasped:
            jaws = self._n_jaw_contacts()
            if jaws >= 1:
                terms["contact"] += self.jaw_contact_reward
            if jaws == 2 and self._has_opposed_gripper_contact():
                terms["contact"] += self.gripper_close_coeff * self._gripper_closedness()
            terms["poke"] = self.poke_force_coeff * self._arm_cube_contact_force()
            terms["tip"] = self.cube_tip_coeff * self._cube_angular_speed()
        if self._min_arm_floor_dist(self.floor_proximity_thresh) < self.floor_proximity_thresh:
            terms["floor_binary"] += self.floor_proximity_penalty
        reward, terminated, info = super()._compute_step(
            ee_pos, cube_pos, ee_cube_dist, grasped, floor_contact)
        assert reward == pytest.approx(sum(terms.values()), abs=1e-9)
        info["reward_terms"] = terms
        return reward, terminated, info


def test_policy_diagnosis():
    if "LIFT_DIAG_MODEL" not in os.environ:
        pytest.skip("Manual policy diagnosis requires LIFT_DIAG_* env vars")
    model_path = Path(os.environ["LIFT_DIAG_MODEL"])
    output = Path(os.environ["LIFT_DIAG_OUTPUT"])
    variant = os.environ["LIFT_DIAG_VARIANT"]
    assert variant in ("normal", "pose_clean", "half_step")
    assert not output.exists()
    cfg = historical_config()
    if variant == "pose_clean":
        # Keep the same dictionary shapes and RNG draws; only noise amplitudes change.
        for key in cfg.obs_noise:
            if key != "tag_depth_factor":
                cfg.obs_noise[key] = 0.0
        for key in cfg.obs_bias:
            cfg.obs_bias[key] = 0.0
    elif variant == "half_step":
        cfg.action_scale *= 0.5
    reference = json.loads(Path(
        "outputs/table_slide_continue_30m_20260916/parent.json").read_text())
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = RewardAuditEnv(env_cfg=cfg.lift_env, cfg=runtime_cfg_from_hydra(cfg))
    rows = []
    try:
        policy = PPO.load(model_path, device="cpu")
        validate_checkpoint_bps(policy, env.bps_config)
        for reference_row in reference["episodes"]:
            seed = reference_row["seed"]
            obs, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(env.data.qpos, reference_row["start_qpos"])
            initial_height = env._get_cube_pos()[2]
            max_height = initial_height
            totals = {}
            discounted = {}
            saturated = []
            live_errors = []
            bps_ages = []
            near_angles = []
            first_contact = None
            grasp_steps = 0
            contact_steps = 0
            post_touch_no_grasp = 0
            slide = 0.0
            for step in range(env.max_steps):
                cube = env._get_cube_pos()
                dist = np.linalg.norm(env._get_ee_pos() - cube)
                if dist < 0.08:
                    live, _ = env._obj.serve(env.data.time)
                    live_errors.append((live - cube).tolist())
                    bps_ages.append(env._bps_state.serve(env.data.time).age_s)
                    if first_contact is None:
                        near_angles.append(float(env.data.qpos[env.joint_qposadr[env.gripper_idx]]))
                with torch.no_grad():
                    tensor, _ = policy.policy.obs_to_tensor(obs)
                    mean = policy.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]
                saturated.append((np.abs(mean) >= 1).astype(float))
                obs, reward, term, trunc, info = env.step(np.clip(mean, -1, 1))
                assert np.isfinite(reward) and np.all(np.isfinite(obs))
                grasp_steps += int(info["grasped"])
                touch = env._n_jaw_contacts() > 0
                contact_steps += int(touch)
                if touch and first_contact is None:
                    first_contact = step
                post_touch_no_grasp += int(first_contact is not None and not info["grasped"])
                max_height = max(max_height, env._get_cube_pos()[2])
                slide += info["table_slide_distance_m"]
                for key, value in info["reward_terms"].items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                    discounted[key] = discounted.get(key, 0.0) + cfg.train.gamma**step * float(value)
                if term or trunc:
                    break
            rows.append({
                "seed": seed, "orientation": reference_row["orientation"],
                "success": bool(info["lift_success"]), "steps": step + 1,
                "terms": totals, "discounted_terms": discounted,
                "slide_m": slide, "grasp_steps": grasp_steps, "contact_steps": contact_steps,
                "post_touch_no_grasp_steps": post_touch_no_grasp,
                "height_credit_excess": totals["height"] - HEIGHT_PROGRESS_COEFF * (max_height - initial_height),
                "saturated_mean_fraction_per_joint": np.mean(saturated, axis=0).tolist(),
                "near_live_errors_m": live_errors, "near_bps_ages_s": bps_ages,
                "near_precontact_angles_rad": near_angles,
            })
            print(f"{variant} {len(rows)}/128 success={info['lift_success']}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump({"model": str(model_path), "variant": variant, "episodes": rows}, handle, indent=2)
    finally:
        env.close()
        torch.set_num_threads(old_threads)
