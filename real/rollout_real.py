"""Roll out the trained reach policy on the real SO-101 arm.

Usage:
    python -m real.rollout_real                       # default: waypoint 0, dry-run
    python -m real.rollout_real --execute             # actually send commands
    python -m real.rollout_real --waypoint 2 --execute
    python -m real.rollout_real --model logs/ppo_reach/best_model.zip --execute

Safety:
  - --execute is OFF by default (dry-run prints intended targets, no servo writes)
  - per-step raw-delta is clamped (constants.MAX_RAW_DELTA_PER_STEP)
  - servos use low speed/accel from twin.constants
  - Ctrl-C disables torque cleanly
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO

from .twin.constants import (
    MAX_RAW_DELTA_PER_STEP,
    SERVO_ACCEL,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from .twin.mapping import compute_ee_pos, load_joint_maps, rad_to_raw, raw_to_rad
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
DEFAULT_CAL = REPO_ROOT.parent / "feetech-servo-sdk" / "calibration.json"

# Reach env constants — must match conf/env/reach.yaml.
ACTION_SCALE = 0.02
CONTROL_HZ = 20
N_WAYPOINTS = 4
MAX_STEPS = 400
TOLERANCE = 0.05
DWELL_STEPS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logs/ppo_reach/final_model.zip")
    p.add_argument("--waypoint", type=int, default=0,
                   help=f"Which of the {N_WAYPOINTS} fixed waypoints to target (0-indexed)")
    p.add_argument("--execute", action="store_true",
                   help="Actually send servo commands. Default: dry-run (read-only).")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    return p.parse_args()


def build_obs(model: mujoco.MjModel, data: mujoco.MjData, qposadr: np.ndarray,
              ee_site_id: int, qpos: np.ndarray, qvel: np.ndarray,
              waypoint_idx: int) -> np.ndarray:
    ee_pos = compute_ee_pos(model, data, qposadr, qpos, ee_site_id)
    onehot = np.zeros(N_WAYPOINTS, dtype=np.float32)
    onehot[waypoint_idx] = 1.0
    return np.concatenate([qpos.astype(np.float32),
                           qvel.astype(np.float32),
                           ee_pos.astype(np.float32),
                           onehot]).astype(np.float32)


def main() -> int:
    args = parse_args()
    assert 0 <= args.waypoint < N_WAYPOINTS

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    qposadr = jm.qposadr()
    xml_low, xml_high = jm.xml_low(), jm.xml_high()
    direction = np.ones(len(jm.items), dtype=np.int8)  # verified via twin: no inversions needed
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_site_id >= 0, "site 'gripperframe' not found in model"

    with open(REPO_ROOT / "conf" / "env" / "reach.yaml") as f:
        reach_cfg = yaml.safe_load(f)["reach_env"]
    waypoints = np.array(reach_cfg["waypoints"], dtype=np.float64)
    assert waypoints.shape == (N_WAYPOINTS, 6)
    target_qpos_goal = waypoints[args.waypoint]
    print(f"Waypoint {args.waypoint} target qpos: {np.round(target_qpos_goal, 3).tolist()}")

    print(f"Loading model: {args.model}")
    policy = PPO.load(args.model)
    expected_obs = 2 * 6 + 3 + N_WAYPOINTS
    assert policy.observation_space.shape[0] == expected_obs, (
        f"Obs dim mismatch: model expects {policy.observation_space.shape}, "
        f"reach env produces {expected_obs}-dim."
    )

    print(f"Waypoint index: {args.waypoint}")
    print(f"Execute: {args.execute}  (dry-run does NOT send servo commands)")
    print(f"Control rate: {CONTROL_HZ} Hz, max steps: {args.max_steps}")

    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()

    stopped = {"flag": False}
    def stop(_sig, _frame) -> None:
        print("\nStopping...")
        stopped["flag"] = True
    signal.signal(signal.SIGINT, stop)

    try:
        raw0 = bus.read_all()
        qpos = raw_to_rad(raw0, jm, direction)
        qvel = np.zeros(6, dtype=np.float64)
        print(f"Initial raw:  {raw0.tolist()}")
        print(f"Initial qpos: {np.round(qpos, 3).tolist()}")
        ee0 = compute_ee_pos(model, data, qposadr, qpos, ee_site_id)
        print(f"Initial EE:   {np.round(ee0, 3).tolist()}")

        slack = 0.05
        if not (np.all(qpos >= xml_low - slack) and np.all(qpos <= xml_high + slack)):
            print("WARNING: initial qpos is outside MuJoCo joint range. "
                  "Calibration may need adjusting. Aborting.")
            return 1

        if args.execute:
            bus.set_position_kp(SERVO_POSITION_KP)
            print(f"Set position-loop Kp (per joint) = {list(SERVO_POSITION_KP)}")
            bus.enable_torque_all()

        dt = 1.0 / CONTROL_HZ
        dwell_count = 0
        step = 0
        t_loop = time.time()
        prev_raw_target = raw0.copy()

        while not stopped["flag"] and step < args.max_steps:
            obs = build_obs(model, data, qposadr, ee_site_id, qpos, qvel, args.waypoint)
            action, _ = policy.predict(obs, deterministic=True)
            action = np.clip(action.astype(np.float64), -1.0, 1.0)

            target_qpos = np.clip(qpos + action * ACTION_SCALE, xml_low, xml_high)
            target_raw = rad_to_raw(target_qpos, jm, direction)

            # Safety: clamp per-step raw delta vs. last commanded
            delta = np.clip(target_raw - prev_raw_target,
                            -MAX_RAW_DELTA_PER_STEP, MAX_RAW_DELTA_PER_STEP)
            target_raw = (prev_raw_target + delta).astype(np.int64)

            if args.execute:
                bus.write_all(target_raw, SERVO_SPEED, SERVO_ACCEL)
            prev_raw_target = target_raw

            t_next = t_loop + dt
            sleep_for = t_next - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            t_loop = time.time()

            raw = bus.read_all()
            new_qpos = raw_to_rad(raw, jm, direction)
            qvel = (new_qpos - qpos) / dt
            qpos = new_qpos

            dist = float(np.linalg.norm(qpos - target_qpos_goal))
            in_target = dist < TOLERANCE
            dwell_count = dwell_count + 1 if in_target else 0
            if step % 5 == 0 or in_target:
                ee = compute_ee_pos(model, data, qposadr, qpos, ee_site_id)
                act_str = ",".join(f"{a:+.2f}" for a in action)
                tgt_err = (target_raw - raw).tolist()
                print(f"step={step:3d}  dist={dist:.3f}  in_target={in_target}  "
                      f"dwell={dwell_count}  ee_z={ee[2]:+.3f}  act=[{act_str}]")
                print(f"          raw      ={raw.tolist()}")
                print(f"          tgt_raw  ={target_raw.tolist()}")
                print(f"          err(tgt-raw)={tgt_err}")
            step += 1

            if dwell_count >= DWELL_STEPS:
                print(f"\nREACHED waypoint {args.waypoint} after {step} steps.")
                break
    finally:
        print("Disabling torque.")
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
