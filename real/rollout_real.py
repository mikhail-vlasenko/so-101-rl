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
import csv
import signal
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3 import PPO


from .twin.constants import (
    MAX_RAW_DELTA_PER_STEP,
    SERVO_ACCEL,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from .twin.mapping import JOINT_NAMES, compute_ee_pos, load_joint_maps, rad_to_raw, raw_to_rad
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
DEFAULT_CAL = REPO_ROOT.parent / "feetech-servo-sdk" / "calibration.json"

def _load_reach_cfg() -> tuple[dict, int]:
    """Compose the full Hydra config with env=reach so ${action_scale} (and any
    other interpolations) resolves to its top-level value in config.yaml.

    Returns (reach_env_cfg, prev_actions_n) — prev_actions_n is a top-level
    config field, must match what the trained policy expects."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["env=reach"])
    return OmegaConf.to_container(cfg.reach_env, resolve=True), int(cfg.prev_actions_n)


def parse_args(reach_cfg: dict) -> argparse.Namespace:
    n_waypoints = len(reach_cfg["waypoints"])
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logs/ppo_reach/final_model.zip")
    p.add_argument("--waypoint", type=int, default=0,
                   help=f"Which of the {n_waypoints} fixed waypoints to target (0-indexed)")
    p.add_argument("--execute", action="store_true",
                   help="Actually send servo commands. Default: dry-run (read-only).")
    p.add_argument("--max-steps", type=int, default=int(reach_cfg["max_steps"]))
    p.add_argument("--ema-alpha", type=float, default=1.0,
                   help="EMA smoothing on policy action: a_t = alpha*a + (1-alpha)*a_{t-1}. "
                        "1.0 = off (default), lower = smoother but laggier. Try 0.3-0.5 to "
                        "damp shaking from policy chatter.")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    return p.parse_args()


def write_csv(out_path: Path, steps: np.ndarray, actions: np.ndarray,
              qpos: np.ndarray, ee: np.ndarray, dist: np.ndarray,
              tolerance: float, control_hz: float) -> None:
    header = (["step", "t_s"]
              + [f"action_{n}" for n in JOINT_NAMES]
              + [f"qpos_{n}" for n in JOINT_NAMES]
              + ["ee_x", "ee_y", "ee_z", "dist", "in_target"])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, s in enumerate(steps):
            w.writerow([int(s), f"{s / control_hz:.4f}",
                        *(f"{a:.6f}" for a in actions[i]),
                        *(f"{q:.6f}" for q in qpos[i]),
                        f"{ee[i, 0]:.6f}", f"{ee[i, 1]:.6f}", f"{ee[i, 2]:.6f}",
                        f"{dist[i]:.6f}", int(dist[i] < tolerance)])


def plot_rollout(steps: np.ndarray, actions: np.ndarray, qpos: np.ndarray,
                 ee: np.ndarray, dist: np.ndarray, target_qpos: np.ndarray,
                 target_ee: np.ndarray, tolerance: float, waypoint_idx: int,
                 control_hz: float, out_path: Path) -> None:
    t = steps / control_hz  # seconds
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Rollout — waypoint {waypoint_idx} ({control_hz:.1f} Hz, "
                 f"{len(steps)} steps)")

    ax = axes[0, 0]
    for j, name in enumerate(JOINT_NAMES):
        ax.plot(t, actions[:, j], label=name)
    ax.set_title("Action per joint (clipped, smoothed)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("action ∈ [-1, 1]")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    cmap = plt.get_cmap("tab10")
    for j, name in enumerate(JOINT_NAMES):
        color = cmap(j)
        ax.plot(t, qpos[:, j], color=color, label=name)
        ax.axhline(target_qpos[j], color=color, ls="--", lw=1.0, alpha=0.6)
    ax.set_title("qpos per joint (dashed = target)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for k, name in enumerate("xyz"):
        color = cmap(k)
        ax.plot(t, ee[:, k], color=color, label=f"ee_{name}")
        ax.axhline(target_ee[k], color=color, ls="--", lw=1.0, alpha=0.6)
    ax.set_title("End-effector position (dashed = target)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, dist, color="C3")
    ax.axhline(tolerance, color="k", ls="--", lw=1.0, alpha=0.6,
               label=f"tolerance={tolerance}")
    ax.set_title("‖qpos − target_qpos‖")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_obs(model: mujoco.MjModel, data: mujoco.MjData, qposadr: np.ndarray,
              ee_site_id: int, qpos: np.ndarray, qvel: np.ndarray,
              waypoint_idx: int, n_waypoints: int,
              prev_actions: np.ndarray) -> np.ndarray:
    ee_pos = compute_ee_pos(model, data, qposadr, qpos, ee_site_id)
    onehot = np.zeros(n_waypoints, dtype=np.float32)
    onehot[waypoint_idx] = 1.0
    return np.concatenate([qpos.astype(np.float32),
                           qvel.astype(np.float32),
                           ee_pos.astype(np.float32),
                           onehot,
                           prev_actions.flatten().astype(np.float32)]).astype(np.float32)


def main() -> int:
    reach_cfg, prev_actions_n = _load_reach_cfg()
    waypoints = np.array(reach_cfg["waypoints"], dtype=np.float64)
    n_waypoints, n_joints_cfg = waypoints.shape
    assert n_joints_cfg == 6, f"waypoints must have 6 joints, got {n_joints_cfg}"
    action_scale = float(reach_cfg["action_scale"])
    tolerance = float(reach_cfg["tolerance"])
    dwell_steps = int(reach_cfg["dwell_steps"])
    n_substeps = int(reach_cfg["n_substeps"])

    args = parse_args(reach_cfg)
    assert 0 <= args.waypoint < n_waypoints
    assert 0.0 < args.ema_alpha <= 1.0, f"--ema-alpha must be in (0, 1], got {args.ema_alpha}"

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    # Control period is (timestep * n_substeps) — same definition the sim env uses.
    control_dt = float(model.opt.timestep) * n_substeps
    control_hz = 1.0 / control_dt
    jm = load_joint_maps(model, Path(args.cal))
    qposadr = jm.qposadr()
    xml_low, xml_high = jm.xml_low(), jm.xml_high()
    direction = np.ones(len(jm.items), dtype=np.int8)  # verified via twin: no inversions needed
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_site_id >= 0, "site 'gripperframe' not found in model"

    target_qpos_goal = waypoints[args.waypoint]
    target_ee_goal = compute_ee_pos(model, data, qposadr, target_qpos_goal, ee_site_id)

    policy = PPO.load(args.model)
    expected_obs = 2 * 6 + 3 + n_waypoints + prev_actions_n * 6
    assert policy.observation_space.shape[0] == expected_obs, (
        f"Obs dim mismatch: model expects {policy.observation_space.shape}, "
        f"reach env produces {expected_obs}-dim."
    )

    print(f"waypoint={args.waypoint} execute={args.execute} "
          f"{control_hz:.1f}Hz action_scale={action_scale} ema={args.ema_alpha}")

    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()

    stopped = {"flag": False}
    def stop(_sig, _frame) -> None:
        stopped["flag"] = True
    signal.signal(signal.SIGINT, stop)

    try:
        raw0 = bus.read_all()
        qpos = raw_to_rad(raw0, jm, direction)
        qvel = np.zeros(6, dtype=np.float64)

        slack = 0.05
        if not (np.all(qpos >= xml_low - slack) and np.all(qpos <= xml_high + slack)):
            print("ABORT: initial qpos outside MuJoCo joint range; check calibration.")
            return 1

        if args.execute:
            bus.set_position_kp(SERVO_POSITION_KP)
            bus.enable_torque_all()

        dt = control_dt
        dwell_count = 0
        step = 0
        t_loop = time.time()
        prev_raw_target = raw0.copy()
        prev_actions = np.zeros((prev_actions_n, 6), dtype=np.float32)
        action_ema: np.ndarray | None = None

        log_steps: list[int] = []
        log_actions: list[np.ndarray] = []
        log_qpos: list[np.ndarray] = []
        log_ee: list[np.ndarray] = []
        log_dist: list[float] = []

        while not stopped["flag"] and step < args.max_steps:
            obs = build_obs(model, data, qposadr, ee_site_id, qpos, qvel,
                            args.waypoint, n_waypoints, prev_actions)
            action, _ = policy.predict(obs, deterministic=True)
            action = np.clip(action.astype(np.float64), -1.0, 1.0)
            if action_ema is None:
                action_ema = action.copy()
            else:
                action_ema = args.ema_alpha * action + (1.0 - args.ema_alpha) * action_ema
            action = action_ema
            # Mirror sim: the (clipped, smoothed) action becomes the most recent
            # entry in the prev-actions buffer used for the NEXT obs.
            if prev_actions_n > 0:
                if prev_actions_n > 1:
                    prev_actions[:-1] = prev_actions[1:]
                prev_actions[-1] = action.astype(np.float32)

            target_qpos = np.clip(qpos + action * action_scale, xml_low, xml_high)
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
            in_target = dist < tolerance
            dwell_count = dwell_count + 1 if in_target else 0
            ee = compute_ee_pos(model, data, qposadr, qpos, ee_site_id)

            log_steps.append(step)
            log_actions.append(action.copy())
            log_qpos.append(qpos.copy())
            log_ee.append(ee.copy())
            log_dist.append(dist)

            if step % 20 == 0:
                print(f"step={step:3d}  dist={dist:.3f}")
            step += 1

            if dwell_count >= dwell_steps:
                print(f"REACHED waypoint {args.waypoint} at step {step} (dist={dist:.3f})")
                break
        else:
            if not stopped["flag"]:
                print(f"TIMEOUT at step {step} (dist={dist:.3f})")
    finally:
        bus.close()

    if log_steps:
        out_dir = REPO_ROOT / "rollouts"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rollout_w{args.waypoint}_{int(time.time())}"
        csv_path = out_dir / f"{stem}.csv"
        plot_path = out_dir / f"{stem}.png"
        steps_arr = np.array(log_steps)
        actions_arr = np.stack(log_actions)
        qpos_arr = np.stack(log_qpos)
        ee_arr = np.stack(log_ee)
        dist_arr = np.array(log_dist)
        write_csv(csv_path, steps_arr, actions_arr, qpos_arr, ee_arr,
                  dist_arr, tolerance, control_hz)
        plot_rollout(
            steps=steps_arr,
            actions=actions_arr,
            qpos=qpos_arr,
            ee=ee_arr,
            dist=dist_arr,
            target_qpos=target_qpos_goal,
            target_ee=target_ee_goal,
            tolerance=tolerance,
            waypoint_idx=args.waypoint,
            control_hz=control_hz,
            out_path=plot_path,
        )
        print(f"saved {csv_path.relative_to(REPO_ROOT)} {plot_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
