"""Roll out the trained reach policy on the real SO-101 arm.

Usage:
    python -m real.rollout_real                       # default: waypoint 0, dry-run, latest checkpoint
    python -m real.rollout_real --execute             # actually send commands
    python -m real.rollout_real --waypoint 2 --execute
    python -m real.rollout_real --model best --execute        # best_model.zip
    python -m real.rollout_real --model logs/ppo_reach/best_model.zip --execute
    python -m real.rollout_real --slow 3 --execute    # 1/3 physical speed, no retraining

Setup, safety gating, and per-tick command shaping (training-matched
quantization, raw clamp, sub-target streaming, --slow time dilation) all live
in real.rollout_common — this script owns only the reach-specific observation,
termination, and plots. --execute is OFF by default; Ctrl-C disables torque.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .rollout_common import (
    ArmLoop,
    add_common_args,
    install_sigint_flag,
    load_env_cfg,
    load_policy,
)
from .calibration import load_calibration, load_compliance
from .twin.mapping import JOINT_NAMES, compute_ee_pos, load_joint_maps
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
LOG_DIR = REPO_ROOT / "logs" / "ppo_reach"


def parse_args(reach_cfg: dict) -> argparse.Namespace:
    n_waypoints = len(reach_cfg["waypoints"])
    p = argparse.ArgumentParser()
    p.add_argument("--waypoint", type=int, default=0,
                   help=f"Which of the {n_waypoints} fixed waypoints to target (0-indexed)")
    add_common_args(p, default_xml=DEFAULT_XML,
                    default_max_steps=int(reach_cfg["max_steps"]))
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
    # Reach has no history-tap wiring (src/train.actor_obs_dim_for); the taps
    # entry only exists for the cube tasks.
    reach_cfg, prev_actions_n, _, _ = load_env_cfg("reach")
    waypoints = np.array(reach_cfg["waypoints"], dtype=np.float64)
    n_waypoints, n_joints_cfg = waypoints.shape
    assert n_joints_cfg == 6, f"waypoints must have 6 joints, got {n_joints_cfg}"
    action_scale = float(reach_cfg["action_scale"])
    tolerance = float(reach_cfg["tolerance"])
    dwell_steps = int(reach_cfg["dwell_steps"])
    n_substeps = int(reach_cfg["n_substeps"])

    args = parse_args(reach_cfg)
    assert 0 <= args.waypoint < n_waypoints

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    qposadr = jm.qposadr()
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_site_id >= 0, "site 'gripperframe' not found in model"

    target_qpos_goal = waypoints[args.waypoint]
    target_ee_goal = compute_ee_pos(model, data, qposadr, target_qpos_goal, ee_site_id)

    expected_obs = 2 * 6 + 3 + n_waypoints + prev_actions_n * 6
    policy = load_policy(args.model, LOG_DIR, expected_obs)

    bus = ServoBus(args.port, jm.servo_ids())
    loop = ArmLoop(model=model, n_substeps=n_substeps, jm=jm, bus=bus,
                   action_scale=action_scale, prev_actions_n=prev_actions_n,
                   execute=args.execute, ema_alpha=args.ema_alpha,
                   slow=args.slow, interp_hz=args.interp_hz,
                   qpos_bias=load_calibration(), compliance=load_compliance())
    print(f"waypoint={args.waypoint} {loop.describe()}")

    bus.connect()
    stopped = install_sigint_flag()

    publisher = None
    if args.stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(model, args.stream_port)

    dwell_count = 0
    step = 0
    dist = float("nan")
    log_steps: list[int] = []
    log_actions: list[np.ndarray] = []
    log_qpos: list[np.ndarray] = []
    log_ee: list[np.ndarray] = []
    log_dist: list[float] = []

    try:
        loop.boot()
        while not stopped["flag"] and step < args.max_steps:
            obs = build_obs(model, data, qposadr, ee_site_id, loop.qpos, loop.qvel,
                            args.waypoint, n_waypoints, loop.prev_actions)
            raw_action, _ = policy.predict(obs, deterministic=True)
            action = loop.tick(raw_action)

            dist = float(np.linalg.norm(loop.qpos - target_qpos_goal))
            in_target = dist < tolerance
            dwell_count = dwell_count + 1 if in_target else 0
            ee = compute_ee_pos(model, data, qposadr, loop.qpos, ee_site_id)
            if publisher is not None:
                publisher.publish(data)  # data holds the live FK pose from compute_ee_pos

            log_steps.append(step)
            log_actions.append(action.copy())
            log_qpos.append(loop.qpos.copy())
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
        if publisher is not None:
            publisher.close()

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
                  dist_arr, tolerance, loop.control_hz)
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
            control_hz=loop.control_hz,
            out_path=plot_path,
        )
        print(f"saved {csv_path.relative_to(REPO_ROOT)} {plot_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
