"""Roll out the trained lift policy on the real SO-101 arm with sim-provided cube state.

The cube cannot be observed directly on the real rig, so we run a MuJoCo sim in
lockstep with the real arm: each control tick, the real encoder readings are
written into the sim, the sim is stepped forward, and the resulting sim cube
position is fed into the policy observation. The real arm executes the policy's
action; the sim is purely a passive cube-tracker driven by the real encoders.

This will NOT actually grasp the real cube — there's no alignment between the
sim cube and any physical object. It is a "what does the policy do on the real
hardware when it thinks there is a cube there" tool.

Usage:
    python -m real.rollout_lift                         # dry-run, latest checkpoint
    python -m real.rollout_lift --execute               # actually drive the servos
    python -m real.rollout_lift --model best --execute  # best_model.zip
    python -m real.rollout_lift --model logs/ppo_lift/checkpoints/ppo_4000000_steps.zip --execute
    python -m real.rollout_lift --seed 0 --execute      # reproducible cube spawn

Safety: --execute is OFF by default. Per-step raw delta is clamped via
constants.MAX_RAW_DELTA_PER_STEP. Ctrl-C disables torque.
"""

from __future__ import annotations

import argparse
import csv
import signal
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from src.base_env import JOINT_NAMES, action_to_target
from src.checkpoints import resolve_model_path

from .twin.constants import (
    INTERP_HZ,
    SERVO_ACCEL,
    SERVO_POSITION_DEADZONE,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from .twin.control import clamp_raw_delta, stream_sub_targets
from .twin.mapping import load_joint_maps, rad_to_raw, raw_to_rad
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene_lift.xml"
DEFAULT_CAL = REPO_ROOT / "real" / "follower_calibration.json"
LOG_DIR = REPO_ROOT / "logs" / "ppo_lift"

LIFT_TASK_ID = 0.0


def _load_lift_cfg() -> tuple[dict, int]:
    """Compose Hydra config with env=lift; returns (lift_env_cfg, prev_actions_n)."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    return OmegaConf.to_container(cfg.lift_env, resolve=True), int(cfg.prev_actions_n)


def parse_args(lift_cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="latest",
                   help="'latest' (newest checkpoint), 'best', or a path to a .zip")
    p.add_argument("--execute", action="store_true",
                   help="Actually send servo commands. Default: dry-run.")
    p.add_argument("--max-steps", type=int, default=int(lift_cfg["max_steps"]))
    p.add_argument("--ema-alpha", type=float, default=1.0,
                   help="EMA smoothing on policy action: 1.0 = off. Lower = smoother.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for cube spawn (default: nondeterministic).")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    p.add_argument("--no-view", action="store_true",
                   help="Disable the MuJoCo passive viewer.")
    p.add_argument("--interp-hz", type=float, default=INTERP_HZ,
                   help="Rate at which interpolated sub-targets are written to "
                        "the bus between policy ticks. Linearly slides from the "
                        "previous commanded target to the new one across each "
                        "control period, so the servo's trapezoidal trajectory "
                        "tracks a moving setpoint instead of restarting from "
                        "rest every tick. Set to the control rate (15) to "
                        "disable.")
    return p.parse_args()


def build_obs(qpos: np.ndarray, qvel: np.ndarray, ee_pos: np.ndarray,
              cube_pos: np.ndarray, prev_actions: np.ndarray) -> np.ndarray:
    """Match SO101LiftEnv._compute_obs: qpos+qvel+ee+cube+[0,0,0,task_id]+prev_actions."""
    extra = np.array([0.0, 0.0, 0.0, LIFT_TASK_ID], dtype=np.float32)
    return np.concatenate([qpos.astype(np.float32),
                           qvel.astype(np.float32),
                           ee_pos.astype(np.float32),
                           cube_pos.astype(np.float32),
                           extra,
                           prev_actions.flatten().astype(np.float32)]).astype(np.float32)


def plot_rollout(out_path: Path, rows: list[dict], target_height: float,
                 control_hz: float) -> None:
    steps = np.array([r["step"] for r in rows])
    t = steps / control_hz
    actions = np.stack([r["action"] for r in rows])
    qpos = np.stack([r["qpos"] for r in rows])
    ee = np.stack([r["ee"] for r in rows])
    cube = np.stack([r["cube"] for r in rows])
    grasped = np.array([r["grasped"] for r in rows], dtype=bool)
    ee_cube_dist = np.linalg.norm(ee - cube, axis=1)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Lift rollout — {control_hz:.1f} Hz, {len(steps)} steps")

    ax = axes[0, 0]
    for j, name in enumerate(JOINT_NAMES):
        ax.plot(t, actions[:, j], label=name)
    ax.set_title("Action per joint")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("action ∈ [-1, 1]")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for j, name in enumerate(JOINT_NAMES):
        ax.plot(t, qpos[:, j], color=cmap(j), label=name)
    ax.set_title("qpos per joint")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for k, name in enumerate("xyz"):
        color = cmap(k)
        ax.plot(t, ee[:, k], color=color, label=f"ee_{name}")
        ax.plot(t, cube[:, k], color=color, ls="--", lw=1.0, alpha=0.7,
                label=f"cube_{name}")
    ax.set_title("End-effector (solid) vs sim cube (dashed)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, cube[:, 2], color="C2", label="cube_z (sim)")
    ax.axhline(target_height, color="k", ls="--", lw=1.0, alpha=0.6,
               label=f"target_height={target_height}")
    ax.plot(t, ee_cube_dist, color="C3", label="‖ee − cube‖")
    if grasped.any():
        ax.fill_between(t, 0, 1, where=grasped, transform=ax.get_xaxis_transform(),
                        color="C1", alpha=0.15, label="grasped (sim)")
    ax.set_title("Lift progress")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_csv(out_path: Path, rows: list[dict], control_hz: float) -> None:
    header = (["step", "t_s"]
              + [f"action_{n}" for n in JOINT_NAMES]
              + [f"qpos_{n}" for n in JOINT_NAMES]
              + ["ee_x", "ee_y", "ee_z", "cube_x", "cube_y", "cube_z", "grasped_sim"])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["step"], f"{r['step'] / control_hz:.4f}",
                        *(f"{a:.6f}" for a in r["action"]),
                        *(f"{q:.6f}" for q in r["qpos"]),
                        f"{r['ee'][0]:.6f}", f"{r['ee'][1]:.6f}", f"{r['ee'][2]:.6f}",
                        f"{r['cube'][0]:.6f}", f"{r['cube'][1]:.6f}", f"{r['cube'][2]:.6f}",
                        int(r["grasped"])])


def main() -> int:
    lift_cfg, prev_actions_n = _load_lift_cfg()
    action_scale = float(lift_cfg["action_scale"])
    n_substeps = int(lift_cfg["n_substeps"])
    cube_low = np.array(lift_cfg["cube_low"], dtype=np.float64)
    cube_high = np.array(lift_cfg["cube_high"], dtype=np.float64)
    target_height = float(lift_cfg["target_height"])

    args = parse_args(lift_cfg)
    assert 0.0 < args.ema_alpha <= 1.0

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    control_dt = float(model.opt.timestep) * n_substeps
    control_hz = 1.0 / control_dt

    jm = load_joint_maps(model, Path(args.cal))
    qposadr = jm.qposadr()
    xml_low, xml_high = jm.xml_low(), jm.xml_high()
    direction = np.ones(len(jm.items), dtype=np.int8)

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    joint_dofadr = model.jnt_dofadr[joint_ids]
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_site_id >= 0, "site 'gripperframe' not found in model"
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qposadr = int(model.jnt_qposadr[cube_joint_id])

    expected_obs = 22 + prev_actions_n * 6
    model_path = resolve_model_path(args.model, str(LOG_DIR))
    print(f"loading model: {model_path}")
    policy = PPO.load(model_path)
    assert policy.observation_space.shape[0] == expected_obs, (
        f"Obs dim mismatch: model expects {policy.observation_space.shape}, "
        f"lift env produces {expected_obs}-dim."
    )

    rng = np.random.default_rng(args.seed)
    cube_pos_init = rng.uniform(cube_low, cube_high)

    print(f"execute={args.execute} {control_hz:.1f}Hz action_scale={action_scale} "
          f"ema={args.ema_alpha} seed={args.seed}")
    print(f"sim cube spawn: ({cube_pos_init[0]:+.3f}, {cube_pos_init[1]:+.3f}, "
          f"{cube_pos_init[2]:+.3f})  target_height={target_height}")

    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()

    stopped = {"flag": False}
    def stop(_sig, _frame) -> None:
        stopped["flag"] = True
    signal.signal(signal.SIGINT, stop)

    viewer = None if args.no_view else mujoco.viewer.launch_passive(model, data)
    try:
        # Initial read; sync the sim arm to the real arm and place the cube.
        raw0 = bus.read_all()
        qpos = raw_to_rad(raw0, jm, direction)
        qvel = np.zeros(6, dtype=np.float64)

        slack = 0.05
        if not (np.all(qpos >= xml_low - slack) and np.all(qpos <= xml_high + slack)):
            print("ABORT: initial qpos outside MuJoCo joint range; check calibration.")
            return 1

        data.qpos[qposadr] = qpos
        data.qvel[joint_dofadr] = 0.0
        data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos_init
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = [1, 0, 0, 0]
        # Init actuators (filter state) at current qpos so they don't snap.
        data.ctrl[:6] = qpos
        if model.na > 0:
            data.act[:] = qpos
        mujoco.mj_forward(model, data)

        if args.execute:
            bus.set_position_kp(SERVO_POSITION_KP)
            bus.set_position_deadzone(SERVO_POSITION_DEADZONE)
            bus.enable_torque_all()

        dwell_count = 0
        dt = control_dt
        n_interp = max(1, int(round(args.interp_hz / control_hz)))
        sub_dt = dt / n_interp
        print(f"interpolation: {n_interp} sub-targets/tick "
              f"({1.0 / sub_dt:.1f} Hz write rate)")
        step = 0
        prev_raw_target = raw0.copy()
        prev_actions = np.zeros((prev_actions_n, 6), dtype=np.float32)
        action_ema: np.ndarray | None = None

        def write_raw(raw: np.ndarray) -> None:
            if args.execute:
                bus.write_all(raw, SERVO_SPEED, SERVO_ACCEL)

        log_rows: list[dict] = []

        while not stopped["flag"] and step < args.max_steps:
            ee_pos = data.site_xpos[ee_site_id].copy()
            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()

            obs = build_obs(qpos, qvel, ee_pos, cube_pos, prev_actions)
            action, _ = policy.predict(obs, deterministic=True)
            action = np.clip(action.astype(np.float64), -1.0, 1.0)
            if action_ema is None:
                action_ema = action.copy()
            else:
                action_ema = args.ema_alpha * action + (1.0 - args.ema_alpha) * action_ema
            action = action_ema

            if prev_actions_n > 0:
                if prev_actions_n > 1:
                    prev_actions[:-1] = prev_actions[1:]
                prev_actions[-1] = action.astype(np.float32)

            # Exactly matches training: raw-unit quantization + stiction deadzone.
            target_qpos = action_to_target(qpos, action, action_scale, xml_low, xml_high)
            target_raw = rad_to_raw(target_qpos, jm, direction)

            target_raw = clamp_raw_delta(prev_raw_target, target_raw)
            stream_sub_targets(prev_raw_target, target_raw, n_interp, sub_dt, write_raw)
            prev_raw_target = target_raw

            # Read the real arm, write the new state into the sim, and step the
            # sim so the cube responds to gripper/floor contacts.
            raw = bus.read_all()
            new_qpos = raw_to_rad(raw, jm, direction)
            qvel = (new_qpos - qpos) / dt
            qpos = new_qpos

            data.qpos[qposadr] = qpos
            data.qvel[joint_dofadr] = qvel
            data.ctrl[:6] = qpos  # actuators hold current pose; cube reacts via contacts
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            # Re-pin the arm: prevent any drift between real and sim arm caused
            # by sim physics during the substep loop. Cube state is preserved.
            data.qpos[qposadr] = qpos
            data.qvel[joint_dofadr] = qvel
            mujoco.mj_forward(model, data)

            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            ee_pos = data.site_xpos[ee_site_id].copy()

            ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
            gripper_val = qpos[JOINT_NAMES.index("gripper")]
            grasped_sim = ee_cube < 0.05 and gripper_val < 0.3
            if cube_pos[2] >= target_height:
                dwell_count += 1
            else:
                dwell_count = 0

            log_rows.append({
                "step": step, "action": action.copy(), "qpos": qpos.copy(),
                "ee": ee_pos.copy(), "cube": cube_pos.copy(), "grasped": grasped_sim,
            })

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    print("Viewer closed; stopping rollout.")
                    break

            if step % 15 == 0:
                print(f"step={step:3d}  ee-cube={ee_cube:.3f}m  "
                      f"cube_z={cube_pos[2]:.3f}m  grasped_sim={int(grasped_sim)}")
            step += 1

            if dwell_count >= 5:
                print(f"SIM cube reached target_height={target_height} at step {step}")
                break
        else:
            if not stopped["flag"]:
                print(f"TIMEOUT at step {step}")
    finally:
        bus.close()
        if viewer is not None:
            viewer.close()

    if log_rows:
        out_dir = REPO_ROOT / "rollouts"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rollout_lift_{int(time.time())}"
        csv_path = out_dir / f"{stem}.csv"
        plot_path = out_dir / f"{stem}.png"
        write_csv(csv_path, log_rows, control_hz)
        plot_rollout(plot_path, log_rows, target_height, control_hz)
        print(f"saved {csv_path.relative_to(REPO_ROOT)} {plot_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
