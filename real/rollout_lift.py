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
    python -m real.rollout_lift --slow 3 --execute      # 1/3 physical speed, no retraining

Setup, safety gating, and per-tick command shaping (training-matched
quantization, raw clamp, sub-target streaming, --slow time dilation) all live
in real.rollout_common — this script owns only the lockstep cube sim,
termination, and plots. --execute is OFF by default; Ctrl-C disables torque.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.base_env import (
    JOINT_NAMES,
    MARKER_SITE_NAMES,
    marker_world_poses,
    markers_visible,
    obs_dim_for,
    sample_cube_orientation,
    tag_cam_world_pos,
)

from .rollout_common import (
    ArmLoop,
    add_common_args,
    install_sigint_flag,
    load_env_cfg,
    load_policy,
)
from .calibration import load_calibration
from .marker_obs import CameraMarkerSource
from .twin.mapping import load_joint_maps
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene_lift.xml"
LOG_DIR = REPO_ROOT / "logs" / "ppo_lift"

LIFT_TASK_ID = 0.0


def parse_args(lift_cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p, default_xml=DEFAULT_XML,
                    default_max_steps=int(lift_cfg["max_steps"]))
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for cube spawn (default: nondeterministic).")
    p.add_argument("--no-view", action="store_true",
                   help="Disable the MuJoCo passive viewer.")
    p.add_argument("--marker-source", default="fk", choices=["fk", "camera"],
                   help="Where marker observations come from: 'fk' (default) "
                        "fills them from the lockstep sim; 'camera' feeds measured "
                        "AprilTag poses mapped to the base frame via the calibrated "
                        "extrinsics (real/calibrate_table_marker.py).")
    p.add_argument("--family", default="apriltag", choices=["apriltag", "aruco"],
                   help="Marker family for --marker-source camera.")
    return p.parse_args()


def build_obs(qpos: np.ndarray, qvel: np.ndarray, marker_pos: np.ndarray,
              marker_rot: np.ndarray, cube_pos: np.ndarray,
              prev_actions: np.ndarray, marker_include_rot: bool) -> np.ndarray:
    """Match SO101LiftEnv._compute_obs: qpos+qvel+markers+cube+[0,0,0,task_id]+prev_actions.

    Marker poses come from FK on the lockstep sim (the same sites the policy saw
    in training) — a stand-in until the camera AprilTag pipeline feeds measured
    poses here. marker_include_rot mirrors the env: positions only when false.
    """
    extra = np.array([0.0, 0.0, 0.0, LIFT_TASK_ID], dtype=np.float32)
    markers = (np.hstack([marker_pos, marker_rot]).flatten()
               if marker_include_rot else marker_pos.flatten())
    return np.concatenate([qpos.astype(np.float32),
                           qvel.astype(np.float32),
                           markers.astype(np.float32),
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
    lift_cfg, prev_actions_n, marker_include_rot = load_env_cfg("lift")
    action_scale = float(lift_cfg["action_scale"])
    n_substeps = int(lift_cfg["n_substeps"])
    cube_low = np.array(lift_cfg["cube_low"], dtype=np.float64)
    cube_high = np.array(lift_cfg["cube_high"], dtype=np.float64)
    target_height = float(lift_cfg["target_height"])

    args = parse_args(lift_cfg)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    jm = load_joint_maps(model, Path(args.cal))
    qposadr = jm.qposadr()

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    joint_dofadr = model.jnt_dofadr[joint_ids]
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_site_id >= 0, "site 'gripperframe' not found in model"
    marker_site_ids = []
    for name in MARKER_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid >= 0, f"site '{name}' not found in model"
        marker_site_ids.append(sid)
    tag_cam_pos = tag_cam_world_pos(model, data)
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qposadr = int(model.jnt_qposadr[cube_joint_id])
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert cube_geom_id >= 0, "geom 'cube_geom' not found in model"

    policy = load_policy(args.model, LOG_DIR,
                         obs_dim_for(prev_actions_n, marker_include_rot))

    rng = np.random.default_rng(args.seed)
    cube_xy_init = rng.uniform(cube_low, cube_high)
    cube_quat_init, rest_half_z = sample_cube_orientation(
        rng, model.geom_size[cube_geom_id])
    cube_pos_init = np.array([cube_xy_init[0], cube_xy_init[1], rest_half_z])

    bus = ServoBus(args.port, jm.servo_ids())
    loop = ArmLoop(model=model, n_substeps=n_substeps, jm=jm, bus=bus,
                   action_scale=action_scale, prev_actions_n=prev_actions_n,
                   execute=args.execute, ema_alpha=args.ema_alpha,
                   slow=args.slow, interp_hz=args.interp_hz,
                   qpos_bias=load_calibration())
    print(f"seed={args.seed} {loop.describe()}")
    print(f"sim cube spawn: ({cube_pos_init[0]:+.3f}, {cube_pos_init[1]:+.3f}, "
          f"{cube_pos_init[2]:+.3f})  target_height={target_height}")

    bus.connect()
    stopped = install_sigint_flag()

    marker_source = CameraMarkerSource(args.family) if args.marker_source == "camera" else None

    viewer = None if args.no_view else mujoco.viewer.launch_passive(model, data)
    publisher = None
    if args.stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(model, args.stream_port)
    log_rows: list[dict] = []
    try:
        if marker_source is not None:
            marker_source.start()
        loop.boot()

        # Sync the sim arm to the real arm and place the cube.
        data.qpos[qposadr] = loop.qpos
        data.qvel[joint_dofadr] = 0.0
        data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos_init
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quat_init
        # Init actuators (filter state) at current qpos so they don't snap.
        data.ctrl[:6] = loop.qpos
        if model.na > 0:
            data.act[:] = loop.qpos
        mujoco.mj_forward(model, data)

        dwell_count = 0
        step = 0
        while not stopped["flag"] and step < args.max_steps:
            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            if marker_source is not None:
                # Measured AprilTag poses, already in the base frame and zeroed
                # for tags the camera can't see (real/marker_obs.py).
                marker_pos, marker_rot = marker_source.marker_poses()
            else:
                marker_pos, marker_rot = marker_world_poses(data, marker_site_ids)
                # Match training (base_env._compute_obs): a tag turned away from
                # tag_cam is zeroed — the policy never saw FK-filled hidden tags.
                hidden = ~markers_visible(data, marker_site_ids, tag_cam_pos)
                marker_pos[hidden] = 0.0
                marker_rot[hidden] = 0.0

            obs = build_obs(loop.qpos, loop.qvel, marker_pos, marker_rot,
                            cube_pos, loop.prev_actions, marker_include_rot)
            raw_action, _ = policy.predict(obs, deterministic=True)
            action = loop.tick(raw_action)

            # Write the real arm's new state into the sim and step it so the
            # cube responds to gripper/floor contacts.
            data.qpos[qposadr] = loop.qpos
            data.qvel[joint_dofadr] = loop.qvel
            data.ctrl[:6] = loop.qpos  # actuators hold current pose; cube reacts via contacts
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            # Re-pin the arm: prevent any drift between real and sim arm caused
            # by sim physics during the substep loop. Cube state is preserved.
            data.qpos[qposadr] = loop.qpos
            data.qvel[joint_dofadr] = loop.qvel
            mujoco.mj_forward(model, data)
            if publisher is not None:
                publisher.publish(data)

            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            ee_pos = data.site_xpos[ee_site_id].copy()

            ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
            gripper_val = loop.qpos[JOINT_NAMES.index("gripper")]
            grasped_sim = ee_cube < 0.05 and gripper_val < 0.3
            if cube_pos[2] >= target_height:
                dwell_count += 1
            else:
                dwell_count = 0

            log_rows.append({
                "step": step, "action": action.copy(), "qpos": loop.qpos.copy(),
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
        if marker_source is not None:
            marker_source.stop()
        bus.close()
        if viewer is not None:
            viewer.close()
        if publisher is not None:
            publisher.close()

    if log_rows:
        out_dir = REPO_ROOT / "rollouts"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rollout_lift_{int(time.time())}"
        csv_path = out_dir / f"{stem}.csv"
        plot_path = out_dir / f"{stem}.png"
        write_csv(csv_path, log_rows, loop.control_hz)
        plot_rollout(plot_path, log_rows, target_height, loop.control_hz)
        print(f"saved {csv_path.relative_to(REPO_ROOT)} {plot_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
