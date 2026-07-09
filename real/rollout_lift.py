"""Roll out the trained lift policy on the real SO-101 arm.

The cube obs is the raw pose (+ age) of the AprilTag on the sponge's largest
face (src/base_env.py convention). Its source follows --marker-source:

- camera: the real sponge. real/marker_obs.py measures tag id 1 and maps it to
  the base frame; the policy chases the physical object, and success is dwell
  on the measured sponge-center height (back-derived from the tag pose for
  termination only). The lockstep sim cube is pinned to the measurement each
  tick so the viewer shows the real sponge.
- fk (default, dry-run): a lockstep MuJoCo sim cube driven by the real
  encoders via contacts, run through the same sim visibility convention
  (cube_tag_visible + hold-last + age), so dry-runs exercise the identical obs
  contract with no physical sponge.

Usage:
    python -m real.rollout_lift                         # dry-run, latest checkpoint
    python -m real.rollout_lift --execute               # actually drive the servos
    python -m real.rollout_lift --marker-source camera --execute   # real sponge
    python -m real.rollout_lift --model best --execute  # best_model.zip
    python -m real.rollout_lift --seed 0                # reproducible fk cube spawn
    python -m real.rollout_lift --slow 3 --execute      # 1/3 physical speed, no retraining

Setup, safety gating, and per-tick command shaping (training-matched
quantization, raw clamp, sub-target streaming, --slow time dilation) all live
in real.rollout_common — this script owns only observation construction,
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
from scipy.spatial.transform import Rotation

from src.base_env import (
    CUBE_TAG_SITE_NAME,
    JOINT_NAMES,
    MARKER_AGE_CAP_S,
    MARKER_SITE_NAMES,
    N_MARKERS,
    cube_tag_visible,
    marker_world_poses,
    markers_visible,
    obs_dim_for,
    priv_dim_for,
    sample_cube_orientation,
    tag_cam_model,
)

from .rollout_common import (
    ArmLoop,
    add_common_args,
    install_sigint_flag,
    load_env_cfg,
    load_policy,
)
from src.obs_history import ObsHistory

from .calibration import load_calibration, load_compliance
from .marker_obs import CameraMarkerSource
from .twin.mapping import load_joint_maps
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene_lift.xml"
LOG_DIR = REPO_ROOT / "logs" / "ppo_lift"

LIFT_TASK_ID = 0.0

# Abort if the newest camera frame is older than this when the policy consumes
# it. In-distribution staleness is <= ~90 ms (42-52 ms pipeline delay plus up
# to one 33 ms frame interval — conf/dr/full.yaml); a frame this old means the
# camera or capture thread stalled and the policy would be steering the arm on
# frozen marker poses.
MAX_MARKER_AGE_S = 0.25

# Marker age fed for a freshly-visible tag under --marker-source fk: the FK
# stand-in has no camera pipeline, so feed the middle of the training age
# distribution (42-52 ms delay + 0-33 ms frame wait — conf/dr/full.yaml).
FK_FRESH_AGE_S = 0.06

# Camera-mode success dwell only counts while the measured cube pose is this
# fresh: a tag that vanished mid-lift freezes its held pose, which must not
# keep faking "above target height".
CUBE_FRESH_DWELL_S = 0.15


def tag_center_z(cube_tag_pos: np.ndarray, cube_tag_rot: np.ndarray, hz: float) -> float:
    """Sponge-center height back-derived from the measured tag pose, for
    termination only — the obs carries the raw tag pose. center = tag_pos -
    hz * (tag +z axis); the offset runs along the tag's own normal, so the
    uncalibrated in-plane glue yaw cannot affect it."""
    z_axis = Rotation.from_rotvec(cube_tag_rot).as_matrix()[:, 2]
    return float(cube_tag_pos[2] - hz * z_axis[2])


def parse_args(lift_cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p, default_xml=DEFAULT_XML,
                    default_max_steps=int(lift_cfg["max_steps"]))
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for cube spawn (default: nondeterministic).")
    p.add_argument("--no-view", action="store_true",
                   help="Disable the MuJoCo passive viewer.")
    p.add_argument("--marker-source", default="fk", choices=["fk", "camera"],
                   help="Where marker AND cube-tag observations come from: 'fk' "
                        "(default) fills them from the lockstep sim; 'camera' feeds "
                        "measured AprilTag poses (arm tags + the sponge's tag id 1) "
                        "mapped to the base frame via the calibrated extrinsics "
                        "(real/calibrate_qpos.py).")
    p.add_argument("--family", default="apriltag", choices=["apriltag", "aruco"],
                   help="Marker family for --marker-source camera.")
    return p.parse_args()


def build_actor_frame(qpos: np.ndarray, qvel: np.ndarray, marker_pos: np.ndarray,
                      marker_rot: np.ndarray, marker_age: np.ndarray,
                      cube_tag_pos: np.ndarray, cube_tag_rot: np.ndarray,
                      cube_age: float, prev_actions: np.ndarray,
                      marker_include_rot: bool) -> np.ndarray:
    """Match SO101LiftEnv's single-frame actor block (base_env.obs_dim_for):
    qpos+qvel+markers+marker_age+cube_tag(pos+rot)+cube_age+[0,0,0,task_id]+
    prev_actions.

    Marker and cube-tag poses/ages come from the camera pipeline
    (--marker-source camera) or the FK stand-in on the lockstep sim — held
    last-detected poses either way. marker_include_rot mirrors the env:
    marker positions only when false (the cube tag always carries its rot).

    The caller feeds this frame through the shared ObsHistory (the identical
    tap convention training used, src/obs_history.py) and appends the zeroed
    privileged tail: only the value function read those dims in training and
    the actor structurally slices them off (src/networks.TakeFirst), so at
    deployment they are never read.
    """
    extra = np.array([0.0, 0.0, 0.0, LIFT_TASK_ID], dtype=np.float32)
    markers = (np.hstack([marker_pos, marker_rot]).flatten()
               if marker_include_rot else marker_pos.flatten())
    return np.concatenate([qpos.astype(np.float32),
                           qvel.astype(np.float32),
                           markers.astype(np.float32),
                           marker_age.astype(np.float32),
                           cube_tag_pos.astype(np.float32),
                           cube_tag_rot.astype(np.float32),
                           np.array([cube_age], dtype=np.float32),
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
    ax.plot(t, cube[:, 2], color="C2", label="cube_z")
    ax.axhline(target_height, color="k", ls="--", lw=1.0, alpha=0.6,
               label=f"target_height={target_height}")
    ax.plot(t, ee_cube_dist, color="C3", label="‖ee − cube‖")
    cube_age = np.array([r["cube_age"] for r in rows])
    ax.plot(t, cube_age * 0.1, color="C4", ls=":", lw=1.0,
            label="cube_age (0.1 = 1 s)")
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


def print_latency_summary(rows: list[dict]) -> None:
    """Per-component latency breakdown over the rollout (ms). NaN-only components
    (e.g. camera stats under --marker-source fk) print 'n/a'."""
    keys = ["loop_ms", "predict_ms", "read_all_ms", "stream_ms", "window_ms",
            "marker_age_ms", "cam_read_ms", "detect_ms"]
    print("latency summary (ms):")
    for key in keys:
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            print(f"  {key:14s} n/a")
        else:
            print(f"  {key:14s} mean={v.mean():6.1f}  p95={np.percentile(v, 95):6.1f}  "
                  f"max={v.max():6.1f}")


def write_csv(out_path: Path, rows: list[dict], control_hz: float) -> None:
    tag_names = [n.removeprefix("marker_") for n in MARKER_SITE_NAMES]
    header = (["step", "t_s"]
              + [f"action_{n}" for n in JOINT_NAMES]
              + [f"qpos_{n}" for n in JOINT_NAMES]
              + ["ee_x", "ee_y", "ee_z", "cube_x", "cube_y", "cube_z", "grasped_sim"]
              # marker obs the policy saw (camera or FK stand-in), the FK pose
              # of the same tick's qpos, and the per-tag age fed to the policy
              + [f"mobs_{t}_{ax}" for t in tag_names for ax in "xyz"]
              + [f"mfk_{t}_{ax}" for t in tag_names for ax in "xyz"]
              + [f"mage_{t}_s" for t in tag_names]
              # cube-tag obs the policy saw (raw tag pose + age)
              + [f"ctag_{ax}" for ax in "xyz"]
              + [f"ctag_r{ax}" for ax in "xyz"]
              + ["cube_age_s"]
              + ["loop_ms", "predict_ms", "read_all_ms", "stream_ms", "window_ms",
                 "marker_age_ms", "cam_read_ms", "detect_ms"])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["step"], f"{r['step'] / control_hz:.4f}",
                        *(f"{a:.6f}" for a in r["action"]),
                        *(f"{q:.6f}" for q in r["qpos"]),
                        f"{r['ee'][0]:.6f}", f"{r['ee'][1]:.6f}", f"{r['ee'][2]:.6f}",
                        f"{r['cube'][0]:.6f}", f"{r['cube'][1]:.6f}", f"{r['cube'][2]:.6f}",
                        int(r["grasped"]),
                        *(f"{v:.6f}" for v in r["marker_obs"].flatten()),
                        *(f"{v:.6f}" for v in r["marker_fk"].flatten()),
                        *(f"{v:.4f}" for v in r["marker_age"]),
                        *(f"{v:.6f}" for v in r["cube_tag_pos"]),
                        *(f"{v:.6f}" for v in r["cube_tag_rot"]),
                        f"{r['cube_age']:.4f}",
                        f"{r['loop_ms']:.2f}", f"{r['predict_ms']:.3f}",
                        f"{r['read_all_ms']:.2f}", f"{r['stream_ms']:.2f}",
                        f"{r['window_ms']:.2f}",
                        f"{r['marker_age_ms']:.2f}", f"{r['cam_read_ms']:.2f}",
                        f"{r['detect_ms']:.2f}"])


def main() -> int:
    lift_cfg, prev_actions_n, marker_include_rot, history_taps = load_env_cfg("lift")
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
    tag_cam = tag_cam_model(model, data)
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qposadr = int(model.jnt_qposadr[cube_joint_id])
    cube_dofadr = int(model.jnt_dofadr[cube_joint_id])
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert cube_geom_id >= 0, "geom 'cube_geom' not found in model"
    cube_body_id = int(model.geom_bodyid[cube_geom_id])
    cube_tag_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CUBE_TAG_SITE_NAME)
    assert cube_tag_site_id >= 0, f"site '{CUBE_TAG_SITE_NAME}' not found in model"
    # Tag-face offset from the sponge center, sourced from the loaded scene XML.
    cube_hz = float(model.geom_size[cube_geom_id][2])
    # Static body->tag-site transform (the site carries the calibrated in-plane
    # glue yaw, euler in scene_*.xml); inverted below to draw the sponge body
    # from the measured tag pose.
    sq = model.site_quat[cube_tag_site_id]  # MuJoCo wxyz
    cube_tag_site_R = Rotation.from_quat([sq[1], sq[2], sq[3], sq[0]])
    cube_tag_site_pos = model.site_pos[cube_tag_site_id].copy()

    policy = load_policy(args.model, LOG_DIR,
                         len(history_taps) * obs_dim_for(prev_actions_n, marker_include_rot)
                         + priv_dim_for(marker_include_rot))
    # Lag-tap history over the actor block, the same convention as training
    # (src/obs_history.py): seeded with the first tick's frame at boot — the
    # env's reset does the identical thing — then advanced once per tick.
    history = ObsHistory(history_taps, obs_dim_for(prev_actions_n, marker_include_rot))
    priv_pad = np.zeros(priv_dim_for(marker_include_rot), dtype=np.float32)

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
                   qpos_bias=load_calibration(), compliance=load_compliance())
    print(f"seed={args.seed} {loop.describe()}")
    if args.marker_source == "fk":
        print(f"sim cube spawn: ({cube_pos_init[0]:+.3f}, {cube_pos_init[1]:+.3f}, "
              f"{cube_pos_init[2]:+.3f})  target_height={target_height}")
    else:
        print(f"cube = measured tag id 1 (camera)  target_height={target_height}")

    bus.connect()
    stopped = install_sigint_flag()

    marker_source = (CameraMarkerSource(args.family, track_cube=True)
                     if args.marker_source == "camera" else None)

    viewer = None if args.no_view else mujoco.viewer.launch_passive(model, data)
    publisher = None
    if args.stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(model, args.stream_port)
    # Overlay the camera's measured marker poses in the passive viewer too, beside
    # the arm's own marker sites, so calibration drift is visible. The stream draws
    # them itself inside publisher.publish; the viewer needs the helper directly.
    draw_markers = None
    if viewer is not None and marker_source is not None:
        from panel.sim_stream import draw_detected_markers as draw_markers
    log_rows: list[dict] = []
    try:
        if marker_source is not None:
            marker_source.start()
            # Let the camera + detector settle before the arm moves: block until
            # the table anchor tag is being detected, so the first obs is built on
            # a camera actually mapped to the base frame, not zeroed/held poses.
            waited = marker_source.warmup()
            print(f"camera warmup: table tag anchored after {waited:.2f} s")
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
        prev_iter_t = None
        # FK-branch hold-last state mirroring training (src/base_env.py): a tag
        # turned away from tag_cam keeps its last visible pose while its age
        # grows; never-yet-visible tags read zero with age at the cap. The
        # lockstep sim's cube tag goes through the identical convention.
        fk_pos = np.zeros((N_MARKERS, 3))
        fk_rot = np.zeros((N_MARKERS, 3))
        fk_seen_t = np.full(N_MARKERS, -np.inf)
        fk_cube_pos = np.zeros(3)
        fk_cube_rot = np.zeros(3)
        fk_cube_seen_t = -np.inf
        while not stopped["flag"] and step < args.max_steps:
            iter_t = time.perf_counter()
            # Realized control period (start-to-start of consecutive ticks).
            loop_ms = (iter_t - prev_iter_t) * 1e3 if prev_iter_t is not None else float("nan")
            prev_iter_t = iter_t

            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            # FK marker poses of the same qpos the obs uses (data is still
            # pinned at loop.qpos from the previous tick). Logged next to the
            # camera's measured poses: their difference is the live
            # camera-vs-encoder-chain disagreement per tag.
            fk_pos_now, fk_rot_now = marker_world_poses(data, marker_site_ids)
            if marker_source is not None:
                # Measured AprilTag poses in the base frame, held at the last
                # detection per tag with their age (real/marker_obs.py).
                marker_pos, marker_rot, marker_age = marker_source.marker_poses()
                cube_tag_pos, cube_tag_rot, cube_age = marker_source.cube_pose()
                # Latency of the newest frame, sampled the instant the policy
                # consumes it — the whole-pipeline stall guard.
                stale_s, cam_read_ms, detect_ms = marker_source.frame_stats()
                if stale_s > MAX_MARKER_AGE_S:
                    raise SystemExit(
                        f"ABORT: marker frame is {stale_s * 1e3:.0f} ms old "
                        f"(limit {MAX_MARKER_AGE_S * 1e3:.0f} ms); camera "
                        f"pipeline stalled.")
                marker_age_ms = stale_s * 1e3
            else:
                # Match training (base_env._compute_obs): a tag turned away
                # from tag_cam holds its last visible pose, age keeps growing.
                # Same convention for the lockstep sim cube's tag, occlusion
                # test included.
                now = time.monotonic()
                vis = markers_visible(data, marker_site_ids, tag_cam)
                fk_pos[vis] = fk_pos_now[vis]
                fk_rot[vis] = fk_rot_now[vis]
                fk_seen_t[vis] = now - FK_FRESH_AGE_S
                marker_pos, marker_rot = fk_pos.copy(), fk_rot.copy()
                marker_age = np.minimum(MARKER_AGE_CAP_S, now - fk_seen_t)
                if cube_tag_visible(model, data, cube_tag_site_id, tag_cam,
                                    cube_body_id):
                    (fk_cube_pos,), (fk_cube_rot,) = marker_world_poses(
                        data, [cube_tag_site_id])
                    fk_cube_seen_t = now - FK_FRESH_AGE_S
                cube_tag_pos, cube_tag_rot = fk_cube_pos.copy(), fk_cube_rot.copy()
                cube_age = min(MARKER_AGE_CAP_S, now - fk_cube_seen_t)
                marker_age_ms = cam_read_ms = detect_ms = float("nan")

            frame = build_actor_frame(loop.qpos, loop.qvel, marker_pos, marker_rot,
                                      marker_age, cube_tag_pos, cube_tag_rot, cube_age,
                                      loop.prev_actions, marker_include_rot)
            tapped = history.reset(frame) if step == 0 else history.push(frame)
            obs = np.concatenate([tapped, priv_pad])
            t_pred = time.perf_counter()
            raw_action, _ = policy.predict(obs, deterministic=True)
            predict_ms = (time.perf_counter() - t_pred) * 1e3
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
            if marker_source is not None and cube_tag_pos.any():
                # Pin the sim cube to the measurement so the viewer shows the
                # real sponge (held pose while the tag is hidden). Center and
                # orientation are back-derived from the tag pose for display,
                # inverting the tag site's body-relative glue yaw so the drawn
                # body matches the real sponge; the obs carries the raw tag pose.
                tag_R = Rotation.from_rotvec(cube_tag_rot)
                body_R = tag_R * cube_tag_site_R.inv()
                data.qpos[cube_qposadr:cube_qposadr + 3] = \
                    cube_tag_pos - body_R.apply(cube_tag_site_pos)
                x, y, z, w = body_R.as_quat()  # scipy xyzw -> MuJoCo wxyz
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = (w, x, y, z)
                data.qvel[cube_dofadr:cube_dofadr + 6] = 0.0
            mujoco.mj_forward(model, data)
            if publisher is not None:
                if marker_source is not None:
                    publisher.publish(data, marker_pos, marker_rot, marker_include_rot)
                else:
                    publisher.publish(data)

            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            ee_pos = data.site_xpos[ee_site_id].copy()

            ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
            gripper_val = loop.qpos[JOINT_NAMES.index("gripper")]
            grasped_sim = ee_cube < 0.05 and gripper_val < 0.3
            if marker_source is not None:
                # Success = dwell on the measured sponge-center height, counted
                # only while the measurement is fresh — a frozen held pose must
                # not fake a lift.
                center_z = tag_center_z(cube_tag_pos, cube_tag_rot, cube_hz)
                if cube_age < CUBE_FRESH_DWELL_S and center_z >= target_height:
                    dwell_count += 1
                else:
                    dwell_count = 0
            elif cube_pos[2] >= target_height:
                dwell_count += 1
            else:
                dwell_count = 0

            log_rows.append({
                "step": step, "action": action.copy(), "qpos": loop.qpos.copy(),
                "ee": ee_pos.copy(), "cube": cube_pos.copy(), "grasped": grasped_sim,
                "marker_obs": marker_pos.copy(), "marker_fk": fk_pos_now.copy(),
                "marker_age": marker_age.copy(),
                "cube_tag_pos": cube_tag_pos.copy(), "cube_tag_rot": cube_tag_rot.copy(),
                "cube_age": cube_age,
                "loop_ms": loop_ms, "predict_ms": predict_ms,
                "read_all_ms": loop.last_read_ms, "stream_ms": loop.last_stream_ms,
                "window_ms": loop.last_window_ms,
                "marker_age_ms": marker_age_ms, "cam_read_ms": cam_read_ms,
                "detect_ms": detect_ms,
            })

            if viewer is not None:
                if draw_markers is not None:
                    viewer.user_scn.ngeom = 0
                    draw_markers(viewer.user_scn, marker_pos, marker_rot,
                                 marker_include_rot)
                viewer.sync()
                if not viewer.is_running():
                    print("Viewer closed; stopping rollout.")
                    break

            if step % 15 == 0:
                print(f"step={step:3d}  ee-cube={ee_cube:.3f}m  "
                      f"cube_z={cube_pos[2]:.3f}m  cube_age={cube_age:.2f}s  "
                      f"grasped_sim={int(grasped_sim)}")
                lat = (f"          lat[ms]: loop={loop_ms:.0f} predict={predict_ms:.2f} "
                       f"read_all={loop.last_read_ms:.0f} stream={loop.last_stream_ms:.0f} "
                       f"window={loop.last_window_ms:.0f}")
                if marker_source is not None:
                    lat += (f" | cam age={marker_age_ms:.0f} "
                            f"read={cam_read_ms:.0f} detect={detect_ms:.0f}")
                    cam_fk_mm = np.linalg.norm(marker_pos - fk_pos_now, axis=1) * 1e3
                    lat += " | cam-fk[mm]=" + "/".join(f"{v:.0f}" for v in cam_fk_mm)
                print(lat)
            step += 1

            if dwell_count >= 5:
                what = ("measured sponge center held above"
                        if marker_source is not None else "SIM cube reached")
                print(f"{what} target_height={target_height} at step {step}")
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
        print_latency_summary(log_rows)
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
