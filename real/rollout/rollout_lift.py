"""Roll out the trained lift policy on the real SO-101 arm.

The cube obs is the tag-free live centroid plus one static-refreshed BPS block.
The FK source is a lockstep MuJoCo sponge driven by the real encoders via
contacts and the same visible-surface/BPS helpers as training. ``camera`` uses
the asynchronous tag-free SAM + dense-stereo worker behind ``FrameBus``; it
validates the rigid rig against the calibrated table-anchor reference before
publishing a cloud and never runs StereoSGBM on the control thread.

Usage:
    python -m real.rollout.rollout_lift                         # dry-run, latest checkpoint
    python -m real.rollout.rollout_lift --execute               # actually drive the servos
    python -m real.rollout.rollout_lift --model best --execute  # best_model.zip
    python -m real.rollout.rollout_lift --seed 0                # reproducible fk cube spawn
    python -m real.rollout.rollout_lift --slow 3 --execute      # 1/3 physical speed, no retraining

Setup, safety gating, and per-tick command shaping (training-matched
quantization, raw clamp, sub-target streaming, --slow time dilation) all live
in real.rollout.rollout_common — this script owns only observation construction,
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
    MARKER_AGE_CAP_S,
    MARKER_SITE_NAMES,
    N_MARKERS,
    TAG_CAM_AUX_NAME,
    cube_surface_points_world,
    cube_visible_surface,
    marker_world_poses,
    markers_visible,
    obs_dim_for,
    priv_dim_for,
    state_dim_for,
    sample_cube_orientation,
    tag_cam_model,
)
from src.obs_history import ObsHistory
from src.bps import BPS_DISTANCE_DIM, BPSObsState, load_bps_config
from src.shape_obs import (
    STATIC_DWELL_S,
    VISIBLE_FRACTION_MIN,
    ObjectChannelDriver,
)
from src.sim_bps import SyntheticBPSGenerator, clean_synthetic_cloud_config

from .rollout_common import (
    ArmLoop,
    add_common_args,
    install_sigint_flag,
    load_env_cfg,
    load_policy,
)

from ..calib.calibration import load_calibration, load_compliance
from ..twin.mapping import load_joint_maps
from ..twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene_lift.xml"
LOG_DIR = REPO_ROOT / "logs" / "ppo_lift"

LIFT_TASK_ID = 0.0

# Ages fed for freshly measured FK arm-marker channels. The stand-in has no
# camera pipeline, so feed the middle of the training age
# distribution (42-52 ms delay + 0-33 ms frame wait — conf/dr/full.yaml).
FK_FRESH_AGE_S = 0.06

# Success dwell only counts while the live channel is this fresh: a channel
# that lost the object mid-lift freezes its held value, which must not keep
# faking "above target height" (plan decision 9).
CUBE_FRESH_DWELL_S = 0.15


class FkObjectSource:
    """FK twin of the real dense-stereo source.

    It drives the training env's surface helpers from the lockstep sim and
    publishes through the shared object/BPS state machines.
    """

    def __init__(self, model, data, cube_geom_id, cube_body_id):
        self.model = model
        self.data = data
        self.cube_geom_id = cube_geom_id
        self.cube_body_id = cube_body_id
        self.cams = (tag_cam_model(model, data),
                     tag_cam_model(model, data, TAG_CAM_AUX_NAME, "aux"))
        self.driver = ObjectChannelDriver()
        self.bps_state = BPSObsState()
        self.half_extents = model.geom_size[cube_geom_id].copy()
        self.bps_config = load_bps_config()
        self.generator = SyntheticBPSGenerator(
            self.bps_config, clean_synthetic_cloud_config(),
        )
        self.rng = np.random.default_rng(0)

    def boot(self):
        """Seed the pre-episode static evidence, like the env reset and the
        real rig's settle-before-start warmup."""
        live, _ = self._measure()
        if live is not None:
            t = time.monotonic() - FK_FRESH_AGE_S
            self.driver.seed_static(t - STATIC_DWELL_S, live)
            self.tick()

    def _measure(self):
        points, normals = cube_surface_points_world(self.data, self.cube_geom_id,
                                                    self.half_extents)
        fracs, centroids = [], []
        for cam in self.cams:
            frac, centroid = cube_visible_surface(self.model, self.data, cam,
                                                  self.cube_body_id, points, normals)
            if centroid is None:
                return None, None
            fracs.append(frac)
            centroids.append(centroid)
        return np.mean(centroids, axis=0), np.array(fracs)

    def tick(self):
        """One measurement of the lockstep sim, at FK freshness."""
        live, vis_frac = self._measure()
        if live is None:
            return
        t = time.monotonic() - FK_FRESH_AGE_S
        self.driver.ingest_live(t, live)
        if self.driver.gate_open(vis_frac):
            capture = self.generator.capture(
                self.model, self.data, self.cams, self.cube_geom_id,
                self.cube_body_id, self.half_extents, self.rng)
            self.bps_state.ingest(t, None if capture is None else capture.measurement)

    def object_obs(self):
        now = time.monotonic()
        return self.driver.serve(now), self.bps_state.serve(now)


def sample_visible_fk_spawn(fk_object, data, cube_qposadr, rng, cube_low,
                            cube_high, max_attempts=100):
    """FK twin of base_env._sample_visible_cube_spawn: reject spawns either
    sim camera cannot comfortably see (visible fraction under the precise
    gate), so the dry-run's channels boot fresh like a real episode. Leaves
    the accepted pose written into data; returns (pos, quat)."""
    for _ in range(max_attempts):
        xy = rng.uniform(cube_low, cube_high)
        quat, rest_half_z = sample_cube_orientation(rng, fk_object.half_extents)
        pos = np.array([xy[0], xy[1], rest_half_z])
        data.qpos[cube_qposadr:cube_qposadr + 3] = pos
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = quat
        mujoco.mj_forward(fk_object.model, data)
        live, vis_frac = fk_object._measure()
        if live is not None and np.all(vis_frac >= VISIBLE_FRACTION_MIN):
            return pos, quat
    raise RuntimeError(f"no both-camera-visible fk cube spawn in {max_attempts} "
                       "attempts; check the spawn box against the sim cameras")


def parse_args(lift_cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p, default_xml=DEFAULT_XML,
                    default_max_steps=int(lift_cfg["max_steps"]))
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for cube spawn (default: nondeterministic).")
    p.add_argument("--no-view", action="store_true",
                   help="Disable the MuJoCo passive viewer.")
    p.add_argument("--marker-source", default="fk", choices=["fk", "camera"],
                   help="Observation source: lockstep MuJoCo stand-in or the "
                        "real tag-free camera pipeline.")
    p.add_argument("--object-prompt", default="sponge",
                   help="SAM3 text prompt used to acquire the object in camera mode.")
    p.add_argument("--sam2-model", default="tiny", choices=["tiny", "base+"],
                   help="Streaming mask tracker used in camera mode.")
    return p.parse_args()


def build_state_frame(qpos: np.ndarray, qvel: np.ndarray, marker_pos: np.ndarray,
                      marker_rot: np.ndarray, marker_age: np.ndarray,
                      live: np.ndarray, live_age: float,
                      prev_actions: np.ndarray,
                      marker_include_rot: bool) -> np.ndarray:
    """Match one historical state frame in SO101LiftEnv.

    Marker poses/ages and the live centroid come from the lockstep FK stand-in
    and hold their last measurement. marker_include_rot mirrors the env:
    marker positions only when false.

    The caller feeds this frame through the shared ObsHistory, then appends the
    one current BPS block and the zeroed
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
                           live.astype(np.float32),
                           np.array([live_age], dtype=np.float32),
                           extra,
                           prev_actions.flatten().astype(np.float32)]).astype(np.float32)


def plot_rollout(out_path: Path, rows: list[dict], target_height: float,
                 control_hz: float) -> None:
    steps = np.array([r["step"] for r in rows])
    t = steps / control_hz
    actions = np.stack([r["action"] for r in rows])
    qpos = np.stack([r["qpos"] for r in rows])
    ee = np.stack([r["ee"] for r in rows])
    live = np.stack([r["live"] for r in rows])
    center = np.stack([r["center"] for r in rows])
    grasped = np.array([r["grasped"] for r in rows], dtype=bool)
    ee_live_dist = np.linalg.norm(ee - live, axis=1)

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
        ax.plot(t, live[:, k], color=color, ls="--", lw=1.0, alpha=0.7,
                label=f"live_{name}")
        ax.plot(t, center[:, k], color=color, ls=":", lw=1.0, alpha=0.7,
                label=f"center_{name}")
    ax.set_title("End-effector (solid) vs live (dashed) vs precise center (dotted)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.legend(fontsize=7, loc="best", ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, live[:, 2], color="C2", label="live_z")
    ax.axhline(target_height, color="k", ls="--", lw=1.0, alpha=0.6,
               label=f"target_height={target_height}")
    ax.plot(t, ee_live_dist, color="C3", label="‖ee − live‖")
    live_age = np.array([r["live_age"] for r in rows])
    precise_age = np.array([r["precise_age"] for r in rows])
    ax.plot(t, live_age * 0.1, color="C4", ls=":", lw=1.0,
            label="live_age (0.1 = 1 s)")
    ax.plot(t, precise_age * 0.1, color="C5", ls=":", lw=1.0,
            label="precise_age (0.1 = 1 s)")
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
            "marker_age_ms", "cam_read_ms", "detect_ms", "sam_ms", "dense_ms"]
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
              # marker obs the policy saw, the FK pose of the same tick's qpos,
              # and the per-tag age fed to the policy
              + [f"mobs_{t}_{ax}" for t in tag_names for ax in "xyz"]
              + [f"mfk_{t}_{ax}" for t in tag_names for ax in "xyz"]
              + [f"mage_{t}_s" for t in tag_names]
              # object channels the policy saw
              + [f"live_{ax}" for ax in "xyz"] + ["live_age_s"]
              + [f"bps_{i:02d}" for i in range(BPS_DISTANCE_DIM)]
              + [f"center_{ax}" for ax in "xyz"]
              + ["precise_age_s", "valid_fraction"]
              + ["dense_valid_count", "dense_valid_fraction",
                 "dense_correspondence_rejected_fraction",
                 "dense_overall_rejected_fraction", "dense_refreshes", "dense_misses",
                 "rig_movement_mm", "rig_movement_deg"]
              + ["loop_ms", "predict_ms", "read_all_ms", "stream_ms", "window_ms",
                 "marker_age_ms", "cam_read_ms", "detect_ms", "sam_ms", "dense_ms"])
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
                        *(f"{v:.6f}" for v in r["live"]),
                        f"{r['live_age']:.4f}",
                        *(f"{v:.6f}" for v in r["bps"]),
                        *(f"{v:.6f}" for v in r["center"]),
                        f"{r['precise_age']:.4f}",
                        f"{r['valid_fraction']:.6f}",
                        r["dense_valid_count"],
                        f"{r['dense_valid_fraction']:.6f}",
                        f"{r['dense_correspondence_rejected_fraction']:.6f}",
                        f"{r['dense_overall_rejected_fraction']:.6f}",
                        r["dense_refreshes"], r["dense_misses"],
                        f"{r['rig_movement_mm']:.4f}",
                        f"{r['rig_movement_deg']:.6f}",
                        f"{r['loop_ms']:.2f}", f"{r['predict_ms']:.3f}",
                        f"{r['read_all_ms']:.2f}", f"{r['stream_ms']:.2f}",
                        f"{r['window_ms']:.2f}",
                        f"{r['marker_age_ms']:.2f}", f"{r['cam_read_ms']:.2f}",
                        f"{r['detect_ms']:.2f}", f"{r['sam_ms']:.2f}",
                        f"{r['dense_ms']:.2f}"])


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
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert cube_geom_id >= 0, "geom 'cube_geom' not found in model"
    cube_body_id = int(model.geom_bodyid[cube_geom_id])

    bps_config = load_bps_config()
    policy = load_policy(
        args.model, LOG_DIR,
        obs_dim_for(prev_actions_n, marker_include_rot, history_taps)
        + priv_dim_for(marker_include_rot),
        bps_config=bps_config,
    )
    # Lag-tap history over the actor block, the same convention as training
    # (src/obs_history.py): seeded with the first tick's frame at boot — the
    # env's reset does the identical thing — then advanced once per tick.
    history = ObsHistory(history_taps,
                         state_dim_for(prev_actions_n, marker_include_rot))
    priv_pad = np.zeros(priv_dim_for(marker_include_rot), dtype=np.float32)

    rng = np.random.default_rng(args.seed)

    servo_bus = ServoBus(args.port, jm.servo_ids())
    loop = ArmLoop(model=model, n_substeps=n_substeps, jm=jm, bus=servo_bus,
                   action_scale=action_scale, prev_actions_n=prev_actions_n,
                   execute=args.execute, ema_alpha=args.ema_alpha,
                   slow=args.slow, interp_hz=args.interp_hz,
                   qpos_bias=load_calibration(), compliance=load_compliance())
    print(f"seed={args.seed} {loop.describe()}")

    fk_object = None
    if args.marker_source == "fk":
        fk_object = FkObjectSource(model, data, cube_geom_id, cube_body_id)
        cube_pos_init, cube_quat_init = sample_visible_fk_spawn(
            fk_object, data, cube_qposadr, rng, cube_low, cube_high)
        print(f"sim cube spawn: ({cube_pos_init[0]:+.3f}, {cube_pos_init[1]:+.3f}, "
              f"{cube_pos_init[2]:+.3f})  target_height={target_height}")
    else:
        cube_pos_init = np.array([
            np.mean((cube_low[0], cube_high[0])),
            np.mean((cube_low[1], cube_high[1])),
            model.geom_size[cube_geom_id, 2],
        ])
        cube_quat_init = np.array([1.0, 0.0, 0.0, 0.0])

    stopped = install_sigint_flag()

    frame_bus = camera_markers = camera_object = None

    viewer = None if args.no_view else mujoco.viewer.launch_passive(model, data)
    publisher = None
    if args.stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(model, args.stream_port)
    # Overlay the camera's measured marker poses and the object channels in
    # the passive viewer too, so calibration drift and the policy's object
    # belief are visible. The stream draws them itself inside
    # publisher.publish; the viewer needs the helpers directly.
    draw_channels = None
    if viewer is not None:
        from panel.sim_stream import draw_object_channels as draw_channels
    log_rows: list[dict] = []
    try:
        servo_bus.connect()
        if args.marker_source == "camera":
            from real.rollout.frame_bus import FrameBus
            from real.rollout.marker_obs import CameraMarkerSource
            from real.rollout.object_obs import ObjectSource
            from real.vision.stereo_rig import CAMERA_NAMES

            frame_bus = FrameBus(CAMERA_NAMES)
            frame_bus.start()
            camera_markers = CameraMarkerSource(frame_bus.feeds["main"])
            camera_object = ObjectSource(
                frame_bus.feeds, (cube_low, cube_high),
                prompt=args.object_prompt, sam2_model=args.sam2_model)
        loop.boot()

        # Sync the sim arm to the real arm and place the FK sponge.
        data.qpos[qposadr] = loop.qpos
        data.qvel[joint_dofadr] = 0.0
        data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos_init
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quat_init
        # Init actuators (filter state) at current qpos so they don't snap.
        data.ctrl[:6] = loop.qpos
        if model.na > 0:
            data.act[:] = loop.qpos
        mujoco.mj_forward(model, data)
        if args.marker_source == "fk":
            fk_object.boot()
        else:
            camera_markers.start()
            marker_warmup_s = camera_markers.warmup()
            print(f"camera marker anchor ready after {marker_warmup_s:.2f}s")
            camera_object.start()

        dwell_count = 0
        step = 0
        prev_iter_t = None
        # FK-branch hold-last state mirroring training (src/base_env.py): a tag
        # turned away from tag_cam keeps its last visible pose while its age
        # grows; never-yet-visible tags read zero with age at the cap.
        fk_pos = np.zeros((N_MARKERS, 3))
        fk_rot = np.zeros((N_MARKERS, 3))
        fk_seen_t = np.full(N_MARKERS, -np.inf)
        while not stopped["flag"] and step < args.max_steps:
            iter_t = time.perf_counter()
            # Realized control period (start-to-start of consecutive ticks).
            loop_ms = (iter_t - prev_iter_t) * 1e3 if prev_iter_t is not None else float("nan")
            prev_iter_t = iter_t

            # FK marker poses at the same qpos remain in camera-mode logs as a
            # direct calibration diagnostic against the measured markers.
            now = time.monotonic()
            fk_pos_now, fk_rot_now = marker_world_poses(data, marker_site_ids)
            if args.marker_source == "fk":
                vis = markers_visible(data, marker_site_ids, tag_cam)
                fk_pos[vis] = fk_pos_now[vis]
                fk_rot[vis] = fk_rot_now[vis]
                fk_seen_t[vis] = now - FK_FRESH_AGE_S
                marker_pos, marker_rot = fk_pos.copy(), fk_rot.copy()
                marker_age = np.minimum(MARKER_AGE_CAP_S, now - fk_seen_t)
                fk_object.tick()
                (live, live_age), bps_obs = fk_object.object_obs()
                marker_age_ms = cam_read_ms = detect_ms = float("nan")
                sam_ms = dense_ms = float("nan")
                dense_valid_count = 0
                dense_valid_fraction = float("nan")
                dense_correspondence_rejected_fraction = float("nan")
                dense_overall_rejected_fraction = float("nan")
                dense_refreshes = dense_misses = 0
                rig_movement_mm = rig_movement_deg = float("nan")
            else:
                marker_pos, marker_rot, marker_age = camera_markers.marker_poses()
                marker_staleness, cam_read_ms, detect_ms = camera_markers.frame_stats()
                marker_age_ms = marker_staleness * 1e3
                (live, live_age), bps_obs = camera_object.object_obs()
                object_stats = camera_object.stats()
                sam_ms = object_stats.sam_ms
                dense_ms = object_stats.dense_ms
                dense_valid_count = object_stats.valid_count
                dense_valid_fraction = object_stats.valid_fraction
                dense_correspondence_rejected_fraction = (
                    object_stats.correspondence_rejected_fraction)
                dense_overall_rejected_fraction = (
                    object_stats.overall_rejected_fraction)
                dense_refreshes = object_stats.dense_refreshes
                dense_misses = object_stats.dense_misses
                rig_movement_mm = object_stats.rig_movement_mm
                rig_movement_deg = object_stats.rig_movement_deg

            frame = build_state_frame(loop.qpos, loop.qvel, marker_pos, marker_rot,
                                      marker_age, live, live_age,
                                      loop.prev_actions, marker_include_rot)
            tapped = history.reset(frame) if step == 0 else history.push(frame)
            obs = np.concatenate([tapped, bps_obs.flat(), priv_pad])
            t_pred = time.perf_counter()
            raw_action, _ = policy.predict(obs, deterministic=True)
            predict_ms = (time.perf_counter() - t_pred) * 1e3
            action = loop.tick(raw_action)

            # Write the real arm's new state into the sim. FK mode steps the
            # sponge through contacts; camera mode only mirrors measured state.
            data.qpos[qposadr] = loop.qpos
            data.qvel[joint_dofadr] = loop.qvel
            if args.marker_source == "fk":
                data.ctrl[:6] = loop.qpos
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                data.qpos[qposadr] = loop.qpos
                data.qvel[joint_dofadr] = loop.qvel
                mujoco.mj_forward(model, data)
            else:
                if bps_obs.valid_fraction > 0.0:
                    data.qpos[cube_qposadr:cube_qposadr + 3] = bps_obs.center_base
                data.qvel[model.jnt_dofadr[cube_joint_id]:
                          model.jnt_dofadr[cube_joint_id] + 6] = 0.0
                mujoco.mj_forward(model, data)
            if publisher is not None:
                publisher.publish(
                    data,
                    None, None,
                    marker_include_rot,
                    object_channels=(live, bps_obs.center_base))

            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            ee_pos = data.site_xpos[ee_site_id].copy()

            ee_live = float(np.linalg.norm(ee_pos - live))
            gripper_val = loop.qpos[JOINT_NAMES.index("gripper")]
            grasped_sim = (args.marker_source == "fk"
                           and ee_live < 0.05 and gripper_val < 0.3)
            # Success = dwell on the live centroid height, counted only while
            # the live channel is fresh — a frozen held value must not fake a
            # lift (plan decision 9).
            if live_age < CUBE_FRESH_DWELL_S and live[2] >= target_height:
                dwell_count += 1
            else:
                dwell_count = 0

            log_rows.append({
                "step": step, "action": action.copy(), "qpos": loop.qpos.copy(),
                "ee": ee_pos.copy(), "cube": cube_pos.copy(), "grasped": grasped_sim,
                "marker_obs": marker_pos.copy(), "marker_fk": fk_pos_now.copy(),
                "marker_age": marker_age.copy(),
                "live": live.copy(), "live_age": live_age,
                "bps": bps_obs.distances.copy(),
                "center": bps_obs.center_base.copy(),
                "precise_age": bps_obs.age_s,
                "valid_fraction": bps_obs.valid_fraction,
                "dense_valid_count": dense_valid_count,
                "dense_valid_fraction": dense_valid_fraction,
                "dense_correspondence_rejected_fraction": (
                    dense_correspondence_rejected_fraction),
                "dense_overall_rejected_fraction": (
                    dense_overall_rejected_fraction),
                "dense_refreshes": dense_refreshes,
                "dense_misses": dense_misses,
                "rig_movement_mm": rig_movement_mm,
                "rig_movement_deg": rig_movement_deg,
                "loop_ms": loop_ms, "predict_ms": predict_ms,
                "read_all_ms": loop.last_read_ms, "stream_ms": loop.last_stream_ms,
                "window_ms": loop.last_window_ms,
                "marker_age_ms": marker_age_ms, "cam_read_ms": cam_read_ms,
                "detect_ms": detect_ms, "sam_ms": sam_ms, "dense_ms": dense_ms,
            })

            if viewer is not None:
                viewer.user_scn.ngeom = 0
                draw_channels(viewer.user_scn, live, bps_obs.center_base)
                viewer.sync()
                if not viewer.is_running():
                    print("Viewer closed; stopping rollout.")
                    break

            if step % 15 == 0:
                print(f"step={step:3d}  ee-live={ee_live:.3f}m  "
                      f"live_z={live[2]:.3f}m  ages live={live_age:.2f}s "
                      f"precise={bps_obs.age_s:.2f}s  grasped_sim={int(grasped_sim)}")
                lat = (f"          lat[ms]: loop={loop_ms:.0f} predict={predict_ms:.2f} "
                       f"read_all={loop.last_read_ms:.0f} stream={loop.last_stream_ms:.0f} "
                       f"window={loop.last_window_ms:.0f}")
                print(lat)
                if args.marker_source == "camera":
                    print(
                        f"          dense: points={dense_valid_count} "
                        f"valid={bps_obs.valid_fraction:.1%} "
                        f"reject={dense_correspondence_rejected_fraction:.1%}/"
                        f"{dense_overall_rejected_fraction:.1%} "
                        f"refresh/miss={dense_refreshes}/{dense_misses}")
            step += 1

            if dwell_count >= 5:
                print(f"live centroid held above target_height={target_height} "
                      f"at step {step}")
                break
        else:
            if not stopped["flag"]:
                print(f"TIMEOUT at step {step}")
    finally:
        if camera_object is not None:
            camera_object.stop()
        if camera_markers is not None:
            camera_markers.stop()
        if frame_bus is not None:
            frame_bus.stop()
        servo_bus.close()
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
