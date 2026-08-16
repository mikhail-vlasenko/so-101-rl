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
    python -m real.rollout.rollout_lift --marker-source camera --interactive --execute

Setup, safety gating, and per-tick command shaping (training-matched
quantization, raw clamp, sub-target streaming, --slow time dilation) all live
in real.rollout.rollout_common — this script owns only observation construction,
termination, and plots. --execute is OFF by default; Ctrl-C disables torque.
In camera-mode ``--interactive``, the process keeps its model and camera stack
warm, revalidates the stereo placement before every episode, and runs another
episode whenever Enter is pressed. A failed initial object prompt asks for Enter
to retry. At the warm prompt, ``e`` switches the next episode between execute
and dry-run while preserving the CLI's initial mode, and ``r`` parks the arm
when execute mode is selected. A normally completed execute episode also
gently parks at the folded rest pose before disabling torque. Ctrl-C during an
episode or rest move stops immediately and disables torque; an interrupted
episode saves its partial log and returns to the warm prompt. Ctrl-C at either
interactive prompt exits the session.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.base_env import (
    CameraModel,
    JOINT_NAMES,
    MARKER_AGE_CAP_S,
    MARKER_SITE_NAMES,
    N_MARKERS,
    TAG_CAM_AUX_NAME,
    cube_surface_points_world,
    cube_visible_surface,
    ee_object_delta,
    marker_world_poses,
    markers_visible,
    obs_dim_for,
    priv_dim_for,
    state_dim_for,
    sample_cube_orientation,
    set_marker_render_colors,
    tag_cam_model,
)
from src.obs_history import ObsHistory
from src.bps import BPS_DISTANCE_DIM, BPSObservation, BPSObsState, load_bps_config
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
from ..twin.constants import FOLDED_REST_QPOS
from ..twin.mapping import JointMaps, load_joint_maps
from ..twin.servo_io import ServoBus

if TYPE_CHECKING:
    from stable_baselines3 import PPO

    from panel.sim_stream import SimStreamPublisher
    from real.rollout.frame_bus import FrameBus
    from real.rollout.marker_obs import StereoCameraMarkerSource
    from real.rollout.object_obs import ObjectSource

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
        self._latest_cloud = np.empty((0, 3), dtype=np.float64)

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
            if capture is not None:
                self._latest_cloud = capture.points_base.copy()

    def object_obs(self):
        now = time.monotonic()
        return self.driver.serve(now), self.bps_state.serve(now)

    def latest_cloud(self) -> np.ndarray:
        return self._latest_cloud.copy()


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
    p.add_argument("--interactive", action="store_true",
                   help="Keep the camera/model stack warm and run another "
                        "camera episode each time Enter is pressed (terminal only).")
    args = p.parse_args()
    if args.interactive and args.marker_source != "camera":
        p.error("--interactive requires --marker-source camera")
    if args.interactive and not sys.stdin.isatty():
        p.error("--interactive requires an attached terminal")
    return args


def build_state_frame(qpos: np.ndarray, qvel: np.ndarray, marker_pos: np.ndarray,
                      marker_rot: np.ndarray, marker_age: np.ndarray,
                      live: np.ndarray, live_age: float,
                      prev_actions: np.ndarray, ee_pos: np.ndarray,
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
    return np.concatenate([
        qpos.astype(np.float32),
        qvel.astype(np.float32),
        markers.astype(np.float32),
        marker_age.astype(np.float32),
        live.astype(np.float32),
        np.array([live_age], dtype=np.float32),
        extra,
        prev_actions.flatten().astype(np.float32),
        ee_object_delta(ee_pos, live).astype(np.float32),
    ]).astype(np.float32)


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


@dataclass(frozen=True)
class LiftRuntimeConfig:
    action_scale: float
    n_substeps: int
    cube_low: np.ndarray
    cube_high: np.ndarray
    target_height: float
    success_dwell_steps: int
    rest_qpos: np.ndarray
    rest_duration_s: float
    rest_action_scale: float
    rest_settle_s: float
    prev_actions_n: int
    marker_include_rot: bool
    history_taps: tuple[int, ...]


@dataclass(frozen=True)
class SceneHandles:
    model: mujoco.MjModel
    data: mujoco.MjData
    qposadr: np.ndarray
    joint_dofadr: np.ndarray
    ee_site_id: int
    marker_site_ids: tuple[int, ...]
    tag_cam: CameraModel
    cube_qposadr: int
    cube_geom_id: int
    cube_body_id: int


@dataclass(frozen=True)
class TickObservation:
    marker_pos: np.ndarray
    marker_rot: np.ndarray
    marker_age: np.ndarray
    marker_detected: np.ndarray
    marker_fk: np.ndarray
    live: np.ndarray
    live_age: float
    bps: BPSObservation
    point_cloud: np.ndarray
    marker_age_ms: float
    cam_read_ms: float
    detect_ms: float
    sam_ms: float
    dense_ms: float
    dense_valid_count: int
    dense_valid_fraction: float
    correspondence_rejected_fraction: float
    overall_rejected_fraction: float
    dense_refreshes: int
    dense_misses: int
    rig_movement_mm: float
    rig_movement_deg: float


@dataclass(frozen=True)
class EpisodeResult:
    rows: list[dict]
    interrupted: bool
    viewer_closed: bool


def build_scene_handles(
        xml_path: str, calibration_path: str) -> tuple[SceneHandles, JointMaps]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_maps = load_joint_maps(model, Path(calibration_path))
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ]
    joint_dofadr = model.jnt_dofadr[joint_ids]
    ee_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_site_id >= 0, "site 'gripperframe' not found in model"
    marker_site_ids = []
    for name in MARKER_SITE_NAMES:
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert site_id >= 0, f"site '{name}' not found in model"
        marker_site_ids.append(site_id)
    cube_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert cube_geom_id >= 0, "geom 'cube_geom' not found in model"
    scene = SceneHandles(
        model=model,
        data=data,
        qposadr=joint_maps.qposadr(),
        joint_dofadr=joint_dofadr,
        ee_site_id=ee_site_id,
        marker_site_ids=tuple(marker_site_ids),
        tag_cam=tag_cam_model(model, data),
        cube_qposadr=int(model.jnt_qposadr[cube_joint_id]),
        cube_geom_id=cube_geom_id,
        cube_body_id=int(model.geom_bodyid[cube_geom_id]),
    )
    return scene, joint_maps


class LiftRolloutSession:
    """One warm process containing many independently reset lift episodes."""

    def __init__(self, args: argparse.Namespace,
                 config: LiftRuntimeConfig) -> None:
        self.args = args
        self.config = config
        self.scene, joint_maps = build_scene_handles(args.xml, args.cal)
        self.policy: PPO = load_policy(
            args.model,
            LOG_DIR,
            obs_dim_for(
                config.prev_actions_n,
                config.marker_include_rot,
                config.history_taps,
            ) + priv_dim_for(config.marker_include_rot),
            bps_config=load_bps_config(),
        )
        self.history = ObsHistory(
            config.history_taps,
            state_dim_for(config.prev_actions_n, config.marker_include_rot),
        )
        self.priv_pad = np.zeros(
            priv_dim_for(config.marker_include_rot), dtype=np.float32)
        self.servo_bus = ServoBus(args.port, joint_maps.servo_ids())
        self.loop = ArmLoop(
            model=self.scene.model,
            n_substeps=config.n_substeps,
            jm=joint_maps,
            bus=self.servo_bus,
            action_scale=config.action_scale,
            prev_actions_n=config.prev_actions_n,
            execute=args.execute,
            ema_alpha=args.ema_alpha,
            slow=args.slow,
            interp_hz=args.interp_hz,
            qpos_bias=load_calibration(),
            compliance=load_compliance(),
        )
        print(f"seed={args.seed} {self.loop.describe()}")

        self.fk_object: FkObjectSource | None = None
        self.cube_pos_init, self.cube_quat_init = self._initial_object_pose()
        self.frame_bus: FrameBus | None = None
        self.camera_markers: StereoCameraMarkerSource | None = None
        self.camera_object: ObjectSource | None = None
        self.viewer = None
        self.publisher: SimStreamPublisher | None = None
        self._draw_detected_markers: Callable | None = None
        self._draw_channels: Callable | None = None
        self._draw_cloud: Callable | None = None

    def _initial_object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.args.marker_source == "fk":
            self.fk_object = FkObjectSource(
                self.scene.model,
                self.scene.data,
                self.scene.cube_geom_id,
                self.scene.cube_body_id,
            )
            position, quaternion = sample_visible_fk_spawn(
                self.fk_object,
                self.scene.data,
                self.scene.cube_qposadr,
                np.random.default_rng(self.args.seed),
                self.config.cube_low,
                self.config.cube_high,
            )
            print(
                f"sim cube spawn: ({position[0]:+.3f}, {position[1]:+.3f}, "
                f"{position[2]:+.3f})  "
                f"target_height={self.config.target_height}")
            return position, quaternion

        position = np.array([
            np.mean((self.config.cube_low[0], self.config.cube_high[0])),
            np.mean((self.config.cube_low[1], self.config.cube_high[1])),
            self.scene.model.geom_size[self.scene.cube_geom_id, 2],
        ])
        self.scene.model.geom_rgba[self.scene.cube_geom_id, 3] = 0.0
        return position, np.array([1.0, 0.0, 0.0, 0.0])

    def start(self) -> bool:
        """Open session resources and warm the camera/model pipeline once."""
        self.servo_bus.connect()
        if not self.args.no_view:
            self.viewer = mujoco.viewer.launch_passive(
                self.scene.model, self.scene.data)
            from panel.sim_stream import (
                draw_detected_markers,
                draw_object_channels,
                draw_point_cloud,
            )
            self._draw_detected_markers = draw_detected_markers
            self._draw_channels = draw_object_channels
            self._draw_cloud = draw_point_cloud
        if self.args.stream_port is not None:
            from panel.sim_stream import SimStreamPublisher
            self.publisher = SimStreamPublisher(
                self.scene.model, self.args.stream_port)
        if self.args.marker_source == "camera":
            return self._start_camera_pipeline()
        return True

    def _start_camera_pipeline(self) -> bool:
        from real.rollout.frame_bus import FrameBus
        from real.rollout.marker_obs import StereoCameraMarkerSource
        from real.rollout.object_obs import ObjectSource
        from real.vision.stereo_rig import CAMERA_NAMES

        self.frame_bus = FrameBus(CAMERA_NAMES)
        self.frame_bus.start()
        self.camera_markers = StereoCameraMarkerSource(self.frame_bus.feeds)
        self.camera_object = ObjectSource(
            self.frame_bus.feeds,
            (self.config.cube_low, self.config.cube_high),
            prompt=self.args.object_prompt,
            sam2_model=self.args.sam2_model,
        )
        self.camera_markers.start()
        marker_warmup_s = self.camera_markers.warmup()
        print(f"camera marker anchor ready after {marker_warmup_s:.2f}s")
        if not self._start_object_source():
            return False
        if self.args.interactive:
            print("\nWarm rollout session ready; torque is disabled.")
        return True

    def _start_object_source(self) -> bool:
        from real.tracking.sam_seg import SAMPromptNoMatchError

        assert self.camera_object is not None
        while True:
            try:
                self.camera_object.start()
                return True
            except SAMPromptNoMatchError as error:
                if not self.args.interactive:
                    raise
                if not self._prompt_for_object_retry(error):
                    return False

    def close(self) -> None:
        """Release all resources retained by the warm session."""
        self.servo_bus.close()
        if self.camera_object is not None:
            self.camera_object.stop()
        if self.camera_markers is not None:
            self.camera_markers.stop()
        if self.frame_bus is not None:
            self.frame_bus.stop()
        if self.viewer is not None:
            self.viewer.close()
        if self.publisher is not None:
            self.publisher.close()

    def run(self) -> None:
        episode_index = 0
        while True:
            if self.args.interactive:
                if not self._prompt_for_episode(episode_index):
                    return
                self._prepare_camera_episode()

            stopped = install_sigint_flag()
            result = self._run_episode(stopped)
            if self.args.interactive:
                signal.signal(signal.SIGINT, signal.default_int_handler)
            self._save_episode(result.rows, episode_index)

            if result.viewer_closed or not self.args.interactive:
                return
            if result.interrupted:
                print("Episode interrupted; returning to the warm prompt.")
            episode_index += 1

    @staticmethod
    def _prompt_for_object_retry(error: RuntimeError) -> bool:
        while True:
            try:
                command = input(
                    f"\n{error}. Reposition the sponge, then "
                    "Enter=retry, Ctrl-C=quit > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopping warm rollout session.")
                return False
            if command in ("", "retry"):
                return True
            print("Enter retries object acquisition; Ctrl-C quits the warm session.")

    def _prompt_for_episode(self, episode_index: int) -> bool:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        while True:
            mode = "EXECUTE" if self.loop.execute else "DRY-RUN"
            try:
                command = input(
                    f"\n[episode {episode_index + 1}] [{mode}] "
                    "Enter=run, e=toggle execute, r=rest, Ctrl-C=quit > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopping warm rollout session.")
                return False
            if command in ("", "run"):
                return True
            if command in ("e", "execute", "toggle"):
                self.loop.set_execute(not self.loop.execute)
                next_mode = "EXECUTE" if self.loop.execute else "DRY-RUN"
                print(f"Next episode mode: {next_mode}.")
                continue
            if command in ("r", "rest", "park"):
                self._return_to_rest_from_prompt()
                continue
            print("Enter starts an episode; e toggles execute; r parks; Ctrl-C quits.")

    def _return_to_rest_from_prompt(self) -> None:
        if not self.loop.execute:
            print("Rest move is disabled in DRY-RUN; press e to enable execute.")
            return

        stopped = install_sigint_flag()
        booted = False
        try:
            self.loop.boot()
            booted = True
            self._return_to_rest(stopped)
        finally:
            if booted:
                self.loop.end_episode()
            signal.signal(signal.SIGINT, signal.default_int_handler)
        if stopped["flag"]:
            print("Rest move interrupted; torque disabled.")
        else:
            print("Arm at folded rest; torque disabled.")

    def _return_to_rest(self, stopped: dict) -> bool:
        if self.loop.execute:
            print("Returning arm to folded rest pose...")
        return self.loop.return_to_rest(
            self.config.rest_qpos,
            self.config.rest_duration_s,
            self.config.rest_action_scale,
            self.config.rest_settle_s,
            lambda: bool(stopped["flag"]),
        )

    def _prepare_camera_episode(self) -> None:
        assert self.camera_markers is not None
        assert self.camera_object is not None
        marker_warmup_s = self.camera_markers.warmup()
        print(f"episode marker anchors ready after {marker_warmup_s:.2f}s")
        self.camera_object.prepare_episode()

    def _run_episode(self, stopped: dict) -> EpisodeResult:
        rows: list[dict] = []
        viewer_closed = False
        booted = False
        completed_normally = False
        try:
            self.loop.boot()
            booted = True
            self._reset_episode_scene()
            dwell_count = 0
            step = 0
            prev_iter_t = None
            fk_pos = np.zeros((N_MARKERS, 3))
            fk_rot = np.zeros((N_MARKERS, 3))
            fk_seen_t = np.full(N_MARKERS, -np.inf)

            while not stopped["flag"] and step < self.args.max_steps:
                iter_t = time.perf_counter()
                loop_ms = ((iter_t - prev_iter_t) * 1e3
                           if prev_iter_t is not None else float("nan"))
                prev_iter_t = iter_t
                tick = self._capture_tick(fk_pos, fk_rot, fk_seen_t)
                action, predict_ms = self._predict_and_tick(tick, step)
                self._advance_scene(tick)

                cube_pos = self.scene.data.qpos[
                    self.scene.cube_qposadr:
                    self.scene.cube_qposadr + 3].copy()
                ee_pos = self.scene.data.site_xpos[
                    self.scene.ee_site_id].copy()
                ee_live = float(np.linalg.norm(ee_pos - tick.live))
                gripper = self.loop.qpos[JOINT_NAMES.index("gripper")]
                grasped_sim = (
                    self.args.marker_source == "fk"
                    and ee_live < 0.05 and gripper < 0.3)
                if (tick.live_age < CUBE_FRESH_DWELL_S
                        and tick.live[2] >= self.config.target_height):
                    dwell_count += 1
                else:
                    dwell_count = 0

                rows.append(self._log_row(
                    step, tick, action, cube_pos, ee_pos, grasped_sim,
                    loop_ms, predict_ms))
                viewer_closed = self._publish_and_render(tick)
                if viewer_closed:
                    print("Viewer closed; stopping rollout session.")
                    break
                if step % 15 == 0:
                    self._print_step(
                        step, tick, ee_live, grasped_sim, loop_ms, predict_ms)
                step += 1
                if dwell_count >= self.config.success_dwell_steps:
                    print(
                        "live centroid held above "
                        f"target_height={self.config.target_height} "
                        f"at step {step}")
                    break
            else:
                if not stopped["flag"]:
                    print(f"TIMEOUT at step {step}")
            completed_normally = True
        finally:
            if booted:
                try:
                    if completed_normally and not stopped["flag"]:
                        self._return_to_rest(stopped)
                finally:
                    self.loop.end_episode()
                if self.args.interactive:
                    print("Episode finished; torque disabled.")
        return EpisodeResult(
            rows=rows,
            interrupted=bool(stopped["flag"]),
            viewer_closed=viewer_closed,
        )

    def _reset_episode_scene(self) -> None:
        data = self.scene.data
        data.qpos[self.scene.qposadr] = self.loop.qpos
        data.qvel[self.scene.joint_dofadr] = 0.0
        start = self.scene.cube_qposadr
        data.qpos[start:start + 3] = self.cube_pos_init
        data.qpos[start + 3:start + 7] = self.cube_quat_init
        data.ctrl[:6] = self.loop.qpos
        if self.scene.model.na > 0:
            data.act[:] = self.loop.qpos
        mujoco.mj_forward(self.scene.model, data)
        if self.fk_object is not None:
            self.fk_object.boot()

    def _capture_tick(self, fk_pos: np.ndarray, fk_rot: np.ndarray,
                      fk_seen_t: np.ndarray) -> TickObservation:
        now = time.monotonic()
        fk_pos_now, fk_rot_now = marker_world_poses(
            self.scene.data, self.scene.marker_site_ids)
        if self.args.marker_source == "fk":
            assert self.fk_object is not None
            visible = markers_visible(
                self.scene.data,
                self.scene.marker_site_ids,
                self.scene.tag_cam,
            )
            fk_pos[visible] = fk_pos_now[visible]
            fk_rot[visible] = fk_rot_now[visible]
            fk_seen_t[visible] = now - FK_FRESH_AGE_S
            self.fk_object.tick()
            (live, live_age), bps = self.fk_object.object_obs()
            return TickObservation(
                marker_pos=fk_pos.copy(),
                marker_rot=fk_rot.copy(),
                marker_age=np.minimum(MARKER_AGE_CAP_S, now - fk_seen_t),
                marker_detected=visible,
                marker_fk=fk_pos_now,
                live=live,
                live_age=live_age,
                bps=bps,
                point_cloud=self.fk_object.latest_cloud(),
                marker_age_ms=float("nan"),
                cam_read_ms=float("nan"),
                detect_ms=float("nan"),
                sam_ms=float("nan"),
                dense_ms=float("nan"),
                dense_valid_count=0,
                dense_valid_fraction=float("nan"),
                correspondence_rejected_fraction=float("nan"),
                overall_rejected_fraction=float("nan"),
                dense_refreshes=0,
                dense_misses=0,
                rig_movement_mm=float("nan"),
                rig_movement_deg=float("nan"),
            )

        assert self.camera_markers is not None
        assert self.camera_object is not None
        marker_pos, marker_rot, marker_age, detected = (
            self.camera_markers.marker_observation())
        marker_staleness, cam_read_ms, detect_ms = (
            self.camera_markers.frame_stats())
        (live, live_age), bps = self.camera_object.object_obs()
        stats = self.camera_object.stats()
        return TickObservation(
            marker_pos=marker_pos,
            marker_rot=marker_rot,
            marker_age=marker_age,
            marker_detected=detected,
            marker_fk=fk_pos_now,
            live=live,
            live_age=live_age,
            bps=bps,
            point_cloud=self.camera_object.latest_cloud(),
            marker_age_ms=marker_staleness * 1e3,
            cam_read_ms=cam_read_ms,
            detect_ms=detect_ms,
            sam_ms=stats.sam_ms,
            dense_ms=stats.dense_ms,
            dense_valid_count=stats.valid_count,
            dense_valid_fraction=stats.valid_fraction,
            correspondence_rejected_fraction=(
                stats.correspondence_rejected_fraction),
            overall_rejected_fraction=stats.overall_rejected_fraction,
            dense_refreshes=stats.dense_refreshes,
            dense_misses=stats.dense_misses,
            rig_movement_mm=stats.rig_movement_mm,
            rig_movement_deg=stats.rig_movement_deg,
        )

    def _predict_and_tick(self, tick: TickObservation,
                          step: int) -> tuple[np.ndarray, float]:
        frame = build_state_frame(
            self.loop.qpos,
            self.loop.qvel,
            tick.marker_pos,
            tick.marker_rot,
            tick.marker_age,
            tick.live,
            tick.live_age,
            self.loop.prev_actions,
            self.scene.data.site_xpos[self.scene.ee_site_id],
            self.config.marker_include_rot,
        )
        tapped = (self.history.reset(frame) if step == 0
                  else self.history.push(frame))
        observation = np.concatenate([tapped, tick.bps.flat(), self.priv_pad])
        started = time.perf_counter()
        raw_action, _ = self.policy.predict(observation, deterministic=True)
        predict_ms = (time.perf_counter() - started) * 1e3
        return self.loop.tick(raw_action), predict_ms

    def _advance_scene(self, tick: TickObservation) -> None:
        data = self.scene.data
        data.qpos[self.scene.qposadr] = self.loop.qpos
        data.qvel[self.scene.joint_dofadr] = self.loop.qvel
        if self.args.marker_source == "fk":
            data.ctrl[:6] = self.loop.qpos
            for _ in range(self.config.n_substeps):
                mujoco.mj_step(self.scene.model, data)
            data.qpos[self.scene.qposadr] = self.loop.qpos
            data.qvel[self.scene.joint_dofadr] = self.loop.qvel
        mujoco.mj_forward(self.scene.model, data)
        set_marker_render_colors(
            self.scene.model,
            self.scene.marker_site_ids,
            tick.marker_detected,
        )

    def _publish_and_render(self, tick: TickObservation) -> bool:
        if self.publisher is not None:
            self.publisher.publish(
                self.scene.data,
                tick.marker_pos if self.args.marker_source == "camera" else None,
                tick.marker_rot if self.args.marker_source == "camera" else None,
                self.config.marker_include_rot,
                object_channels=(tick.live, tick.bps.center_base),
                point_cloud=tick.point_cloud,
            )
        if self.viewer is None:
            return False
        assert self._draw_detected_markers is not None
        assert self._draw_channels is not None
        assert self._draw_cloud is not None
        self.viewer.user_scn.ngeom = 0
        if self.args.marker_source == "camera":
            self._draw_detected_markers(
                self.viewer.user_scn,
                tick.marker_pos,
                tick.marker_rot,
                self.config.marker_include_rot,
            )
        self._draw_channels(
            self.viewer.user_scn, tick.live, tick.bps.center_base)
        self._draw_cloud(self.viewer.user_scn, tick.point_cloud)
        self.viewer.sync()
        return not self.viewer.is_running()

    def _log_row(self, step: int, tick: TickObservation,
                 action: np.ndarray, cube_pos: np.ndarray,
                 ee_pos: np.ndarray, grasped_sim: bool,
                 loop_ms: float, predict_ms: float) -> dict:
        return {
            "step": step,
            "action": action.copy(),
            "qpos": self.loop.qpos.copy(),
            "ee": ee_pos.copy(),
            "cube": cube_pos.copy(),
            "grasped": grasped_sim,
            "marker_obs": tick.marker_pos.copy(),
            "marker_fk": tick.marker_fk.copy(),
            "marker_age": tick.marker_age.copy(),
            "live": tick.live.copy(),
            "live_age": tick.live_age,
            "bps": tick.bps.distances.copy(),
            "center": tick.bps.center_base.copy(),
            "precise_age": tick.bps.age_s,
            "valid_fraction": tick.bps.valid_fraction,
            "dense_valid_count": tick.dense_valid_count,
            "dense_valid_fraction": tick.dense_valid_fraction,
            "dense_correspondence_rejected_fraction": (
                tick.correspondence_rejected_fraction),
            "dense_overall_rejected_fraction": tick.overall_rejected_fraction,
            "dense_refreshes": tick.dense_refreshes,
            "dense_misses": tick.dense_misses,
            "rig_movement_mm": tick.rig_movement_mm,
            "rig_movement_deg": tick.rig_movement_deg,
            "loop_ms": loop_ms,
            "predict_ms": predict_ms,
            "read_all_ms": self.loop.last_read_ms,
            "stream_ms": self.loop.last_stream_ms,
            "window_ms": self.loop.last_window_ms,
            "marker_age_ms": tick.marker_age_ms,
            "cam_read_ms": tick.cam_read_ms,
            "detect_ms": tick.detect_ms,
            "sam_ms": tick.sam_ms,
            "dense_ms": tick.dense_ms,
        }

    def _print_step(self, step: int, tick: TickObservation,
                    ee_live: float, grasped_sim: bool,
                    loop_ms: float, predict_ms: float) -> None:
        print(
            f"step={step:3d}  ee-live={ee_live:.3f}m  "
            f"live_z={tick.live[2]:.3f}m  ages live={tick.live_age:.2f}s "
            f"precise={tick.bps.age_s:.2f}s  "
            f"grasped_sim={int(grasped_sim)}")
        print(
            f"          lat[ms]: loop={loop_ms:.0f} "
            f"predict={predict_ms:.2f} "
            f"read_all={self.loop.last_read_ms:.0f} "
            f"stream={self.loop.last_stream_ms:.0f} "
            f"window={self.loop.last_window_ms:.0f}")
        if self.args.marker_source == "camera":
            print(
                f"          dense: points={tick.dense_valid_count} "
                f"valid={tick.bps.valid_fraction:.1%} "
                f"reject={tick.correspondence_rejected_fraction:.1%}/"
                f"{tick.overall_rejected_fraction:.1%} "
                f"refresh/miss={tick.dense_refreshes}/{tick.dense_misses}")

    def _save_episode(self, rows: list[dict], episode_index: int) -> None:
        if not rows:
            return
        print_latency_summary(rows)
        out_dir = REPO_ROOT / "rollouts"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_e{episode_index + 1}" if self.args.interactive else ""
        stem = f"rollout_lift_{int(time.time())}{suffix}"
        csv_path = out_dir / f"{stem}.csv"
        plot_path = out_dir / f"{stem}.png"
        write_csv(csv_path, rows, self.loop.control_hz)
        plot_rollout(
            plot_path, rows, self.config.target_height, self.loop.control_hz)
        print(
            f"saved {csv_path.relative_to(REPO_ROOT)} "
            f"{plot_path.relative_to(REPO_ROOT)}")


def main() -> int:
    lift_cfg, prev_actions_n, marker_include_rot, history_taps = load_env_cfg("lift")
    args = parse_args(lift_cfg)
    config = LiftRuntimeConfig(
        action_scale=float(lift_cfg["action_scale"]),
        n_substeps=int(lift_cfg["n_substeps"]),
        cube_low=np.array(lift_cfg["cube_low"], dtype=np.float64),
        cube_high=np.array(lift_cfg["cube_high"], dtype=np.float64),
        target_height=float(lift_cfg["target_height"]),
        success_dwell_steps=int(lift_cfg["success_dwell_steps"]),
        rest_qpos=np.array(FOLDED_REST_QPOS, dtype=np.float64),
        rest_duration_s=float(lift_cfg["rest_duration_s"]),
        rest_action_scale=float(lift_cfg["rest_action_scale"]),
        rest_settle_s=float(lift_cfg["rest_settle_s"]),
        prev_actions_n=prev_actions_n,
        marker_include_rot=marker_include_rot,
        history_taps=history_taps,
    )
    session = LiftRolloutSession(args, config)
    try:
        if session.start():
            session.run()
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
