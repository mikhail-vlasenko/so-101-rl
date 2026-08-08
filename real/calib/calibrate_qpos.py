"""Self-driven encoder-bias + camera-extrinsics calibration from the arm tags.

It solves the per-joint encoder bias `b` *jointly* with the camera pose so
FK(theta_enc - b) lands the arm tags where the camera says they are, then re-derives
the table-tag anchor with the corrected kinematics. (Trusting the raw encoders to
register the camera instead leaves ~11 mm of arm-tag residual dominated by
uncalibrated zero-offsets — commit 888c3bc — which this closes.) One pass writes both
`extrinsics.yaml` (`t_base_cam_fixed`, `table_anchors`, `quarter_turns`) and
`calibration.yaml` (`qpos_bias`).

How it works:
  - The arm drives *itself* through a deterministic Cartesian sweep: a grid of
    end-effector targets hugging the table in front of the base at pan=middle, plus
    blocks at panned angles (<=35 deg — past that a tag leaves the camera frame). The
    band sits low on purpose: the bias warps the markers most during a grasp, so the
    samples are concentrated where the gripper meets objects on the table (markers at
    ~0.06-0.10 m), not up high where a fit extrapolates poorly to contact. Per target,
    IK over the four non-base joints both reaches the point and aims *both* arm tags
    at the camera; that coupling moves every joint between nearby poses, so all
    observable biases are excited. Each pose is verified collision-free and
    both-tags-visible in sim (including the marker_visible height ceiling that models
    the real camera's top-of-frame cut-off) before driving.
  - The kept poses are re-ordered for drive safety (a greedy nearest-neighbour walk
    keyed on the EE-height-driving joints, starting from the highest pose) and that
    ordering is verified collision-free along every straight-line interpolation in
    sim, so the homing ramp between two low samples can never dive into the table.
  - Before the final solve it rejects outlier arm-tag detections: a single tag
    mis-detected at one pose (solvePnP fluke / motion blur) is dropped on a robust
    residual cutoff (against the compliance-aware forward model, so a real per-pose
    outlier can't hide in the looser bias-only field) and the fit re-run, so one bad
    point can't skew it.

The dry-run (no --execute) previews the sweep in the sim viewer (one pose/second),
so the trajectory can be eyeballed before any hardware moves. With --stream-port
(set automatically by the panel), the dry-run streams that sim preview and the
--execute run streams the live annotated camera feed — same MJPEG endpoint, so the
panel's sim-view slot shows whichever is running.

Gravity compliance: on top of the constant encoder bias, the lift/elbow/wrist_flex
links flex elastically under gravity load, so the true joint angle lags the encoder by
`compliance * tau_grav` (per-joint, rad per N.m). Those three coefficients are solved
jointly with the bias and mounts (see real/calib/compliance.py, COMP_JOINTS) and written to
calibration.yaml; deployment applies the same correction. It roughly halves the
held-out marker residual (cross-validated 3.6 -> 2.0 mm mean, 8.2 -> 4.7 mm max).

Position-only throughout (tag centres / tvec), so the solve is immune to solvePnP
rvec flips on the small arm tags.

What is and isn't observable (position-only, single fixed camera):
  - lift / elbow / wrist_flex / wrist_roll biases each warp the tag positions in a
    pose-dependent way a rigid camera transform cannot absorb -> solved.
  - shoulder_pan bias is a rotation about the base z-axis through the base origin,
    exactly what a yaw of `T_base_cam` about the same axis reproduces: the two are
    a gauge degeneracy, unobservable from these tags. We pin pan bias to 0 and let
    the camera absorb it; deployment maps markers through the *same* `T_base_cam`,
    so the base frame stays self-consistent (a constant pan offset is harmless for
    a camera-relative task).
  - gripper bias moves neither tag (both sit on links proximal to the gripper
    joint) -> pinned to 0.

Freed tag mounts: each arm tag's XML site position starts as a by-hand estimate of where
the tag is taped, so its 3D centre offset is solved jointly with the bias (strong XML
prior; see MOUNT_PRIOR_W) to keep a mis-taped tag from being absorbed as a fake encoder
offset. The solved site centres are written back into so101/so101.xml: training's marker
observations are FK of those sites (src/base_env.py), so a site that doesn't match the
physical tag is a systematic sim-vs-real offset in the marker channels (measured ~1.5 mm
mean / ~3 mm at the tails before this write-back existed).

Run:
    conda run -n mujoco_env python -m real.calib.calibrate_qpos --execute
    conda run -n mujoco_env python -m real.calib.calibrate_qpos              # dry-run: preview poses
    conda run -n mujoco_env python -m real.calib.calibrate_qpos --from-samples <json>
"""
import argparse
import re
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import least_squares

from real.calib.calibrate_camera import FOCUS_ABSOLUTE
from real.calib.calib_solve import (
    determine_quarter_turns,
    load_samples,
    save_samples,
    sim_cam_R_opencv,
    solve_table_anchors,
)
from real.calib.calibration import CALIBRATION_PATH, save_calibration
from real.vision.camera import open_camera
from real.calib.compliance import COMP_JOINTS, gravity_deflection
from real.vision.detect import make_detector
from real.calib.extrinsics import EXTRINSICS_PATH, rigid_register, save_extrinsics
from real.marker_spec import (
    ARM_TAG_TO_SITE,
    MARKER_EXPOSURE,
    MARKER_GAIN,
    TABLE_TAG_IDS,
)
from real.vision.overlay import annotate_detections
from real.vision.pose import PoseEstimator, load_intrinsics
from real.twin.constants import (
    SERVO_POSITION_DEADZONE,
    SERVO_POSITION_KP,
    SERVO_TORQUE_LIMIT,
)
from real.twin.mapping import JOINT_NAMES, load_joint_maps, raw_to_rad
from real.twin.servo_io import ServoBus
from src.base_env import MARKER_SITE_NAMES, markers_visible, tag_cam_model
from src.units import max_raw_delta_per_step
from sysid.record_real import CONFIG_YAML, HOMING_S, SETTLE_S, smooth_ramp, stream
from sysid.trajectories import SYSID_HZ

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
# The marker sites are defined in the arm XML (included by DEFAULT_XML); the solved
# mounts are written back there so training FK matches the physical tags.
ARM_XML = REPO_ROOT / "so101" / "so101.xml"
DEFAULT_CAL = REPO_ROOT / "real" / "follower_calibration.json"
SAMPLES_PATH = REPO_ROOT / "real" / "qpos_calib_samples.json"
MIN_SAMPLES = 8

# qpos_bias indices solved jointly with the camera; pan (0) and gripper (5) are
# pinned to 0 (see module docstring on observability).
OBSERVABLE_JOINTS = (1, 2, 3, 4)

# Freed tag mounts. The arm-tag XML site positions start as a CAD/by-hand estimate of
# where the tags are physically taped, good to a few mm at best. Solving each tag's 3D
# centre offset (in its parent body frame) jointly with the bias de-confounds a mis-taped
# tag from a real encoder offset, so both the bias and the camera fit cleaner. The solved
# site centres are persisted into ARM_XML (write_marker_sites): deployment reads the
# *camera-measured* tag pose, but training's marker obs are FK of the sites, so the XML
# must track the physical tags or the policy sees a systematic sim-vs-real marker offset.
# A strong prior toward the XML is mandatory and resolves a gauge: wrist_roll is
# the last joint before the finger tag's body and marker_wrist is proximal to it, so the
# finger tag is the *only* observer of roll — and a constant roll bias delta is exactly a
# constant finger mount offset (Rz(delta)·p − p in the gripper frame). The two are
# interchangeable; with the prior off the solver splits them arbitrarily (wrist_roll ran
# to −151 deg in a synthetic check). The prior picks the gauge — trust the XML mount for
# that one direction, leaving roll bias observable — while still letting the offset's two
# non-degenerate directions (and the fully-observable wrist tag) move a few mm.
MOUNT_PRIOR_W = 3.0          # L2 spring (per mount-offset metre) pulling each tag to XML
MOUNT_OFFSET_WARN_MM = 12.0  # past this the prior can't be the whole story: a tag is
#                              physically mis-stuck/peeling -> re-check the hardware

# Outlier rejection. The per-pose median-over-frames capture knocks down pixel
# noise, but a single arm tag mis-detected at one pose (solvePnP corner fluke,
# motion blur, partial occlusion) survives as a point whose post-fit residual sits
# far above the few-mm bulk. Before the final solve we drop such points and re-fit,
# so one bad detection can't tug the bias or the camera. The residual is scored
# against the compliance-aware forward model (solve_bias_compliance), not bias-only:
# the unmodelled load-deflection roughly doubles the bias-only spread, and a genuine
# per-pose outlier can sit under the robust cutoff of that looser field while standing
# clear of the tighter compliance-aware one. Threshold is robust (median + K *
# scaled-MAD of the residual) with a floor so a tight clean fit isn't pruned. Only arm
# tags are rejected here; the table anchors' repeatability has its own solve gate.
#
# When a tag is flagged, both arm tags at that pose share one encoder qpos and one
# camera frame, so a pose-level fault (arm not settled, motion blur, camera bump)
# corrupts *both* even if only one crosses the cutoff. So we also drop any co-tag at
# the flagged pose whose residual clears a looser *companion* bar (COMPANION_K < MAD_K,
# same median/MAD) -- corroborated as pose-level by its flagged sibling. A co-tag near
# the clean bulk stays, so a lone-tag fluke (finger occlusion / peel, wrist fine) still
# keeps the good tag. Companion drops fire only at an already-flagged pose, never on
# clean data.
OUTLIER_MAD_K = 3.5          # drop arm points past median + K * (1.4826*MAD) of the residual
OUTLIER_FLOOR_MM = 6.0       # ...but never below this, so a clean fit isn't over-pruned
OUTLIER_COMPANION_K = 2.0    # at a flagged pose, also drop co-tags past median + this*(1.4826*MAD)
OUTLIER_MAX_ITERS = 5        # re-fit after each cut; a gross outlier inflates the MAD
OUTLIER_MAX_DROP_FRAC = 0.2  # past this the capture itself is suspect -> fail loud

# Cartesian sweep (base frame). The arm visits a deterministic grid of
# end-effector targets in front (+x) and just above (+z) the table at pan=middle,
# then extra blocks at panned angles. Per target, IK over the four non-base joints
# both reaches the point and aims both arm tags at the camera — that visibility
# coupling forces the wrist joints to track each target, so every joint is exercised
# and no two nearby poses share a configuration.
#
# The band hugs the table on purpose: the encoder bias this script measures is what
# warps the marker positions during a grasp, and that accuracy matters most right
# where the gripper meets objects on the table (a high-pose fit extrapolates poorly
# down to contact). The gripperframe sits at the fingertip, so a target z of a few
# cm puts the *markers* — what the camera measures — at ~0.06-0.10 m, the height a
# table grasp actually sees (vs ~0.09-0.18 m for the old high band).
POSE_MARGIN_RAD = 0.12     # keep IK targets off the joint hard limits
GRIPPER_FIXED = 0.5        # gripper parked: it moves neither tag
FREE_JOINTS = (1, 2, 3, 4)  # lift, elbow, wrist_flex, wrist_roll (pan & gripper fixed per block)
GRID_X = np.linspace(0.20, 0.36, 8)   # forward reach, m
# The *realized* gripperframe (the fingertip) is floored at MIN_EE_Z, not just the
# IK target: the wrist sweeps below pull the fingertip lower than its target, and the
# bias being measured (up to ~8 deg on the elbow -> a few cm at the EE) can drop the
# real fingertip below the nominal sim pose. The floor trades a little contact-regime
# coverage for not pressing the uncalibrated arm into the table; ncon==0 (sim
# collision-free) is enforced on top. The camera FOV culls tags out the top of frame;
# these low poses sit comfortably inside it.
MIN_EE_Z = 0.013          # hard floor on the realized gripperframe z, m (real-arm clearance)
GRID_Z = np.linspace(0.025, 0.08, 4)  # height, m  (8x4 = 32 targets at pan=middle)
GRID_Z_DITHER = 0.01       # checkerboard height jitter -> lift & elbow both move between neighbours
REACH_TOL = 0.025          # accept a pan=middle target only if the EE reaches within this, m
PAN_OFFSETS_DEG = (-35.0, -17.5, 17.5, 35.0)   # extra blocks; past ~35 deg a tag
# leaves the camera frame so both-visible IK no longer solves
OFF_X = np.linspace(0.22, 0.34, 6)
OFF_Z = (0.025, 0.06)      # low, matching the grid band so panned poses stay near the table
W_POS_GRID = 20.0          # IK position-vs-visibility weight at pan=middle (reach dominates)
W_POS_OFF = 6.0            # panned blocks: trade some reach to keep both tags visible
# Wrist decorrelation: nudge the wrist joints away from their visibility-optimal value
# (alternating between neighbours) so they sweep a range instead of sitting still. Both
# sweeps are pushed wide: roll because visibility barely constrains it, flex to recover
# the wrist_flex span the (now-removed) high wrist-up block used to provide, keeping the
# pitch bias observable from the low band alone.
W_ROLL_DECORR, ROLL_SWEEP_AMP = 0.4, 0.6   # weight, alternating offset (rad) about the seed
W_FLEX_DECORR, FLEX_SWEEP_AMP = 0.5, 1.1
IK_SEED = np.array([-0.6, 0.8, 0.6, 0.0])      # lift, elbow, wrist_flex, wrist_roll

# Drive-safety ordering weights (per JOINT_NAMES): EE-height-driving joints
# (lift/elbow/wrist_flex) heavy, pan/roll light — pan is a horizontal sweep that
# cannot lower the EE. See order_drive_safe / verify_drive_safe.
DRIVE_ORDER_WEIGHTS = np.array([0.2, 1.0, 1.0, 1.0, 0.3, 0.0])
INTERP_CHECK_STEPS = 24    # interpolation samples per transition (verify_drive_safe)

# Per-pose capture: median over several frames knocks down tvec pixel noise.
CAPTURE_FRAMES = 12


def _rotz(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def _ik_residual(x, pan, target, w_pos, roll_bias, flex_bias,
                 model, data, qposadr, ee_id, marker_sids, cam_pos):
    """Least-squares residual: reach `target` (weighted) while pointing both tag
    normals at the camera, with a weak wrist nudge for neighbour decorrelation."""
    q = np.zeros(6)
    q[0], q[5] = pan, GRIPPER_FIXED
    q[list(FREE_JOINTS)] = x
    data.qpos[qposadr] = q
    mujoco.mj_kinematics(model, data)
    r = list(w_pos * (data.site_xpos[ee_id] - target))
    for sid in marker_sids:
        normal = data.site_xmat[sid].reshape(3, 3)[:, 2]
        u = cam_pos - data.site_xpos[sid]
        r.append(1.0 - normal @ u / np.linalg.norm(u))   # 0 when the tag faces the camera
    r.append(W_FLEX_DECORR * (x[2] - flex_bias))
    r.append(W_ROLL_DECORR * (x[3] - roll_bias))
    return r


def order_drive_safe(model, data, qposadr, ee_id, poses):
    """Greedy nearest-neighbour ordering keyed on the EE-height-driving joints
    (DRIVE_ORDER_WEIGHTS: lift/elbow/wrist_flex heavy, pan/roll light), starting from
    the highest pose. drive_to ramps the arm along the straight joint-space line to
    the next pose, so keeping consecutive poses close in these joints keeps that line
    shallow — it never has to dive toward the table to get between two samples. Pure
    reordering: every pose is kept, the generation order is irrelevant to safety."""
    ee_z = []
    for q in poses:
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        ee_z.append(data.site_xpos[ee_id][2])
    order = [int(np.argmax(ee_z))]
    remaining = set(range(len(poses))) - set(order)
    while remaining:
        last = poses[order[-1]]
        order.append(min(remaining, key=lambda i:
                         float(np.sum((DRIVE_ORDER_WEIGHTS * (poses[i] - last)) ** 2))))
        remaining.discard(order[-1])
    return [poses[i] for i in order]


def verify_drive_safe(model, data, qposadr, ee_id, poses):
    """Assert the straight joint-space interpolation drive_to ramps along between
    consecutive poses stays collision-free in sim, failing loud on the first
    transition that would drive the arm into the table (caught in the dry-run before
    any hardware moves). Returns the worst gripperframe sag below its endpoints (m)."""
    worst_sag = 0.0
    for i in range(len(poses) - 1):
        a, b = poses[i], poses[i + 1]
        endpoint_lo = 1e9
        for q in (a, b):
            data.qpos[qposadr] = q
            mujoco.mj_forward(model, data)
            endpoint_lo = min(endpoint_lo, float(data.site_xpos[ee_id][2]))
        for u in np.linspace(0.0, 1.0, INTERP_CHECK_STEPS):
            data.qpos[qposadr] = (1.0 - u) * a + u * b
            mujoco.mj_forward(model, data)
            if data.ncon > 0:
                raise RuntimeError(
                    f"drive path {i}->{i + 1} collides in sim at u={u:.2f} "
                    f"(ncon={data.ncon}): the interpolation would drive the arm into "
                    "the table. Tighten the grid spacing or raise the height band.")
            worst_sag = max(worst_sag, endpoint_lo - float(data.site_xpos[ee_id][2]))
    return worst_sag


def generate_poses(model, data, jm):
    """Deterministic Cartesian sweep -> list of joint poses (see module docstring).

    Solves IK per target so both arm tags face the camera; keeps a pose only if both
    are visible and it is collision-free (and, at pan=middle, actually reaches the
    target). The kept poses are then re-ordered for drive safety (order_drive_safe)
    and the ordering is verified collision-free (verify_drive_safe). Returns
    (poses, n_grid) where n_grid is the pan=middle grid count (composition, not order:
    after the safety re-order the grid poses are interleaved with the panned ones)."""
    qposadr = jm.qposadr()
    b_lo = jm.xml_low()[list(FREE_JOINTS)] + POSE_MARGIN_RAD
    b_hi = jm.xml_high()[list(FREE_JOINTS)] - POSE_MARGIN_RAD
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_id >= 0, "site 'gripperframe' not found in model"
    marker_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                   for n in MARKER_SITE_NAMES]
    cam = tag_cam_model(model, data)

    def attempt(pan, target, seed, w_pos, parity):
        sign = 1.0 if parity else -1.0
        roll_bias = seed[3] + sign * ROLL_SWEEP_AMP
        flex_bias = seed[2] + sign * FLEX_SWEEP_AMP
        res = least_squares(
            _ik_residual, np.clip(seed, b_lo, b_hi), bounds=(b_lo, b_hi),
            args=(pan, target, w_pos, roll_bias, flex_bias, model, data,
                  qposadr, ee_id, marker_sids, cam.pos))
        q = np.zeros(6)
        q[0], q[5] = pan, GRIPPER_FIXED
        q[list(FREE_JOINTS)] = res.x
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        reached = float(np.linalg.norm(data.site_xpos[ee_id] - target))
        ok = (bool(markers_visible(data, marker_sids, cam).all())
              and data.ncon == 0
              and data.site_xpos[ee_id][2] >= MIN_EE_Z)
        return q, ok, reached

    poses, seed, k = [], IK_SEED.copy(), 0
    for j, z in enumerate(GRID_Z):
        for x in (GRID_X if j % 2 == 0 else GRID_X[::-1]):   # snake order -> smooth seeding
            zt = z + (GRID_Z_DITHER if k % 2 == 0 else -GRID_Z_DITHER)
            q, ok, reached = attempt(0.0, np.array([x, 0.0, zt]), seed, W_POS_GRID, k % 2)
            if ok and reached < REACH_TOL:
                poses.append(q)
                seed = q[list(FREE_JOINTS)]
            k += 1
    n_grid = len(poses)

    mid_seed = poses[len(poses) // 2][list(FREE_JOINTS)] if poses else IK_SEED.copy()
    for pan in np.radians(PAN_OFFSETS_DEG):
        seed, p = mid_seed.copy(), 0
        for i, x in enumerate(OFF_X):
            for z in (OFF_Z if i % 2 == 0 else OFF_Z[::-1]):   # snake z -> smooth seeding
                target = _rotz(pan) @ np.array([x, 0.0, z])
                q, ok, _ = attempt(pan, target, seed, W_POS_OFF, p % 2)
                if ok:
                    poses.append(q)
                    seed = q[list(FREE_JOINTS)]
                p += 1

    if len(poses) < MIN_SAMPLES:
        raise RuntimeError(
            f"sweep produced only {len(poses)} valid poses; need >= {MIN_SAMPLES}. "
            "Check the sim camera placement or loosen the grid/visibility bounds.")
    poses = order_drive_safe(model, data, qposadr, ee_id, poses)
    sag = verify_drive_safe(model, data, qposadr, ee_id, poses)
    print(f"  drive-safe order: worst interpolation sag {sag * 1000:.0f} mm below "
          "endpoints, all transitions collision-free")
    return poses, n_grid


def preview_poses(model, data, jm, poses, stream_port):
    """Cycle the generated poses in the sim (one per second), like show_starts.

    Native passive viewer by default; offscreen MJPEG stream when `stream_port` is
    set (the panel's sim view). Loops until the viewer closes or the process is
    interrupted."""
    qposadr = jm.qposadr()
    publisher = viewer = None
    if stream_port is not None:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(model, stream_port)
    else:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    try:
        i, t_pose = 0, 0.0
        while publisher is not None or viewer.is_running():
            if time.time() - t_pose >= 1.0:
                data.qpos[qposadr] = poses[i % len(poses)]
                mujoco.mj_forward(model, data)
                print(f"  pose {i % len(poses) + 1}/{len(poses)}: "
                      f"{poses[i % len(poses)].round(3).tolist()}")
                i += 1
                t_pose = time.time()
            if publisher is not None:
                publisher.publish(data)
            else:
                viewer.sync()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            publisher.close()
        if viewer is not None:
            viewer.close()


class _FrameStreamer:
    """Serve annotated BGR camera frames as MJPEG (panel run-page sim view slot)."""

    def __init__(self, port):
        from panel.streamer import FrameBox, JpegStreamer
        self._box = FrameBox()
        self._streamer = JpegStreamer(port, self._box)
        self._streamer.start()
        print(f"camera stream: http://0.0.0.0:{self._streamer.port}/stream")

    def publish(self, bgr):
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encoding of camera frame failed")
        self._box.publish(jpeg.tobytes())

    def close(self):
        self._streamer.close()


class MarkerCamera(threading.Thread):
    """Single owner of the webcam: a background thread that continuously reads,
    detects, and (if streaming) publishes the annotated frame, exposing the latest
    per-tag camera-frame poses. Capture pulls a median from the same stream, so the
    live view and the calibration data come from one reader (no device contention).
    """

    def __init__(self, family, stream_port):
        super().__init__(daemon=True)
        self.cap = open_camera(focus=FOCUS_ABSOLUTE, exposure=MARKER_EXPOSURE, gain=MARKER_GAIN)
        self.detector = make_detector(family)
        self.estimator = PoseEstimator(*load_intrinsics())
        self.wanted = set(ARM_TAG_TO_SITE) | set(TABLE_TAG_IDS)
        self._stream = _FrameStreamer(stream_port) if stream_port is not None else None
        self._lock = threading.Lock()
        self._latest = {}      # id -> (rvec, tvec) from the newest decoded frame
        self._seq = 0
        self._running = True
        self.error = None      # set if the read loop dies; the driver re-raises it

    def run(self):
        try:
            while self._running:
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("camera read failed mid-session")
                dets = self.detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                poses = {d.id: self.estimator.estimate(d) for d in dets if d.id in self.wanted}
                with self._lock:
                    self._latest = poses
                    self._seq += 1
                if self._stream is not None:
                    annotate_detections(frame, dets, self.estimator)
                    self._stream.publish(frame)
        except Exception as exc:   # surface to the main thread; a dead camera must fail loud
            self.error = exc

    def capture_median(self, n_frames, timeout=4.0):
        """Median tvec per tag over the next `n_frames` distinct frames (rejects
        single-frame flukes; needs a tag in >= half of them). Frames are post-settle
        because the reader has been draining the camera live during the arm's move."""
        tvecs = {t: [] for t in self.wanted}
        rvecs = {}
        last_seq, got, t0 = -1, 0, time.time()
        while got < n_frames and time.time() - t0 < timeout:
            with self._lock:
                seq, poses = self._seq, self._latest
            if seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq, got = seq, got + 1
            for t, (rvec, tvec) in poses.items():
                tvecs[t].append(tvec)
                rvecs[t] = rvec
        return {t: (rvecs[t], np.median(np.stack(v), axis=0))
                for t, v in tvecs.items() if len(v) >= n_frames // 2}

    def close(self):
        self._running = False
        self.join(timeout=2.0)
        self.cap.release()
        if self._stream is not None:
            self._stream.close()


def drive_to(bus, jm, direction, pose, prev_raw, max_raw_delta):
    """Ramp the arm from its current pose to `pose` and settle, then return the
    last raw target. Same clamped, sub-target-streamed shaping as the rollouts."""
    qnow = raw_to_rad(bus.read_all(), jm, direction)
    n_home = int(round(HOMING_S * SYSID_HZ))
    homing = smooth_ramp(qnow, pose, n_home)
    prev_raw = stream(bus, homing, jm, direction, prev_raw, True, max_raw_delta, None)
    n_settle = max(1, int(round(SETTLE_S * SYSID_HZ)))
    settle = np.tile(pose, (n_settle, 1))
    return stream(bus, settle, jm, direction, prev_raw, True, max_raw_delta, None)


def capture(args, jm, direction, poses, max_raw_delta):
    """Drive to each pose and capture (qpos, tags). Returns the samples list.

    The camera runs in a background thread (`MarkerCamera`) that also serves the
    annotated live stream when `--stream-port` is set."""
    cam = MarkerCamera(args.family, args.stream_port)
    cam.start()
    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()
    samples = []
    try:
        bus.set_position_kp(SERVO_POSITION_KP)
        bus.set_position_deadzone(SERVO_POSITION_DEADZONE)
        bus.set_torque_limit(SERVO_TORQUE_LIMIT)
        bus.enable_torque_all()
        prev_raw = bus.read_all().copy()
        for i, pose in enumerate(poses):
            prev_raw = drive_to(bus, jm, direction, pose, prev_raw, max_raw_delta)
            if cam.error is not None:
                raise cam.error
            tag_poses = cam.capture_median(CAPTURE_FRAMES)
            arm_seen = [t for t in ARM_TAG_TO_SITE if t in tag_poses]
            anchors_seen = [tag for tag in TABLE_TAG_IDS if tag in tag_poses]
            if len(anchors_seen) != len(TABLE_TAG_IDS) or not arm_seen:
                print(f"  pose {i + 1}/{len(poses)}: SKIP "
                      f"(table anchors={anchors_seen or 'none'}, "
                      f"arm={arm_seen or 'none'})")
                continue
            qpos = raw_to_rad(bus.read_all(), jm, direction)
            samples.append((qpos.copy(), tag_poses))
            print(f"  pose {i + 1}/{len(poses)}: captured arm tags {arm_seen} "
                  f"({len(samples)} kept)")
    finally:
        bus.close()  # torque off
        cam.close()
    return samples


def _backlash_dofs(model):
    """(qposadr, dofadr, range) of the passive gear-play joints (names end `_play`).

    Empty arrays if the model has none, so the solve degrades to rigid FK."""
    jids = [j for j in range(model.njnt)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_play")]
    return (model.jnt_qposadr[jids].copy(),
            model.jnt_dofadr[jids].copy(),
            model.jnt_range[jids].copy())


def _apply_gravity_slack(model, data, play):
    """Pose the gear-play joints where gravity loads them, given the motor joints are
    already set in data.qpos. Quasi-static: the gap is tiny (±0.5°), so the link
    rests against whichever side the gravitational generalized force pushes it to —
    only the sign is pose-dependent (matches a dynamic settle, see tests). Leaves
    data's kinematics updated for the settled pose. No-op if the model has no play
    joints. Assumes qvel == 0 so qfrc_bias is the pure gravity term."""
    qadr, dadr, rng = play
    if len(qadr) == 0:
        mujoco.mj_kinematics(model, data)
        return
    data.qpos[qadr] = 0.0
    mujoco.mj_forward(model, data)               # qfrc_bias = gravity force at slack=0
    g = data.qfrc_bias[dadr]
    data.qpos[qadr] = np.where(-g >= 0.0, rng[:, 1], rng[:, 0])
    mujoco.mj_kinematics(model, data)            # re-place the sites with slack applied


def settled_site_xpos(model, data, qposadr, qpos, site_id):
    """Base-frame position of `site_id` at motor pose `qpos`, gear-play settled under
    gravity. Single source of the settled-FK forward model (the bias solve inverts
    exactly this; the synthetic-data tests generate against it)."""
    data.qpos[qposadr] = qpos
    _apply_gravity_slack(model, data, _backlash_dofs(model))
    return data.site_xpos[site_id].copy()


def paired_points(samples, model, data, qposadr, site_ids):
    """Like calib_solve.paired_points, but the FK settles the gear-play joints under
    gravity per pose (the encoder pins the motor; the link sags within the backlash
    gap toward the camera-measured position). Pairs each detected arm tag's camera
    tvec (src) with its settled base-frame FK position (dst)."""
    play = _backlash_dofs(model)
    src, dst, tags = [], [], []
    for qpos, poses in samples:
        data.qpos[qposadr] = qpos
        _apply_gravity_slack(model, data, play)
        for tag, sid in site_ids.items():
            if tag not in poses:
                continue
            src.append(poses[tag][1])               # tvec: tag centre in camera frame
            dst.append(data.site_xpos[sid].copy())  # settled tag centre in base frame
            tags.append(tag)
    return np.array(src), np.array(dst), np.array(tags)


def _true_poses(samples, model, data, qposadr, b_full, compliance):
    """Map each sample's encoder qpos to the true motor pose -- subtract the constant
    bias, then the load-dependent gravity deflection (real/calib/compliance.py) -- leaving the
    tag dict untouched. The settled FK in paired_points then applies gear play on top."""
    out = []
    for qpos, poses in samples:
        q = qpos - b_full
        out.append((q - gravity_deflection(model, data, qposadr, q, compliance), poses))
    return out


def solve_camera(samples, model, data, qposadr, site_ids):
    """T_base_cam (Umeyama) on the settled-FK tag centres; see paired_points. Position-
    only, so immune to the glue rotation and solvePnP rvec flips on the small tags.
    Returns (T_base_cam, rms_mm, tags, err_mm) with per-point residuals."""
    src, dst, tags = paired_points(samples, model, data, qposadr, site_ids)
    T_base_cam, rms = rigid_register(src, dst)
    err_mm = np.linalg.norm(dst - (src @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]), axis=1) * 1000.0
    return T_base_cam, rms * 1000.0, tags, err_mm


def bias_residuals(b4, samples, model, data, qposadr, site_ids):
    """Position residuals of FK(qpos - b) vs the optimally-aligned camera points.

    For a candidate bias the best camera is the closed-form Umeyama fit, so the
    only free parameters are the 4 observable joint biases (separable least
    squares). Returns the flattened per-point residual the optimiser drives to 0."""
    b_full = np.zeros(6)
    b_full[list(OBSERVABLE_JOINTS)] = b4
    corrected = [(qpos - b_full, poses) for qpos, poses in samples]
    src, dst, _ = paired_points(corrected, model, data, qposadr, site_ids)
    T, _ = rigid_register(src, dst)
    return (dst - (src @ T[:3, :3].T + T[:3, 3])).ravel()


def solve_bias(samples, model, data, qposadr, site_ids):
    res = least_squares(bias_residuals, np.zeros(len(OBSERVABLE_JOINTS)),
                        args=(samples, model, data, qposadr, site_ids))
    b_full = np.zeros(6)
    b_full[list(OBSERVABLE_JOINTS)] = res.x
    return b_full


def _unpack_calibration(params, n_sites):
    """Split the solver's flat parameter vector into (b_full, offsets, compliance):
    4 observable-joint biases, then 3 centre-offset components per tag, then one
    gravity-compliance coefficient per COMP_JOINTS joint. offsets is (n_sites, 3);
    b_full and compliance are 6-vectors padded with 0 on the pinned/rigid joints."""
    n_b, n_off = len(OBSERVABLE_JOINTS), 3 * n_sites
    b_full = np.zeros(6)
    b_full[list(OBSERVABLE_JOINTS)] = params[:n_b]
    offsets = params[n_b:n_b + n_off].reshape(n_sites, 3)
    compliance = np.zeros(6)
    compliance[list(COMP_JOINTS)] = params[n_b + n_off:]
    return b_full, offsets, compliance


def bias_compliance_residuals(params, samples, model, data, qposadr, site_ids):
    """Position residuals of the settled FK of the bias+compliance-corrected pose vs
    the optimally-aligned camera points (`params` = 4 observable biases then the
    COMP_JOINTS compliance coefficients; layout is _unpack_calibration with 0 sites).
    Mounts are held at their current XML values -- they shift a tag <0.5 mm, so they
    don't change which point is an outlier, and leaving them fixed keeps this solve
    cheap for the reject loop. The camera stays the closed-form Umeyama inner fit."""
    b_full, _, compliance = _unpack_calibration(params, 0)
    corrected = _true_poses(samples, model, data, qposadr, b_full, compliance)
    src, dst, _ = paired_points(corrected, model, data, qposadr, site_ids)
    T, _ = rigid_register(src, dst)
    return (dst - (src @ T[:3, :3].T + T[:3, 3])).ravel()


def solve_bias_compliance(samples, model, data, qposadr, site_ids):
    """4 observable-joint biases + per-COMP_JOINTS gravity compliance (mounts fixed),
    the forward model outlier rejection thresholds against. Solving compliance here too
    matters: the bias-only field is ~2x looser (its unmodelled load-deflection inflates
    the residual spread), so a real per-pose outlier hides under the robust cutoff that
    the tighter compliance-aware field flags. Returns (b_full, compliance) as 6-vectors
    padded with 0 on the pinned/rigid joints."""
    n = len(OBSERVABLE_JOINTS) + len(COMP_JOINTS)
    res = least_squares(bias_compliance_residuals, np.zeros(n),
                        args=(samples, model, data, qposadr, site_ids))
    b_full, _, compliance = _unpack_calibration(res.x, 0)
    return b_full, compliance


def mount_bias_residuals(params, samples, model, data, qposadr, site_ids, nominal):
    """Residual for the joint bias + per-tag mount-offset + gravity-compliance solve
    (`params` layout: see _unpack_calibration). Each offset shifts its tag's site to
    nominal+offset in the parent-body frame and the compliance bends the motor pose
    (real/calib/compliance.py) before the settled FK, so dst is the fully-corrected base-frame
    tag centre; the camera stays the closed-form Umeyama inner fit. The trailing
    MOUNT_PRIOR_W terms are the XML prior, pulling every offset toward 0 so a
    near-degenerate direction can't trade against a joint bias. Compliance carries no
    prior -- it is well-observed (its per-pose torque signature is distinct from a
    constant bias) and cross-validates."""
    b_full, offsets, compliance = _unpack_calibration(params, len(site_ids))
    for nom, off, sid in zip(nominal, offsets, site_ids.values()):
        model.site_pos[sid] = nom + off
    corrected = _true_poses(samples, model, data, qposadr, b_full, compliance)
    src, dst, _ = paired_points(corrected, model, data, qposadr, site_ids)
    T, _ = rigid_register(src, dst)
    fit = (dst - (src @ T[:3, :3].T + T[:3, 3])).ravel()
    return np.concatenate([fit, MOUNT_PRIOR_W * offsets.ravel()])


def solve_calibration(samples, model, data, qposadr, site_ids):
    """Solve the 4 observable joint biases jointly with each arm tag's 3D mount-centre
    offset (XML-prior-regularised; see MOUNT_PRIOR_W) and the per-joint gravity
    compliance (COMP_JOINTS; see real/calib/compliance.py). Leaves model.site_pos at the
    solved mounts so the caller's camera/table fit uses the same forward model, and
    returns (b_full, offsets, compliance) with offsets {tag: 3-vector in the body
    frame, m} and compliance a 6-vector (rad per N.m, 0 on the rigid joints)."""
    nominal = np.array([model.site_pos[sid].copy() for sid in site_ids.values()])
    n = len(OBSERVABLE_JOINTS) + 3 * len(site_ids) + len(COMP_JOINTS)
    res = least_squares(mount_bias_residuals, np.zeros(n),
                        args=(samples, model, data, qposadr, site_ids, nominal))
    b_full, offsets, compliance = _unpack_calibration(res.x, len(site_ids))
    for nom, off, sid in zip(nominal, offsets, site_ids.values()):
        model.site_pos[sid] = nom + off       # leave the solved mounts applied
    return b_full, dict(zip(site_ids, offsets)), compliance


def _arm_point_keys(samples, site_ids):
    """(sample_idx, tag) per arm-tag point, in the exact order paired_points emits
    them (sample-major, then site_ids order), so an err_mm index maps back to a
    sample and tag."""
    keys = []
    for i, (_, poses) in enumerate(samples):
        for tag in site_ids:
            if tag in poses:
                keys.append((i, tag))
    return keys


def reject_outliers(samples, model, data, qposadr, site_ids):
    """Drop arm-tag observations whose post-fit residual is a robust outlier,
    re-solving the bias, compliance & camera after each cut (see solve_bias_compliance
    on why the reject model must carry compliance). Returns (kept_samples, dropped),
    where dropped is a list of (sample_idx, tag, err_mm) on the original indices.

    A flagged tag also condemns any co-tag at the same pose that clears the looser
    companion bar (OUTLIER_COMPANION_K), since both tags share the pose's qpos/frame;
    a clean co-tag is kept, so a lone-tag fluke doesn't lose the good tag. The
    table anchors are never touched (their solve gates them separately). Fails loud if
    more than OUTLIER_MAX_DROP_FRAC of the arm points would be discarded — that
    points at a bad capture (camera bumped mid-run, wrong calibration json), not a
    handful of flukes."""
    samples = [(qpos, dict(poses)) for qpos, poses in samples]   # copy dicts; we mutate
    n_points0 = len(_arm_point_keys(samples, site_ids))
    dropped = []
    for _ in range(OUTLIER_MAX_ITERS):
        b_full, compliance = solve_bias_compliance(samples, model, data, qposadr, site_ids)
        corrected = _true_poses(samples, model, data, qposadr, b_full, compliance)
        _, _, _, err_mm = solve_camera(corrected, model, data, qposadr, site_ids)
        keys = _arm_point_keys(samples, site_ids)
        assert len(keys) == len(err_mm)
        med = np.median(err_mm)
        mad = np.median(np.abs(err_mm - med))
        cutoff = max(OUTLIER_FLOOR_MM, med + OUTLIER_MAD_K * 1.4826 * mad)
        worst = int(np.argmax(err_mm))
        if err_mm[worst] <= cutoff:
            break
        # Drop the worst point's whole pose-level fault, then re-fit: the flagged tag
        # plus any co-tag at that pose past the looser companion bar (both share the
        # pose's qpos/frame). Only one pose per iteration — a gross outlier contaminates
        # the bias, inflating otherwise-clean residuals elsewhere, so re-fitting between
        # poses keeps the cut from taking collateral on unrelated points.
        i = keys[worst][0]
        companion = med + OUTLIER_COMPANION_K * 1.4826 * mad
        condemned = sorted(
            (keys[j][1], float(err_mm[j])) for j in range(len(keys))
            if keys[j][0] == i and err_mm[j] > companion)   # includes worst (> cutoff > companion)
        if len(dropped) + len(condemned) > OUTLIER_MAX_DROP_FRAC * n_points0:
            raise RuntimeError(
                f"outlier rejection wants to drop more than "
                f"{OUTLIER_MAX_DROP_FRAC:.0%} of {n_points0} arm points; the capture "
                "is suspect (camera bumped mid-run, wrong calibration json, or a tag "
                "glued askew).")
        for tag, e in condemned:
            del samples[i][1][tag]
            dropped.append((i, tag, e))
    return samples, dropped


def write_marker_sites(xml_path, site_positions):
    """Persist solved marker-site centres into the arm XML (textual pos= edit).

    The sites feed training's marker observations, so the XML is the single source
    of truth the physical tags must match. Fails loud if a site element or its pos
    attribute isn't found exactly once — never a silent partial write."""
    text = xml_path.read_text()
    for name, pos in site_positions.items():
        needle = f'<site name="{name}"'
        start = text.find(needle)
        assert start != -1, f"{xml_path}: site {name!r} not found"
        assert text.find(needle, start + 1) == -1, f"{xml_path}: site {name!r} not unique"
        end = text.index("/>", start)
        element = text[start:end]
        m = re.search(r'\bpos="[^"]*"', element)
        assert m is not None, f"{xml_path}: site {name!r} has no pos attribute"
        new_pos = " ".join(f"{v:.6g}" for v in pos)
        text = (f'{text[:start]}{element[:m.start()]}pos="{new_pos}"'
                f"{element[m.end():]}{text[end:]}")
    xml_path.write_text(text)


def report_and_save(args, samples, model, data, jm, site_ids):
    qposadr = jm.qposadr()
    samples, dropped = reject_outliers(samples, model, data, qposadr, site_ids)
    if dropped:
        print(f"\ndiscarded {len(dropped)} outlier arm-tag detection(s) "
              "(robust residual cutoff):")
        for i, tag, e in sorted(dropped):
            print(f"    sample {i + 1} tag {tag} ({ARM_TAG_TO_SITE[tag]}): "
                  f"residual {e:.1f} mm")
    _, rms_before, _, _ = solve_camera(samples, model, data, qposadr, site_ids)
    b_full, mount_offsets, compliance = solve_calibration(samples, model, data, qposadr, site_ids)
    corrected = _true_poses(samples, model, data, qposadr, b_full, compliance)

    # solve_calibration left model.site_pos at the solved mounts, so the camera and
    # table fits below use the same mount-corrected forward model.
    T_base_cam, rms_after, tags, err_mm = solve_camera(corrected, model, data, qposadr, site_ids)
    table_anchors, t_mm, t_deg = solve_table_anchors(corrected, T_base_cam)
    quarter_turns, _ = determine_quarter_turns(
        corrected, model, data, qposadr, site_ids, sim_cam_R_opencv(model, data))

    print("\nencoder bias solve (position-only, pan & gripper pinned to 0):")
    print(f"  arm-tag RMS {rms_before:.1f} mm -> {rms_after:.1f} mm over {len(tags)} points")
    for i, name in enumerate(JOINT_NAMES):
        pinned = " (pinned)" if i not in OBSERVABLE_JOINTS else ""
        comp = (f"  compliance {np.degrees(compliance[i]):+7.2f} deg/N.m"
                if i in COMP_JOINTS else "")
        print(f"    {name:13s} bias {np.degrees(b_full[i]):+6.2f} deg{pinned}{comp}")
    for tag in site_ids:
        e = err_mm[tags == tag]
        print(f"    tag {tag} ({ARM_TAG_TO_SITE[tag]}): mean {e.mean():.1f} mm "
              f"(max {e.max():.1f}) over {len(e)} points")
    print("  freed tag mounts (offset from previous XML, written back to the sites):")
    for tag in site_ids:
        d = mount_offsets[tag] * 1000.0
        warn = "  <-- mis-stuck/peeling? re-tape" if np.linalg.norm(d) > MOUNT_OFFSET_WARN_MM else ""
        print(f"    tag {tag} ({ARM_TAG_TO_SITE[tag]}) centre offset "
              f"[{d[0]:+.1f} {d[1]:+.1f} {d[2]:+.1f}] mm  |d| {np.linalg.norm(d):.1f}{warn}")
    for tag in TABLE_TAG_IDS:
        print(f"  table anchor {tag} in base: "
              f"{table_anchors[tag][:3, 3].round(4).tolist()} m")
    print(f"  pooled anchor detection repeatability {np.median(t_mm):.1f} mm / "
          f"{np.median(t_deg):.2f} deg")
    print("  target a few mm. Worse => a tag glued askew beyond the prior, or bad capture.")

    write_marker_sites(ARM_XML, {ARM_TAG_TO_SITE[tag]: model.site_pos[sid].copy()
                                 for tag, sid in site_ids.items()})
    # The model was loaded from args.xml (which includes ARM_XML); reload and confirm
    # the written mounts actually reach it, so a mismatched --xml can't desync sim FK
    # from the calibration that was just saved.
    check = mujoco.MjModel.from_xml_path(args.xml)
    for tag, sid in site_ids.items():
        name = ARM_TAG_TO_SITE[tag]
        sid_check = mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid_check >= 0 and np.allclose(
            check.site_pos[sid_check], model.site_pos[sid], atol=1e-6), (
            f"site {name!r}: {args.xml} did not pick up the mount written to {ARM_XML} "
            "(is --xml a model that doesn't include it?)")

    save_extrinsics(args.out, table_anchors, T_base_cam, FOCUS_ABSOLUTE,
                    len(samples), rms_after, float(np.median(t_deg)), quarter_turns)
    save_calibration(args.cal_out, b_full, compliance, len(samples), rms_before, rms_after)
    print(f"\nwrote {ARM_XML} (marker site mounts)\nwrote {args.out}\nwrote {args.cal_out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    p.add_argument("--family", default="apriltag", choices=["apriltag", "aruco"])
    p.add_argument("--execute", action="store_true",
                   help="Drive the arm and capture. Default: dry-run (preview the sweep).")
    p.add_argument("--stream-port", type=int, default=None,
                   help="Serve the dry-run preview as MJPEG on this port (panel sim view).")
    p.add_argument("--out", default=EXTRINSICS_PATH)
    p.add_argument("--cal-out", default=CALIBRATION_PATH)
    p.add_argument("--samples-out", default=str(SAMPLES_PATH))
    p.add_argument("--from-samples", default=None,
                   help="Skip capture; load samples from this JSON and just solve.")
    return p.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    direction = np.ones(6, dtype=np.int8)  # follower: no inversions (verified via twin)
    site_ids = {}
    for tag_id, site_name in ARM_TAG_TO_SITE.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert sid >= 0, f"site {site_name!r} not found in model"
        site_ids[tag_id] = sid

    if args.from_samples:
        samples = load_samples(args.from_samples)
        print(f"loaded {len(samples)} samples from {args.from_samples}")
    else:
        poses, n_grid = generate_poses(model, data, jm)
        print(f"generated {len(poses)} sweep poses "
              f"({n_grid} pan=middle grid, {len(poses) - n_grid} panned), table-hugging")
        if not args.execute:
            print("dry-run: previewing the sweep in sim "
                  "(pass --execute to drive the arm and capture).")
            preview_poses(model, data, jm, poses, args.stream_port)
            return
        max_raw_delta = max_raw_delta_per_step(
            float(OmegaConf.load(CONFIG_YAML)["action_scale"]))
        samples = capture(args, jm, direction, poses, max_raw_delta)
        save_samples(args.samples_out, samples)
        print(f"saved {len(samples)} samples to {args.samples_out}")

    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(f"only {len(samples)} samples; need >= {MIN_SAMPLES}")
    report_and_save(args, samples, model, data, jm, site_ids)


if __name__ == "__main__":
    main()
