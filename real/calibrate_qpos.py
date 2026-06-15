"""Self-driven encoder-bias + camera-extrinsics calibration from the arm tags.

The table-marker calibrator (`real/calibrate_table_marker.py`) anchors the fixed
table tag by *trusting the encoders*: it registers `T_base_cam` from the arm-tag
FK positions. That surfaced ~11 mm of arm-tag residual dominated by uncalibrated
encoder zero-offsets (commit 888c3bc). This script closes that loop: it solves the
per-joint encoder bias `b` *jointly* with the camera pose so FK(theta_enc - b)
lands the arm tags where the camera says they are, then re-derives the table-tag
anchor with the corrected kinematics. One pass writes both `extrinsics.yaml`
(`t_base_cam_fixed`, `t_base_table`, `quarter_turns`) and `calibration.yaml`
(`qpos_bias`).

Differences from the hand-posed table calibrator:
  - The arm drives *itself* through a deterministic Cartesian sweep: a low grid of
    end-effector targets in front of and above the base at pan=middle, blocks at
    panned angles (<=35 deg — past that a tag leaves the camera frame), and a
    wrist-up block of low targets with the gripper pitched skyward. Per target, IK
    over the four non-base joints both reaches the point and aims *both* arm tags at
    the camera; that coupling moves every joint between nearby poses, so all
    observable biases are excited. Each pose is verified collision-free and
    both-tags-visible in sim (including the marker_visible height ceiling that
    models the real camera's top-of-frame cut-off) before driving.
  - It solves `qpos_bias`, not just the extrinsics.
  - Before the final solve it rejects outlier arm-tag detections: a single tag
    mis-detected at one pose (solvePnP fluke / motion blur) is dropped on a robust
    residual cutoff and the bias+camera re-fit, so one bad point can't skew either.

The dry-run (no --execute) previews the sweep in the sim viewer (one pose/second),
so the trajectory can be eyeballed before any hardware moves. With --stream-port
(set automatically by the panel), the dry-run streams that sim preview and the
--execute run streams the live annotated camera feed — same MJPEG endpoint, so the
panel's sim-view slot shows whichever is running.

Position-only throughout (tag centres / tvec), so it inherits the table
calibrator's immunity to solvePnP rvec flips on the small arm tags.

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

Run:
    conda run -n mujoco_env python -m real.calibrate_qpos --execute
    conda run -n mujoco_env python -m real.calibrate_qpos              # dry-run: preview poses
    conda run -n mujoco_env python -m real.calibrate_qpos --from-samples <json>
"""
import argparse
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import least_squares

from real.calibrate_camera import FOCUS_ABSOLUTE
from real.calibrate_table_marker import (
    determine_quarter_turns,
    load_samples,
    save_samples,
    sim_cam_R_opencv,
    solve_table,
)
from real.calibration import CALIBRATION_PATH, save_calibration
from real.camera import open_camera
from real.detect import make_detector
from real.extrinsics import EXTRINSICS_PATH, rigid_register, save_extrinsics
from real.marker_spec import ARM_TAG_TO_SITE, MARKER_EXPOSURE, MARKER_GAIN, TABLE_TAG_ID
from real.overlay import annotate_detections
from real.pose import PoseEstimator, load_intrinsics
from real.twin.constants import SERVO_POSITION_DEADZONE, SERVO_POSITION_KP
from real.twin.mapping import JOINT_NAMES, load_joint_maps, raw_to_rad
from real.twin.servo_io import ServoBus
from src.base_env import MARKER_SITE_NAMES, markers_visible, tag_cam_world_pos
from src.units import max_raw_delta_per_step
from sysid.record_real import CONFIG_YAML, HOMING_S, SETTLE_S, smooth_ramp, stream
from sysid.trajectories import SYSID_HZ

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
DEFAULT_CAL = REPO_ROOT / "real" / "follower_calibration.json"
SAMPLES_PATH = REPO_ROOT / "real" / "qpos_calib_samples.json"
MIN_SAMPLES = 8

# qpos_bias indices solved jointly with the camera; pan (0) and gripper (5) are
# pinned to 0 (see module docstring on observability).
OBSERVABLE_JOINTS = (1, 2, 3, 4)

# Outlier rejection. The per-pose median-over-frames capture knocks down pixel
# noise, but a single arm tag mis-detected at one pose (solvePnP corner fluke,
# motion blur, partial occlusion) survives as a point whose post-fit residual sits
# far above the few-mm bulk. Before the final solve we drop such points and re-fit,
# so one bad detection can't tug the bias or the camera. Threshold is robust
# (median + K * scaled-MAD of the residual) with a floor so a tight clean fit isn't
# pruned. Only arm tags are rejected here; the table tag's repeatability is its own
# gate in solve_table.
OUTLIER_MAD_K = 3.5          # drop arm points past median + K * (1.4826*MAD) of the residual
OUTLIER_FLOOR_MM = 6.0       # ...but never below this, so a clean fit isn't over-pruned
OUTLIER_MAX_ITERS = 5        # re-fit after each cut; a gross outlier inflates the MAD
OUTLIER_MAX_DROP_FRAC = 0.2  # past this the capture itself is suspect -> fail loud

# Cartesian sweep (base frame). The arm visits a deterministic grid of
# end-effector targets in front (+x) and above (+z) the base at pan=middle, then
# extra blocks at panned angles. Per target, IK over the four non-base joints both
# reaches the point and aims both arm tags at the camera — that visibility coupling
# forces the wrist joints to track each target, so every joint is exercised and
# no two nearby poses share a configuration.
POSE_MARGIN_RAD = 0.12     # keep IK targets off the joint hard limits
GRIPPER_FIXED = 0.5        # gripper parked: it moves neither tag
FREE_JOINTS = (1, 2, 3, 4)  # lift, elbow, wrist_flex, wrist_roll (pan & gripper fixed per block)
GRID_X = np.linspace(0.20, 0.37, 8)   # forward reach, m (pushed further out front)
# Height band sits low in the camera frame: the marker_visible() height ceiling
# (0.23 m) already culls anything that leaves the top of frame, so the band is
# tuned down to where most poses land well inside it. Bottom floored at 0.04 to
# keep ~2.5 cm fingertip-to-table clearance on the still-uncalibrated arm, where
# the encoder bias we are measuring can put the real EE a couple cm off from sim.
GRID_Z = np.linspace(0.04, 0.13, 4)   # height, m  (8x4 = 32 targets at pan=middle)
GRID_Z_DITHER = 0.012      # checkerboard height jitter -> lift & elbow both move between neighbours
REACH_TOL = 0.02           # accept a pan=middle target only if the EE reaches within this, m
PAN_OFFSETS_DEG = (-35.0, -17.5, 17.5, 35.0)   # extra blocks; past ~35 deg a tag
# leaves the 0.23 m frame ceiling so both-visible IK no longer solves
OFF_X = np.linspace(0.22, 0.35, 6)
OFF_Z = (0.05, 0.11)       # low, below the grid top so panned poses stay in the camera frame
W_POS_GRID = 20.0          # IK position-vs-visibility weight at pan=middle (reach dominates)
W_POS_OFF = 6.0            # panned blocks: trade some reach to keep both tags visible
# Wrist decorrelation: nudge the wrist joints away from their visibility-optimal value
# (alternating between neighbours) so they sweep a range instead of sitting still. Both
# sweeps are pushed wide: roll because visibility barely constrains it, flex so the
# grid includes low poses with the wrist pitched up (not just aimed flat at the camera).
W_ROLL_DECORR, ROLL_SWEEP_AMP = 0.4, 0.6   # weight, alternating offset (rad) about the seed
W_FLEX_DECORR, FLEX_SWEEP_AMP = 0.25, 0.5
# Wrist-up block: a handful of low forward targets solved with the gripper pitched
# to point skyward (W_UP pulls the gripper's approach axis toward world-up). Gives
# the sweep low poses with the wrist pointing up — wrist_flex decorrelated from
# height, the configuration position-only IK otherwise never visits.
WRIST_UP_X = np.linspace(0.20, 0.30, 4)
WRIST_UP_Z = (0.06, 0.12)
W_POS_UP = 6.0             # reach weight (relaxed: pointing up trades some reach)
W_UP = 3.0                # pull the gripper approach axis toward world-up
IK_SEED = np.array([-0.6, 0.8, 0.6, 0.0])      # lift, elbow, wrist_flex, wrist_roll

# Per-pose capture: median over several frames knocks down tvec pixel noise.
CAPTURE_FRAMES = 12


def _rotz(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def _ik_residual(x, pan, target, w_pos, roll_bias, flex_bias, up_weight,
                 model, data, qposadr, ee_id, marker_sids, cam_pos):
    """Least-squares residual: reach `target` (weighted) while pointing both tag
    normals at the camera, with a weak wrist nudge for neighbour decorrelation.
    `up_weight` > 0 additionally pitches the gripper's approach axis skyward (the
    wrist-up block), decorrelating wrist_flex from height."""
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
    if up_weight:
        grip_x_up = data.site_xmat[ee_id].reshape(3, 3)[2, 0]  # gripper x-axis, world z
        r.append(up_weight * (1.0 - grip_x_up))   # 0 when the gripper points straight up
    return r


def generate_poses(model, data, jm):
    """Deterministic Cartesian sweep -> list of joint poses (see module docstring).

    Solves IK per target so both arm tags face the camera; keeps a pose only if
    both are visible and it is collision-free (and, at pan=middle, actually reaches
    the target). Returns (poses, n_grid) where n_grid is the pan=middle count."""
    qposadr = jm.qposadr()
    b_lo = jm.xml_low()[list(FREE_JOINTS)] + POSE_MARGIN_RAD
    b_hi = jm.xml_high()[list(FREE_JOINTS)] - POSE_MARGIN_RAD
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_id >= 0, "site 'gripperframe' not found in model"
    marker_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                   for n in MARKER_SITE_NAMES]
    cam_pos = tag_cam_world_pos(model, data)

    def attempt(pan, target, seed, w_pos, parity, up_weight=0.0):
        sign = 1.0 if parity else -1.0
        roll_bias = seed[3] + sign * ROLL_SWEEP_AMP
        flex_bias = seed[2] + sign * FLEX_SWEEP_AMP
        res = least_squares(
            _ik_residual, np.clip(seed, b_lo, b_hi), bounds=(b_lo, b_hi),
            args=(pan, target, w_pos, roll_bias, flex_bias, up_weight, model, data,
                  qposadr, ee_id, marker_sids, cam_pos))
        q = np.zeros(6)
        q[0], q[5] = pan, GRIPPER_FIXED
        q[list(FREE_JOINTS)] = res.x
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        reached = float(np.linalg.norm(data.site_xpos[ee_id] - target))
        ok = bool(markers_visible(data, marker_sids, cam_pos).all()) and data.ncon == 0
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
        for x in OFF_X:
            for z in OFF_Z:
                target = _rotz(pan) @ np.array([x, 0.0, z])
                q, ok, _ = attempt(pan, target, seed, W_POS_OFF, p % 2)
                if ok:
                    poses.append(q)
                    seed = q[list(FREE_JOINTS)]
                p += 1

    seed = IK_SEED.copy()
    for i, x in enumerate(WRIST_UP_X):
        for z in WRIST_UP_Z:
            q, ok, _ = attempt(0.0, np.array([x, 0.0, z]), seed, W_POS_UP, i % 2,
                               up_weight=W_UP)
            if ok:
                poses.append(q)
                seed = q[list(FREE_JOINTS)]
    if len(poses) < MIN_SAMPLES:
        raise RuntimeError(
            f"sweep produced only {len(poses)} valid poses; need >= {MIN_SAMPLES}. "
            "Check the sim camera placement or loosen the grid/visibility bounds.")
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
        self.wanted = set(ARM_TAG_TO_SITE) | {TABLE_TAG_ID}
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
        bus.enable_torque_all()
        prev_raw = bus.read_all().copy()
        for i, pose in enumerate(poses):
            prev_raw = drive_to(bus, jm, direction, pose, prev_raw, max_raw_delta)
            if cam.error is not None:
                raise cam.error
            tag_poses = cam.capture_median(CAPTURE_FRAMES)
            arm_seen = [t for t in ARM_TAG_TO_SITE if t in tag_poses]
            if TABLE_TAG_ID not in tag_poses or not arm_seen:
                print(f"  pose {i + 1}/{len(poses)}: SKIP "
                      f"(table={'ok' if TABLE_TAG_ID in tag_poses else 'MISSING'}, "
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
    """Like calibrate_table_marker.paired_points, but the FK settles the gear-play
    joints under gravity per pose (the encoder pins the motor; the link sags within
    the backlash gap toward the camera-measured position). Pairs each detected arm
    tag's camera tvec (src) with its settled base-frame FK position (dst)."""
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
    re-solving the bias+camera after each cut. Returns (kept_samples, dropped),
    where dropped is a list of (sample_idx, tag, err_mm) on the original indices.

    The table tag is never touched (solve_table needs it and gates it separately);
    a sample that loses an arm tag is kept for its remaining tags. Fails loud if
    more than OUTLIER_MAX_DROP_FRAC of the arm points would be discarded — that
    points at a bad capture (camera bumped mid-run, wrong calibration json), not a
    handful of flukes."""
    samples = [(qpos, dict(poses)) for qpos, poses in samples]   # copy dicts; we mutate
    n_points0 = len(_arm_point_keys(samples, site_ids))
    dropped = []
    for _ in range(OUTLIER_MAX_ITERS):
        b_full = solve_bias(samples, model, data, qposadr, site_ids)
        corrected = [(qpos - b_full, poses) for qpos, poses in samples]
        _, _, _, err_mm = solve_camera(corrected, model, data, qposadr, site_ids)
        keys = _arm_point_keys(samples, site_ids)
        assert len(keys) == len(err_mm)
        med = np.median(err_mm)
        mad = np.median(np.abs(err_mm - med))
        cutoff = max(OUTLIER_FLOOR_MM, med + OUTLIER_MAD_K * 1.4826 * mad)
        worst = int(np.argmax(err_mm))
        if err_mm[worst] <= cutoff:
            break
        # Drop only the single worst point, then re-fit: a gross outlier contaminates
        # the bias, inflating otherwise-clean residuals, so cutting all-at-once would
        # take collateral. One-at-a-time over the iterations isolates the real ones.
        if len(dropped) + 1 > OUTLIER_MAX_DROP_FRAC * n_points0:
            raise RuntimeError(
                f"outlier rejection wants to drop more than "
                f"{OUTLIER_MAX_DROP_FRAC:.0%} of {n_points0} arm points; the capture "
                "is suspect (camera bumped mid-run, wrong calibration json, or a tag "
                "glued askew).")
        i, tag = keys[worst]
        del samples[i][1][tag]
        dropped.append((i, tag, float(err_mm[worst])))
    return samples, dropped


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
    b_full = solve_bias(samples, model, data, qposadr, site_ids)
    corrected = [(qpos - b_full, poses) for qpos, poses in samples]

    T_base_cam, rms_after, tags, err_mm = solve_camera(corrected, model, data, qposadr, site_ids)
    T_base_table, t_mm, t_deg = solve_table(corrected, T_base_cam)
    quarter_turns, _ = determine_quarter_turns(
        corrected, model, data, qposadr, site_ids, sim_cam_R_opencv(model, data))

    print("\nencoder bias solve (position-only, pan & gripper pinned to 0):")
    print(f"  arm-tag RMS {rms_before:.1f} mm -> {rms_after:.1f} mm over {len(tags)} points")
    for i, name in enumerate(JOINT_NAMES):
        pinned = " (pinned)" if i not in OBSERVABLE_JOINTS else ""
        print(f"    {name:13s} bias {np.degrees(b_full[i]):+6.2f} deg{pinned}")
    for tag in site_ids:
        e = err_mm[tags == tag]
        print(f"    tag {tag} ({ARM_TAG_TO_SITE[tag]}): mean {e.mean():.1f} mm "
              f"(max {e.max():.1f}) over {len(e)} points")
    print(f"  table tag in base: {T_base_table[:3, 3].round(4).tolist()} m  "
          f"(detection repeatability {np.median(t_mm):.1f} mm / {np.median(t_deg):.2f} deg)")
    print("  target a few mm. Worse => XML site placement off or a tag glued askew.")

    save_extrinsics(args.out, T_base_table, T_base_cam, FOCUS_ABSOLUTE,
                    len(samples), rms_after, float(np.median(t_deg)), quarter_turns)
    save_calibration(args.cal_out, b_full, len(samples), rms_before, rms_after)
    print(f"\nwrote {args.out}\nwrote {args.cal_out}")


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
              f"({n_grid} pan=middle grid, {len(poses) - n_grid} panned + wrist-up)")
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
