"""Calibrate the fixed table tag's pose in the arm base frame (`T_base_table`).

The camera measures every tag in its own frame; the only thing it can't know is
how that frame relates to the arm base. The *arm* tags hand it over for free: at
any pose, MuJoCo FK already knows where `marker_finger` / `marker_wrist` sit in
the base frame, and the camera measures those same tags — so each captured frame
yields `T_base_cam = T_base_armtag(FK) ∘ T_cam_armtag⁻¹`. Anchor that into the
fixed table tag (`T_base_table = T_base_cam ∘ T_cam_table`) and average over a
spread of poses. From then on the table tag alone re-derives `T_base_cam` every
frame, robust to the camera being bumped (see real/extrinsics.py).

A tag glued with its printed "up" pointing down/sideways sits at a constant
k·90° offset from the sim marker-site frame; because each arm pose re-expresses
that fixed link-frame rotation differently in the base frame, it shows up as a
large *spread*, not a constant bias. We resolve it per tag using the *approximate*
sim camera orientation as a prior — it only has to be good enough to tell apart
candidates 90° apart — comparing the measured tag orientation to the FK-intended
one and snapping to the nearest quarter turn. The chosen k is applied
(`quarter_turn_mat(-k)`) before solving, and stored so the rollout applies the
same correction to its marker_rot channel.

Capture is hand-posed: the arm boots compliant (torque OFF) so you can backdrive
it; press H to freeze it in the air (torque ON, holding the current pose) so it
doesn't sag, line up a pose where the camera sees the table tag plus at least one
arm tag, and press SPACE. Press H again to go compliant and reposition. Repeat
for ~10 poses covering the workspace, then ENTER to solve. Captured samples are
saved so a solve can be re-run offline with --from-samples (no rig needed). The
post-correction spread is the quality gate — if it stays large, the offset wasn't
a clean 90° (tag glued askew) or the encoder calibration is off.

Controls (focus the preview window):
  H          toggle torque hold (freeze in air <-> compliant backdrive)
  SPACE      capture the current frame (needs table tag + >=1 arm tag visible)
  BACKSPACE  drop the last captured sample
  ENTER / q  finish and solve (needs >= MIN_SAMPLES)
  ESC        abort without solving

Run:
    conda run -n mujoco_env python -m real.calibrate_table_marker
    conda run -n mujoco_env python -m real.calibrate_table_marker --from-samples <json>
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from real.calibrate_camera import FOCUS_ABSOLUTE
from real.camera import open_camera, HEIGHT
from real.detect import make_detector
from real.extrinsics import (
    EXTRINSICS_PATH,
    average_transforms,
    rigid_register,
    rt_to_mat,
    save_extrinsics,
    snap_inplane_offset,
    transform_spread,
)
from real.marker_spec import ARM_TAG_TO_SITE, MARKER_EXPOSURE, MARKER_GAIN, TABLE_TAG_ID
from real.overlay import annotate_detections
from real.pose import PoseEstimator, load_intrinsics
from real.twin.constants import (
    SERVO_ACCEL,
    SERVO_POSITION_DEADZONE,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from real.twin.mapping import load_joint_maps, raw_to_rad
from real.twin.servo_io import ServoBus
from src.base_env import TAG_CAM_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene_lift.xml"
DEFAULT_CAL = REPO_ROOT / "real" / "follower_calibration.json"
SAMPLES_PATH = REPO_ROOT / "real" / "table_marker_samples.json"
MIN_SAMPLES = 8


def site_mat(data, site_id):
    """4×4 world(base)-frame transform of a MuJoCo site from its xpos/xmat."""
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = data.site_xpos[site_id]
    return T


def sim_cam_R_opencv(model, data):
    """Approximate camera orientation (world->cam, OpenCV convention) from the sim.

    MuJoCo's camera frame is OpenGL (looks down -z, +y up); solvePnP is OpenCV
    (+z forward, +y down), so flip y and z. Only used as a coarse prior to pick
    the right quarter turn, where being tens of degrees off is harmless."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, TAG_CAM_NAME)
    assert cam_id >= 0, f"camera {TAG_CAM_NAME!r} not found in model"
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    R_gl = data.cam_xmat[cam_id].reshape(3, 3).copy()
    return R_gl @ np.diag([1.0, -1.0, -1.0])


def enter_hold(bus):
    """Freeze the arm at its current pose: goal=present (no snap), then torque on.

    Writes the goal before enabling torque so the servo holds where the hand left
    it instead of jerking to a stale goal register. Uses the same Kp/deadzone as
    the rollouts so gravity-loaded joints hold without sagging."""
    raw = bus.read_all()
    bus.write_all(raw, SERVO_SPEED, SERVO_ACCEL)
    bus.set_position_kp(SERVO_POSITION_KP)
    bus.set_position_deadzone(SERVO_POSITION_DEADZONE)
    bus.enable_torque_all()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    p.add_argument("--family", default="apriltag", choices=["apriltag", "aruco"])
    p.add_argument("--out", default=EXTRINSICS_PATH)
    p.add_argument("--samples-out", default=str(SAMPLES_PATH),
                   help="Where captured samples are written for offline re-solve.")
    p.add_argument("--from-samples", default=None,
                   help="Skip capture; load samples from this JSON and just solve.")
    return p.parse_args()


def detect_poses(frame, detector, estimator):
    """Detect tags and return (dets, {id: (rvec, tvec)}) for the ids we calibrate."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector.detect(gray)
    wanted = set(ARM_TAG_TO_SITE) | {TABLE_TAG_ID}
    poses = {d.id: estimator.estimate(d) for d in dets if d.id in wanted}
    return dets, poses


def save_samples(path, samples):
    out = {"samples": [
        {"qpos": qpos.tolist(),
         "tags": {str(tag): {"rvec": rvec.tolist(), "tvec": tvec.tolist()}
                  for tag, (rvec, tvec) in poses.items()}}
        for qpos, poses in samples]}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    samples = []
    for s in data["samples"]:
        poses = {int(tag): (np.array(t["rvec"]), np.array(t["tvec"]))
                 for tag, t in s["tags"].items()}
        samples.append((np.array(s["qpos"]), poses))
    return samples


def determine_quarter_turns(samples, model, data, qposadr, site_ids, R_cam_approx):
    """Vote each arm tag's in-plane glue offset k (90° multiples) via the sim prior.

    Returns (quarter_turns {tag: k}, report {tag: (k, residuals_deg, votes)})."""
    votes = {tag: [] for tag in site_ids}
    resid = {tag: [] for tag in site_ids}
    for qpos, poses in samples:
        data.qpos[qposadr] = qpos
        mujoco.mj_kinematics(model, data)
        for tag, sid in site_ids.items():
            if tag not in poses:
                continue
            R_meas = R_cam_approx @ rt_to_mat(*poses[tag])[:3, :3]
            k, res = snap_inplane_offset(data.site_xmat[sid].reshape(3, 3), R_meas)
            votes[tag].append(k)
            resid[tag].append(res)
    quarter_turns, report = {}, {}
    for tag in site_ids:
        if not votes[tag]:
            raise RuntimeError(f"tag {tag} never captured; cannot determine its offset")
        k = int(np.bincount(votes[tag]).argmax())
        quarter_turns[tag] = k
        report[tag] = (k, np.array(resid[tag]), np.array(votes[tag]))
    return quarter_turns, report


def solve_camera(samples, model, data, qposadr, site_ids):
    """Register T_base_cam from tag *centres* (Umeyama): pair each tag's camera-frame
    position (tvec) with its base-frame FK position, across all poses/tags.

    Position-only, so it is immune to the glue rotation and to solvePnP's rvec
    flips on the small arm tags — the failure modes that wreck an orientation
    bridge. Returns (T_base_cam, rms_mm, tags, err_mm) with per-point residuals."""
    src, dst, tags = [], [], []
    for qpos, poses in samples:
        data.qpos[qposadr] = qpos
        mujoco.mj_kinematics(model, data)
        for tag, sid in site_ids.items():
            if tag not in poses:
                continue
            src.append(poses[tag][1])               # tvec: tag centre in camera frame
            dst.append(data.site_xpos[sid].copy())  # tag centre in base frame (FK)
            tags.append(tag)
    src, dst, tags = np.array(src), np.array(dst), np.array(tags)
    T_base_cam, rms = rigid_register(src, dst)
    err_mm = np.linalg.norm(dst - (src @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]), axis=1) * 1000.0
    return T_base_cam, rms * 1000.0, tags, err_mm


def solve_table(samples, T_base_cam):
    """Anchor the fixed table tag: T_base_table = T_base_cam ∘ T_cam_table, averaged
    over poses. The candidates' spread is the table-detection repeatability (camera
    and table are both fixed, so it should be tiny)."""
    cands = np.array([T_base_cam @ rt_to_mat(*poses[TABLE_TAG_ID]) for _, poses in samples])
    T_base_table = average_transforms(cands)
    trans_mm, rot_deg = transform_spread(cands, T_base_table)
    return T_base_table, trans_mm, rot_deg


def capture(args, jm, direction, detector, estimator):
    """Run the live capture loop; return the samples list, or None if aborted."""
    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()
    bus.disable_torque_all()  # hand-pose the arm; we only read encoders
    cap = open_camera(focus=FOCUS_ABSOLUTE, exposure=MARKER_EXPOSURE, gain=MARKER_GAIN)

    samples = []  # list of (qpos[6], {id: (rvec, tvec)})
    held = False  # torque hold: False = compliant backdrive, True = frozen in air
    aborted = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed mid-session")
            dets, poses = detect_poses(frame, detector, estimator)
            arm_seen = [t for t in ARM_TAG_TO_SITE if t in poses]
            table_seen = TABLE_TAG_ID in poses
            capturable = table_seen and bool(arm_seen)

            view = frame.copy()
            annotate_detections(view, dets, estimator)
            status = (f"samples {len(samples)}/{MIN_SAMPLES}+   "
                      f"{'HELD' if held else 'COMPLIANT'}   "
                      f"table={'ok' if table_seen else 'MISSING'}   "
                      f"arm={arm_seen if arm_seen else 'none'}   "
                      f"{'READY (SPACE)' if capturable else 'need table + >=1 arm tag'}")
            cv2.putText(view, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if capturable else (0, 0, 255), 2)
            cv2.putText(view, "H hold  SPACE capture  BACKSPACE undo  ENTER/q solve  ESC abort",
                        (12, HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("calibrate_table_marker", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("h"), ord("H")):                # toggle torque hold
                if held:
                    bus.disable_torque_all()
                    held = False
                    print("compliant: backdrive to the next pose")
                else:
                    enter_hold(bus)
                    held = True
                    print("held: arm frozen in the air")
            elif key == 32:                                # SPACE
                if not capturable:
                    print("skip: need the table tag and at least one arm tag in view")
                    continue
                qpos = raw_to_rad(bus.read_all(), jm, direction)
                samples.append((qpos.copy(), poses))
                print(f"captured #{len(samples)}: arm tags {arm_seen}")
            elif key == 8 and samples:                     # BACKSPACE
                samples.pop()
                print(f"dropped; {len(samples)} left")
            elif key in (13, ord("q")):                    # ENTER / q
                break
            elif key == 27:                                # ESC
                aborted = True
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        bus.close()
    return None if aborted else samples


def report_and_save(args, samples, model, data, jm, site_ids):
    qposadr = jm.qposadr()
    T_base_cam, rms_mm, tags, err_mm = solve_camera(samples, model, data, qposadr, site_ids)
    T_base_table, t_mm, t_deg = solve_table(samples, T_base_cam)
    quarter_turns, qt_report = determine_quarter_turns(
        samples, model, data, qposadr, site_ids, sim_cam_R_opencv(model, data))

    print("\ncamera solve (position-based, immune to tag orientation / solvePnP flips):")
    print(f"  registration RMS {rms_mm:.1f} mm over {len(tags)} points")
    for tag in site_ids:
        e = err_mm[tags == tag]
        print(f"    tag {tag} ({ARM_TAG_TO_SITE[tag]}): mean {e.mean():.1f} mm "
              f"(max {e.max():.1f}) over {len(e)} points")
    print("  target a few mm. Worse => encoder calibration or XML site *position* "
          "is off (calibration_plan.md Step 1).")
    print(f"  table tag in base: {T_base_table[:3, 3].round(4).tolist()} m  "
          f"(detection repeatability {np.median(t_mm):.1f} mm / {np.median(t_deg):.2f} deg)")

    print("\nmarker_rot glue offset (rollout rotation channel):")
    for tag, (k, res, votes) in qt_report.items():
        unanimous = "unanimous" if len(set(votes.tolist())) == 1 else f"votes={votes.tolist()}"
        print(f"  tag {tag} ({ARM_TAG_TO_SITE[tag]}): k={k} ({k * 90} deg), {unanimous}")
    any_res = np.median(qt_report[next(iter(site_ids))][1])
    print(f"  (snap residual ~{any_res:.0f} deg mostly reflects the rough sim-camera "
          "prior, not glue quality; the prior only picks the integer k.)")

    save_extrinsics(args.out, T_base_table, T_base_cam, FOCUS_ABSOLUTE,
                    len(samples), rms_mm, float(np.median(t_deg)), quarter_turns)
    print(f"\nwrote {args.out}")


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
        estimator = PoseEstimator(*load_intrinsics())
        detector = make_detector(args.family)
        print(__doc__)
        print(f"detector={args.family}  arm tags={list(ARM_TAG_TO_SITE)}  table tag={TABLE_TAG_ID}\n")
        samples = capture(args, jm, direction, detector, estimator)
        if samples is None:
            print("aborted")
            return
        save_samples(args.samples_out, samples)
        print(f"saved {len(samples)} samples to {args.samples_out}")

    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(f"only {len(samples)} samples; need >= {MIN_SAMPLES}")
    report_and_save(args, samples, model, data, jm, site_ids)


if __name__ == "__main__":
    main()
