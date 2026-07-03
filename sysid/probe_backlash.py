"""Per-joint backlash / hysteresis probe: approach identical targets from opposite
directions and compare how far the *link* moved (camera tags) vs how far the *motor*
moved (encoder).

Why: the qpos calibration's residual is load-correlated, and on a gravity-only sweep
a linear compliance (bracket/horn flex, deg per N·m) and a backlash step (±k·sign(τ))
fit the same data equally well — the sweep never reverses the load on a joint at a
fixed pose. This probe measures the hysteresis directly: drive to the same encoder
target twice, once arriving from below (target−δ first) and once from above
(target+δ first), and capture both the encoder reading and the camera-measured arm
tag positions after settling. Per (pose, joint, tag):

    encoder Δ = motor-side hysteresis (firmware position deadzone),
    link Δ    = camera tag displacement projected on the joint's site Jacobian,
    play      = link Δ − encoder Δ: slack between motor and link the encoder can't see.

Interpretation: play ≈ 0 where |gravity torque| is large means gravity keeps one
gear flank loaded regardless of approach — the calibration solve's quasi-static
`_apply_gravity_slack` settle assumption holds there. Large play at small |τ|
measures the true backlash width, which sizes both the sim `backlash` joint class
(±0.5° guess in so101.xml) and the wrist term of any compliance model.

Sensitivity: encoder LSB is 0.088°; the camera sees 1° of link motion as ~2-4 mm at
the lift/elbow lever arms (12-frame median tvec noise is sub-mm), so play well below
the ±0.5° sim guess is resolvable. Tags whose lever arm to the probed joint is tiny
(e.g. the wrist tag is proximal to wrist_roll: zero lever) are skipped.

The probe drives the arm through base poses drawn from the calibration sweep
(collision-free, both tags visible); every transition, including the ±δ approach
legs, is verified collision-free and above the fingertip floor in sim before any
hardware moves. Dry-run (no --execute) previews the full drive plan in the sim
viewer; --stream-port serves it as MJPEG (the panel's sim view), and the --execute
run streams the annotated camera feed instead.

Run:
    conda run -n mujoco_env python -m sysid.probe_backlash              # dry-run preview
    conda run -n mujoco_env python -m sysid.probe_backlash --execute
    conda run -n mujoco_env python -m sysid.probe_backlash --from-records <json>
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from real.calibrate_qpos import (
    CAPTURE_FRAMES,
    DEFAULT_CAL,
    DEFAULT_XML,
    INTERP_CHECK_STEPS,
    MIN_EE_Z,
    MarkerCamera,
    drive_to,
    generate_poses,
    preview_poses,
)
from real.calibration import load_calibration
from real.extrinsics import load_extrinsics
from real.marker_spec import ARM_TAG_TO_SITE, TABLE_TAG_ID
from real.twin.constants import SERVO_POSITION_DEADZONE, SERVO_POSITION_KP
from real.twin.mapping import JOINT_NAMES, load_joint_maps, raw_to_rad
from real.twin.servo_io import ServoBus
from src.units import max_raw_delta_per_step
from sysid.io import OUT_DIR
from sysid.record_real import CONFIG_YAML

RECORDS_PATH = OUT_DIR / "backlash_records.json"

# All joints that move at least one arm tag. The gripper moves neither. Pan's
# encoder *bias* is gauge-unobservable to the calibration, but its hysteresis is
# a plain link-vs-motor difference — fully measurable here.
PROBE_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

APPROACH_DEG = 6.0      # ±δ approach offset; must exceed any plausible play so the
#                         gear flank is engaged from a known side before the final leg
N_POSES = 3             # base poses drawn from the calibration sweep
LIMIT_MARGIN_RAD = 0.02  # ±δ poses must clear the XML limits by this, or rad_to_raw
#                          clips and the approach silently shrinks
MIN_PAIRS_PER_JOINT = 2  # fail loud if sim validation leaves a joint thinner than this
MIN_LEVER_M = 0.02      # skip a tag whose |site Jacobian| to the joint is below this
#                         (m/rad): the projection would amplify camera noise into
#                         meaningless degrees (e.g. wrist tag ⊥ wrist_roll: zero lever)
TABLE_DRIFT_WARN_MM = 1.0  # table-tag apparent motion between the two captures: the
#                            camera and table are fixed, so past this something moved


@dataclass(frozen=True)
class Step:
    """One drive-plan waypoint. `key` is (pose_idx, joint_name, side) at the two
    capture arrivals ("minus"/"plus" = the side the target was approached from),
    None for the ±δ staging waypoints."""
    qpos: np.ndarray
    key: tuple[int, str, str] | None


def _path_problem(model, data, qposadr, ee_id, a, b):
    """Why the straight joint-space line a→b is unsafe, or None if it is safe
    (collision-free and fingertip above MIN_EE_Z throughout)."""
    for u in np.linspace(0.0, 1.0, INTERP_CHECK_STEPS):
        data.qpos[qposadr] = (1.0 - u) * a + u * b
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            return f"collides in sim at u={u:.2f}"
        if data.site_xpos[ee_id][2] < MIN_EE_Z:
            return f"fingertip below floor at u={u:.2f}"
    return None


def build_plan(model, data, jm, joints, n_poses, approach_rad):
    """Drive plan over `n_poses` calibration-sweep poses × `joints`, four steps per
    combo: −δ staging, target (capture "minus"), +δ staging, target (capture "plus").

    Every consecutive transition — including from the previous combo's end — is
    verified safe in sim; a combo whose ±δ leg is out of limits or unsafe is skipped
    with a printed reason. Fails loud if any joint keeps < MIN_PAIRS_PER_JOINT poses."""
    qposadr = jm.qposadr()
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert ee_id >= 0, "site 'gripperframe' not found in model"
    lo = jm.xml_low() + LIMIT_MARGIN_RAD
    hi = jm.xml_high() - LIMIT_MARGIN_RAD

    sweep, _ = generate_poses(model, data, jm)
    base_poses = [sweep[i] for i in
                  np.linspace(0, len(sweep) - 1, n_poses).round().astype(int)]

    steps: list[Step] = []
    kept_pairs = {name: 0 for name in joints}
    for pose_idx, q in enumerate(base_poses):
        for name in joints:
            j = JOINT_NAMES.index(name)
            q_minus, q_plus = q.copy(), q.copy()
            q_minus[j] -= approach_rad
            q_plus[j] += approach_rad
            if not (lo[j] <= q_minus[j] and q_plus[j] <= hi[j]):
                print(f"  skip pose {pose_idx} {name}: ±δ leaves the joint limits")
                continue
            combo = [Step(q_minus, None), Step(q, (pose_idx, name, "minus")),
                     Step(q_plus, None), Step(q, (pose_idx, name, "plus"))]
            prev = steps[-1].qpos if steps else q
            problem = None
            for step in combo:
                problem = _path_problem(model, data, qposadr, ee_id, prev, step.qpos)
                if problem is not None:
                    break
                prev = step.qpos
            if problem is not None:
                print(f"  skip pose {pose_idx} {name}: {problem}")
                continue
            steps.extend(combo)
            kept_pairs[name] += 1

    thin = {name: n for name, n in kept_pairs.items() if n < MIN_PAIRS_PER_JOINT}
    if thin:
        raise RuntimeError(
            f"joints with < {MIN_PAIRS_PER_JOINT} valid probe poses: {thin}. "
            "Raise --n-poses or lower --approach-deg.")
    return steps


def capture_records(args, jm, direction, steps, max_raw_delta):
    """Drive the plan and capture (encoder qpos, tag tvecs) at each keyed arrival."""
    cam = MarkerCamera(args.family, args.stream_port)
    cam.start()
    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()
    records = []
    try:
        bus.set_position_kp(SERVO_POSITION_KP)
        bus.set_position_deadzone(SERVO_POSITION_DEADZONE)
        bus.enable_torque_all()
        prev_raw = bus.read_all().copy()
        for i, step in enumerate(steps):
            prev_raw = drive_to(bus, jm, direction, step.qpos, prev_raw, max_raw_delta)
            if cam.error is not None:
                raise cam.error
            if step.key is None:
                continue
            pose_idx, name, side = step.key
            tag_poses = cam.capture_median(CAPTURE_FRAMES)
            arm_seen = [t for t in ARM_TAG_TO_SITE if t in tag_poses]
            qpos = raw_to_rad(bus.read_all(), jm, direction)
            records.append({
                "pose": pose_idx, "joint": name, "side": side,
                "qpos": qpos.tolist(),
                "tags": {str(t): tvec.tolist() for t, (_, tvec) in tag_poses.items()},
            })
            print(f"  step {i + 1}/{len(steps)}: pose {pose_idx} {name} {side}, "
                  f"arm tags {arm_seen or 'NONE'}")
    finally:
        bus.close()  # torque off
        cam.close()
    return records


def save_records(path, records, approach_deg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"approach_deg": approach_deg, "records": records}, f, indent=2)


def load_records(path):
    with open(path) as f:
        return json.load(f)["records"]


def _gravity_torque(model, data, qposadr, qpos, dofadr):
    """Gravity generalized force on one motor dof at pose `qpos` (qvel = 0)."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[qposadr] = qpos
    mujoco.mj_forward(model, data)
    return float(data.qfrc_bias[dofadr])


def _site_jac_col(model, data, qposadr, qpos, sid, dofadr):
    """Site translational Jacobian column of one motor dof at pose `qpos` (m/rad)."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[qposadr] = qpos
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, sid)
    return jacp[:, dofadr].copy()


def hysteresis_estimates(records, model, data, jm, qpos_bias, R_base_cam):
    """Pair minus/plus records into per-(pose, joint, tag) hysteresis rows.

    Each row: enc_delta_deg (motor-side), link_delta_deg (camera tag displacement
    rotated into the base frame and projected on the joint's site Jacobian, signed),
    play_deg = link − encoder, plus context: tau_nm (gravity torque at the pose),
    lever_m (|Jacobian|), table_drift_mm (fixed table tag's apparent motion between
    the two captures — should be ~0). The camera translation cancels in the
    difference, so only the extrinsics *rotation* enters."""
    qposadr = jm.qposadr()
    site_ids = {}
    for tag, site_name in ARM_TAG_TO_SITE.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert sid >= 0, f"site {site_name!r} not found in model"
        site_ids[tag] = sid
    dofadrs = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint {name!r} not found in model"
        dofadrs[name] = int(model.jnt_dofadr[jid])

    by_key = {(r["pose"], r["joint"], r["side"]): r for r in records}
    assert len(by_key) == len(records), "duplicate (pose, joint, side) record"
    rows = []
    for pose_idx, name in sorted({(r["pose"], r["joint"]) for r in records}):
        minus = by_key.get((pose_idx, name, "minus"))
        plus = by_key.get((pose_idx, name, "plus"))
        if minus is None or plus is None:
            print(f"  pose {pose_idx} {name}: only one side captured, skipping pair")
            continue
        j = JOINT_NAMES.index(name)
        q_minus = np.array(minus["qpos"]) - qpos_bias
        q_plus = np.array(plus["qpos"]) - qpos_bias
        enc_delta = q_plus[j] - q_minus[j]
        tau = _gravity_torque(model, data, qposadr, q_minus, dofadrs[name])
        table_drift = float("nan")
        key = str(TABLE_TAG_ID)
        if key in minus["tags"] and key in plus["tags"]:
            table_drift = float(np.linalg.norm(
                np.array(plus["tags"][key]) - np.array(minus["tags"][key]))) * 1000.0
        for tag, sid in site_ids.items():
            key = str(tag)
            if key not in minus["tags"] or key not in plus["tags"]:
                continue
            jac = _site_jac_col(model, data, qposadr, q_minus, sid, dofadrs[name])
            lever = float(np.linalg.norm(jac))
            if lever < MIN_LEVER_M:
                continue   # tag barely moves under this joint; projection is noise
            dp_cam = np.array(plus["tags"][key]) - np.array(minus["tags"][key])
            dp_base = R_base_cam @ dp_cam
            link_delta = float(dp_base @ jac / (jac @ jac))
            rows.append({
                "pose": pose_idx, "joint": name, "tag": tag,
                "enc_delta_deg": float(np.degrees(enc_delta)),
                "link_delta_deg": float(np.degrees(link_delta)),
                "play_deg": float(np.degrees(link_delta - enc_delta)),
                "tau_nm": tau, "lever_m": lever, "table_drift_mm": table_drift,
            })
    return rows


def report(rows):
    print("\nhysteresis (plus-approach vs minus-approach, identical encoder targets):")
    print("  play = link Δ (camera) − encoder Δ (motor): slack the encoder can't see.")
    print("  ~0 at large |tau|: gravity keeps one flank loaded (the calibration's")
    print("  settle assumption holds). Large at small |tau|: true backlash width —")
    print("  compare against the sim backlash class (±0.5°) and re-fit if far off.")
    for name in JOINT_NAMES:
        joint_rows = [r for r in rows if r["joint"] == name]
        if not joint_rows:
            continue
        play = np.array([r["play_deg"] for r in joint_rows])
        print(f"  {name:13s} play median {np.median(play):+5.2f} deg "
              f"(min {play.min():+5.2f}, max {play.max():+5.2f}) "
              f"over {len(joint_rows)} tag-pairs")
        for r in joint_rows:
            drift = (f", table drift {r['table_drift_mm']:.1f} mm  <-- CAMERA/TABLE MOVED?"
                     if r["table_drift_mm"] > TABLE_DRIFT_WARN_MM else "")
            print(f"      pose {r['pose']} tag {r['tag']} "
                  f"({ARM_TAG_TO_SITE[r['tag']]}): enc {r['enc_delta_deg']:+.2f}  "
                  f"link {r['link_delta_deg']:+.2f}  play {r['play_deg']:+.2f} deg  "
                  f"(tau {r['tau_nm']:+.2f} N.m, lever {r['lever_m'] * 1000:.0f} mm/rad"
                  f"{drift})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    p.add_argument("--family", default="apriltag", choices=["apriltag", "aruco"])
    p.add_argument("--joints", default=",".join(PROBE_JOINTS),
                   help="Comma-separated joints to probe.")
    p.add_argument("--n-poses", type=int, default=N_POSES,
                   help="Base poses drawn from the calibration sweep.")
    p.add_argument("--approach-deg", type=float, default=APPROACH_DEG,
                   help="±δ approach offset per joint, degrees.")
    p.add_argument("--execute", action="store_true",
                   help="Drive the arm and capture. Default: dry-run (preview the plan).")
    p.add_argument("--stream-port", type=int, default=None,
                   help="Serve the dry-run preview as MJPEG on this port (panel sim view).")
    p.add_argument("--out", default=str(RECORDS_PATH))
    p.add_argument("--from-records", default=None,
                   help="Skip capture; load records from this JSON and just analyze.")
    return p.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    direction = np.ones(6, dtype=np.int8)  # follower: no inversions (verified via twin)
    joints = tuple(args.joints.split(","))
    for name in joints:
        assert name in PROBE_JOINTS, (
            f"joint {name!r} not probeable (choose from {PROBE_JOINTS}; "
            "the gripper moves neither tag)")

    if args.from_records:
        records = load_records(args.from_records)
        print(f"loaded {len(records)} records from {args.from_records}")
    else:
        steps = build_plan(model, data, jm, joints, args.n_poses,
                           np.radians(args.approach_deg))
        n_pairs = sum(1 for s in steps if s.key is not None) // 2
        print(f"drive plan: {len(steps)} waypoints, {n_pairs} minus/plus pairs "
              f"({len(joints)} joints, ±{args.approach_deg:g} deg approaches)")
        if not args.execute:
            print("dry-run: previewing the plan in sim "
                  "(pass --execute to drive the arm and capture).")
            preview_poses(model, data, jm, [s.qpos for s in steps], args.stream_port)
            return
        max_raw_delta = max_raw_delta_per_step(
            float(OmegaConf.load(CONFIG_YAML)["action_scale"]))
        records = capture_records(args, jm, direction, steps, max_raw_delta)
        save_records(Path(args.out), records, args.approach_deg)
        print(f"saved {len(records)} records to {args.out}")

    qpos_bias = load_calibration()
    _, T_base_cam, _, _ = load_extrinsics()
    rows = hysteresis_estimates(records, model, data, jm, qpos_bias, T_base_cam[:3, :3])
    report(rows)


if __name__ == "__main__":
    main()
