"""Measure the real camera pipeline latency against the joint encoders.

Wiggle the arm while recording two timestamped series on one clock
(time.monotonic):

  - encoders: bus.read_all() -> calibration + gravity-compliance corrected true
    qpos -> MuJoCo FK -> marker-site world positions. The bus read is ~1.4 ms,
    so this series is effectively instantaneous ground truth.
  - camera: every processed frame's base-frame AprilTag positions, stamped with
    t_recv (the instant cap.read() handed the frame back), via
    CameraMarkerSource's on_frame hook.

The camera series is a time-shifted copy of the FK series: a frame shows the
world as it was around mid-exposure, one pipeline latency (exposure + sensor
readout + USB transfer + MJPG decode) before t_recv. Scanning that shift for
the least-squares alignment — after removing each axis's static offset, the
mm-scale calibration residual — recovers tau = capture -> t_recv. The
policy-visible delay for conf/dr/*.yaml `cam_latency.delay_ms` is
tau + AprilTag detection time; both are printed, with the suggested config
value. The further wait-for-consumption staleness rollouts measure
(marker_age_ms) is the camera/control frame beat, which the sim's CameraSim
reproduces by itself and must NOT be folded into delay_ms.

Modes:
  - --execute (the reliable mode): drive --joint with a sine of --amp rad
    around the boot pose at --freq Hz, streamed with the standard sub-target
    shaping; encoders are sampled once per 15 Hz tick. Default joint is
    wrist_flex: it arcs the finger tag in the world x-z plane, which the
    side-mounted camera sees as image-plane motion. Avoid shoulder_pan from a
    forward pose — its tangential motion runs along the camera's optical axis,
    the noisiest solvePnP direction, whose angle-correlated depth wobble can
    bias the lag fit.
  - default (dry-run, torque stays off): record only, while the arm is moved
    BY HAND. Rough check only — back-driving moves the link before the motor
    encoder registers it (gear play + drivetrain elasticity), and hand force
    violates the gravity-only compliance correction, so encoder FK is no
    longer ground truth: per-axis lags scatter tens of ms, some pinned at 0
    (camera "leading" the encoders). Measured example: a hand-wiggle run
    scattered 0-63 ms across axes where the driven sine gave one clean
    39.5 ms estimate at 0.9 mm residual.

Usage:
    python -m sysid.probe_cam_latency                  # passive: wiggle by hand
    python -m sysid.probe_cam_latency --execute        # self-driven wrist sine
"""

import argparse
import csv
import signal
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from omegaconf import OmegaConf

from real.calib.calibration import load_calibration, load_compliance
from real.calib.compliance import gravity_deflection
from real.rollout.marker_obs import CameraMarkerSource
from real.twin.constants import INTERP_HZ, SERVO_ACCEL, SERVO_POSITION_KP, SERVO_SPEED
from real.twin.control import clamp_raw_delta, stream_sub_targets
from real.twin.mapping import load_joint_maps, rad_to_raw, raw_to_rad
from real.twin.servo_io import ServoBus
from src.base_env import JOINT_NAMES, MARKER_SITE_NAMES
from src.units import max_raw_delta_per_step
from sysid.io import OUT_DIR, REPO_ROOT
from sysid.trajectories import SYSID_DT, SYSID_HZ

DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
CONFIG_YAML = REPO_ROOT / "conf" / "config.yaml"
OUT_DIR_CAM = OUT_DIR / "cam_latency"

# Passive-mode encoder sampling rate (Hz). read_all blocks ~1.4 ms at 1 Mbaud,
# so 50 Hz leaves the bus mostly idle while sampling well above the arm motion
# and camera frame rates.
PASSIVE_HZ = 50.0

# Lag scan grid: the C922 pipeline is expected well under this bound.
MAX_LAG_S = 0.25
LAG_STEP_S = 0.001

# An FK axis with less motion than this carries no timing signal; skip it.
MIN_MOTION_STD_M = 0.005
# Minimum camera detections of a tag for its lag estimate to count.
MIN_CAM_SAMPLES = 30


def solve_lag(t_fk: np.ndarray, fk: np.ndarray, t_cam: np.ndarray, cam: np.ndarray,
              detected: np.ndarray) -> list[dict]:
    """Per-(tag, axis) lag between the FK series and the camera series.

    t_fk (T,), fk (T, n_tags, 3): encoder-FK marker positions (ground truth).
    t_cam (F,), cam (F, n_tags, 3), detected (F, n_tags): camera measurements.

    For each tag axis with enough motion and detections, scans lag in
    [0, MAX_LAG_S]: predicts the camera samples as fk interpolated at
    (t_cam - lag), removes the static per-axis offset, and takes the
    least-squares lag with a parabolic sub-grid refinement. Returns a list of
    {"site", "axis", "lag_s", "motion_std_m"} sorted by motion (strongest
    signal first).
    """
    lags = np.arange(0.0, MAX_LAG_S + LAG_STEP_S / 2, LAG_STEP_S)
    results = []
    for i, site in enumerate(MARKER_SITE_NAMES):
        # Valid camera samples for this tag whose shifted lookup stays inside
        # the FK record for every candidate lag.
        ok = detected[:, i] & (t_cam - MAX_LAG_S >= t_fk[0]) & (t_cam <= t_fk[-1])
        if ok.sum() < MIN_CAM_SAMPLES:
            continue
        tc = t_cam[ok]
        for k, axis in enumerate("xyz"):
            fk_series = fk[:, i, k]
            motion = float(fk_series.std())
            if motion < MIN_MOTION_STD_M:
                continue
            meas = cam[ok, i, k]
            sse = np.empty(len(lags))
            for li, lag in enumerate(lags):
                pred = np.interp(tc - lag, t_fk, fk_series)
                r = meas - pred
                r -= r.mean()
                sse[li] = float(r @ r)
            best = int(np.argmin(sse))
            lag = lags[best]
            if 0 < best < len(lags) - 1:
                # Parabolic refinement through the three points around the min.
                a, b, c = sse[best - 1], sse[best], sse[best + 1]
                denom = a - 2 * b + c
                if denom > 0:
                    lag += LAG_STEP_S * 0.5 * (a - c) / denom
            results.append({"site": site, "axis": axis, "lag_s": float(lag),
                            "motion_std_m": motion})
    results.sort(key=lambda r: -r["motion_std_m"])
    return results


def weighted_lag(results: list[dict]) -> float:
    """Motion-power-weighted mean lag: strong axes carry the timing signal."""
    w = np.array([r["motion_std_m"] ** 2 for r in results])
    lags = np.array([r["lag_s"] for r in results])
    return float((w * lags).sum() / w.sum())


def _true_qpos(model, data, qposadr, raws, jm, direction, qpos_bias, compliance):
    """Encoder raw -> true joint angle, the ArmLoop read model:
    q_true = raw_to_rad(raw) - bias - compliance * tau_grav(...)."""
    out = np.empty((len(raws), len(JOINT_NAMES)))
    for i, raw in enumerate(raws):
        q_bc = raw_to_rad(raw, jm, direction) - qpos_bias
        out[i] = q_bc - gravity_deflection(model, data, qposadr, q_bc, compliance)
    return out


def _fk_marker_pos(model, data, qposadr, qpos, site_ids):
    """World marker-site positions for each true-qpos row, via kinematics only."""
    out = np.empty((len(qpos), len(site_ids), 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[qposadr] = q
        mujoco.mj_kinematics(model, data)
        for j, sid in enumerate(site_ids):
            out[i, j] = data.site_xpos[sid]
    return out


def _record_passive(bus, seconds, stopped) -> tuple[list[float], list[np.ndarray]]:
    """Sample encoders at PASSIVE_HZ while the user wiggles the arm by hand."""
    t_enc: list[float] = []
    raws: list[np.ndarray] = []
    t0 = time.monotonic()
    i = 0
    while not stopped["flag"]:
        now = time.monotonic()
        if now - t0 >= seconds:
            break
        raw = bus.read_all()
        t_enc.append((now + time.monotonic()) / 2.0)
        raws.append(raw)
        i += 1
        next_t = t0 + i / PASSIVE_HZ
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    return t_enc, raws


def _record_sine(bus, jm, direction, joint_idx, amp, freq, seconds, max_raw_delta,
                 raw0, stopped) -> tuple[list[float], list[np.ndarray]]:
    """Drive `joint_idx` with a sine around the boot pose, sampling encoders once
    per SYSID_DT tick. Same sub-target shaping as ArmLoop / sysid.record_real."""
    n_interp = max(1, int(round(INTERP_HZ / SYSID_HZ)))
    sub_dt = SYSID_DT / n_interp
    q0 = raw_to_rad(raw0, jm, direction)

    def write(raw: np.ndarray) -> None:
        bus.write_all(raw, SERVO_SPEED, SERVO_ACCEL)

    t_enc: list[float] = []
    raws: list[np.ndarray] = []
    prev_raw = raw0.copy()
    n_ticks = int(round(seconds * SYSID_HZ))
    for tick in range(n_ticks):
        if stopped["flag"]:
            break
        target = q0.copy()
        target[joint_idx] += amp * np.sin(2 * np.pi * freq * (tick + 1) * SYSID_DT)
        target_raw = rad_to_raw(target, jm, direction)
        target_raw = clamp_raw_delta(prev_raw, target_raw, max_raw_delta)
        stream_sub_targets(prev_raw, target_raw, n_interp, sub_dt, write)
        prev_raw = target_raw

        t0 = time.monotonic()
        raw = bus.read_all()
        t_enc.append((t0 + time.monotonic()) / 2.0)
        raws.append(raw)
    return t_enc, raws


def _write_csvs(stem: Path, t_ref: float, t_enc, qpos_true, t_cam, cam_pos, detected,
                detect_ms):
    enc_path = stem.with_name(stem.name + "_enc.csv")
    with enc_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [f"qpos_{n}" for n in JOINT_NAMES])
        for t, q in zip(t_enc, qpos_true):
            w.writerow([f"{t - t_ref:.5f}", *(f"{v:.6f}" for v in q)])
    cam_path = stem.with_name(stem.name + "_cam.csv")
    with cam_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["t_recv_s"]
        for site in MARKER_SITE_NAMES:
            header += [f"{site}_{a}" for a in "xyz"] + [f"{site}_detected"]
        header.append("detect_ms")
        w.writerow(header)
        for t, pos, det, dms in zip(t_cam, cam_pos, detected, detect_ms):
            row = [f"{t - t_ref:.5f}"]
            for i in range(len(MARKER_SITE_NAMES)):
                row += [f"{v:.6f}" for v in pos[i]] + [int(det[i])]
            row.append(f"{dms:.2f}")
            w.writerow(row)
    return enc_path, cam_path


def _plot(path: Path, t_ref, t_fk, fk, t_cam, cam, detected, results, tau):
    best = results[0]
    i = MARKER_SITE_NAMES.index(best["site"])
    k = "xyz".index(best["axis"])
    ok = detected[:, i]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    ax.plot(t_fk - t_ref, fk[:, i, k], label="encoder FK", lw=1.0)
    off = (cam[ok, i, k] - np.interp(t_cam[ok] - tau, t_fk, fk[:, i, k])).mean()
    ax.plot(t_cam[ok] - t_ref, cam[ok, i, k] - off, ".", ms=3, alpha=0.6,
            label="camera (offset removed)")
    ax.plot(t_cam[ok] - t_ref - tau, cam[ok, i, k] - off, ".", ms=3, alpha=0.6,
            label=f"camera shifted by tau={tau * 1e3:.1f} ms")
    ax.set_title(f"{best['site']} {best['axis']} — strongest-motion axis")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("m")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    per = [f"{r['site']}·{r['axis']}" for r in results]
    ax.bar(per, [r["lag_s"] * 1e3 for r in results],
           color=["C0" if r["motion_std_m"] > 2 * MIN_MOTION_STD_M else "C1"
                  for r in results])
    ax.axhline(tau * 1e3, color="k", ls="--", lw=1.0,
               label=f"weighted tau = {tau * 1e3:.1f} ms")
    ax.set_ylabel("lag [ms]")
    ax.set_title("per-axis lag estimates (weighted by motion power)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true",
                   help="Drive the sine on the servos. Default: passive dry-run "
                        "(torque off — wiggle the arm by hand).")
    p.add_argument("--joint", default="wrist_flex", choices=JOINT_NAMES,
                   help="Joint to drive with --execute (see the module docstring "
                        "for why wrist_flex and not shoulder_pan).")
    p.add_argument("--amp", type=float, default=0.25,
                   help="Sine amplitude (rad) around the boot pose.")
    p.add_argument("--freq", type=float, default=0.4, help="Sine frequency (Hz).")
    p.add_argument("--seconds", type=float, default=30.0, help="Recording length.")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(REPO_ROOT / "real" / "follower_calibration.json"))
    p.add_argument("--family", default="apriltag", choices=["apriltag", "aruco"])
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    qposadr = jm.qposadr()
    direction = np.ones(len(jm.items), dtype=np.int8)
    qpos_bias = load_calibration()
    compliance = load_compliance()
    site_ids = []
    for name in MARKER_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid >= 0, f"site '{name}' not found in model"
        site_ids.append(sid)

    # Camera frames land in this list from the capture thread (GIL-atomic
    # appends); analysis filters them to the encoder recording window.
    frames: list[tuple[float, np.ndarray, np.ndarray, float]] = []

    def on_frame(t_recv, pos, rot, detect_ms):
        frames.append((t_recv, pos.copy(), ~np.all(pos == 0.0, axis=1), detect_ms))
        if len(frames) % 60 == 0:
            n_det = int(frames[-1][2].sum())
            print(f"  camera: {len(frames)} frames, last has {n_det}/"
                  f"{len(MARKER_SITE_NAMES)} arm tags")

    marker_source = CameraMarkerSource(args.family, on_frame=on_frame)
    bus = ServoBus(args.port, jm.servo_ids())

    stopped = {"flag": False}

    def stop(_sig, _frame):
        stopped["flag"] = True
    signal.signal(signal.SIGINT, stop)

    bus.connect()
    try:
        marker_source.start()
        raw0 = bus.read_all()
        q0_true = _true_qpos(model, data, qposadr, [raw0], jm, direction,
                             qpos_bias, compliance)[0]
        xml_low = jm.xml_low()
        xml_high = jm.xml_high()
        slack = 0.05
        if not (np.all(q0_true >= xml_low - slack) and np.all(q0_true <= xml_high + slack)):
            raise SystemExit(f"ABORT: boot qpos outside joint range: {q0_true.tolist()}")

        if args.execute:
            j = JOINT_NAMES.index(args.joint)
            if not (q0_true[j] - args.amp >= xml_low[j]
                    and q0_true[j] + args.amp <= xml_high[j]):
                raise SystemExit(
                    f"ABORT: {args.joint} sine {q0_true[j]:.3f}±{args.amp} rad exceeds "
                    f"joint range [{xml_low[j]:.3f}, {xml_high[j]:.3f}]; recenter the arm.")
            max_raw_delta = max_raw_delta_per_step(
                float(OmegaConf.load(CONFIG_YAML)["action_scale"]))
            bus.set_position_kp(SERVO_POSITION_KP)
            bus.enable_torque_all()
            print(f"driving {args.joint}: ±{args.amp} rad @ {args.freq} Hz "
                  f"for {args.seconds}s")
            t_enc, raws = _record_sine(bus, jm, direction, j, args.amp, args.freq,
                                       args.seconds, max_raw_delta, raw0, stopped)
        else:
            print(f"passive recording for {args.seconds}s — wiggle the arm by hand "
                  f"(slow ~5-10 cm swings, tags facing the camera)")
            print("WARNING: hand-driven runs are a rough check only — back-driving "
                  "reaches the tag before the encoder, so expect scattered per-axis "
                  "lags. Use --execute for the number that goes into conf/dr.")
            t_enc, raws = _record_passive(bus, args.seconds, stopped)
    finally:
        marker_source.stop()
        bus.close()

    assert len(t_enc) > 2 * MIN_CAM_SAMPLES, f"too few encoder samples: {len(t_enc)}"
    t_fk = np.array(t_enc)
    qpos_true = _true_qpos(model, data, qposadr, raws, jm, direction,
                           qpos_bias, compliance)
    fk = _fk_marker_pos(model, data, qposadr, qpos_true, site_ids)

    in_window = [f for f in frames if t_fk[0] <= f[0] <= t_fk[-1]]
    assert len(in_window) >= MIN_CAM_SAMPLES, \
        f"only {len(in_window)} camera frames inside the recording window"
    t_cam = np.array([f[0] for f in in_window])
    cam = np.stack([f[1] for f in in_window])
    detected = np.stack([f[2] for f in in_window])
    detect_ms_all = np.array([f[3] for f in in_window])

    results = solve_lag(t_fk, fk, t_cam, cam, detected)
    if not results:
        raise SystemExit(
            "no (tag, axis) had enough motion + detections — move the arm more "
            "and keep the tags facing the camera")
    tau = weighted_lag(results)
    detect_ms = float(detect_ms_all.mean())

    print("\nper-axis lag (capture -> cap.read return):")
    for r in results:
        print(f"  {r['site']:>14} {r['axis']}: {r['lag_s'] * 1e3:6.1f} ms  "
              f"(motion std {r['motion_std_m'] * 1e3:.1f} mm)")
    spread = np.std([r["lag_s"] for r in results]) * 1e3
    delay = tau * 1e3 + detect_ms
    print(f"\ntau (weighted)        : {tau * 1e3:6.1f} ms  (per-axis std {spread:.1f} ms)")
    print(f"AprilTag detect (mean): {detect_ms:6.1f} ms")
    print(f"capture -> pose available = tau + detect = {delay:.1f} ms")
    print(f"-> conf/dr/*.yaml cam_latency.delay_ms should center on {delay:.0f}; "
          f"repeat this probe a few times and set [lo, hi] to cover the spread.")

    OUT_DIR_CAM.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR_CAM / f"probe_{int(time.time())}"
    enc_path, cam_path = _write_csvs(stem, t_fk[0], t_fk, qpos_true, t_cam, cam,
                                     detected, detect_ms_all)
    plot_path = stem.with_name(stem.name + ".png")
    _plot(plot_path, t_fk[0], t_fk, fk, t_cam, cam, detected, results, tau)
    print(f"saved {enc_path.relative_to(REPO_ROOT)} {cam_path.relative_to(REPO_ROOT)} "
          f"{plot_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
