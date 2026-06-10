"""Isolate the source of joint shake (shoulder_pan chatter in particular).

Samples ONE servo's position+speed at several hundred Hz (single-servo read,
see ServoBus.read_pos_speed) — fast enough to resolve the mechanical resonance
that the 15 Hz rollout logs alias — while the rest of the arm holds its boot
pose under torque. Two modes:

  hold   Torque on, target frozen at the boot pose, just record. Any motion
         here is the servo's own position-loop limit cycle (PID hunting across
         the gear backlash), independent of all command shaping. Tap the arm
         mid-recording to also get a ring-down (resonant frequency + damping).
  sweep  Stream a slow triangle wave on the probed joint via the same
         sub-target writes the rollout uses, dwelling --dwell-s at each
         extreme. Reproduces the "gentle joystick motion" case with everything
         else controlled; the summary's settled error at the dwells measures
         steady-state sag under load (the cost of lowering Kp on
         gravity-loaded joints).

`--kp/--deadzone` apply to the probed servo only (others keep the standard
constants); `--speed/--accel` apply to all writes. The standard register
values from real/twin/constants.py are restored on exit because Kp and the
deadzone live in servo EEPROM and would otherwise outlive the experiment.

Run a matrix of settings and compare the printed rms / dominant-frequency
summary, e.g.:

    python -m scripts.shake_probe --mode hold --duration 10
    python -m scripts.shake_probe --mode hold --deadzone 1   # factory deadzone
    python -m scripts.shake_probe --mode sweep --accel 5     # pre-bump accel
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from real.twin.constants import (
    SERVO_ACCEL,
    SERVO_POSITION_DEADZONE,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from real.twin.mapping import JOINT_NAMES, JOINT_TO_SERVO
from real.twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", default="shoulder_pan", choices=JOINT_NAMES)
    p.add_argument("--mode", default="hold", choices=("hold", "sweep"))
    p.add_argument("--watch-joint", default=None, choices=JOINT_NAMES,
                   help="Sweep mode: record THIS joint while --joint is swept "
                        "(default: the swept joint itself). Sweeping lift/elbow "
                        "while watching shoulder_pan measures reaction-torque "
                        "rattle through the pan gear backlash.")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Recording length in seconds.")
    p.add_argument("--kp", type=int, default=None,
                   help="Position-loop Kp for the probed servo "
                        "(default: value from constants.py).")
    p.add_argument("--deadzone", type=int, default=None,
                   help="Position-correction deadzone for the probed servo "
                        "(default: value from constants.py).")
    p.add_argument("--speed", type=int, default=SERVO_SPEED,
                   help="SyncWritePosEx speed argument for all writes.")
    p.add_argument("--accel", type=int, default=SERVO_ACCEL,
                   help="SyncWritePosEx accel argument for all writes.")
    p.add_argument("--sweep-range", type=int, default=150,
                   help="Sweep mode: triangle amplitude in raw units around the "
                        "boot pose (150 ≈ 13°). Make sure the joint has clearance.")
    p.add_argument("--sweep-raw-per-s", type=float, default=100.0,
                   help="Sweep mode: target velocity in raw units/s "
                        "(100 ≈ 8.8°/s; the policy's typical pan speed is "
                        "~70 raw/s, its peak ~680 raw/s).")
    p.add_argument("--dwell-s", type=float, default=2.0,
                   help="Sweep mode: pause this long at each sweep extreme; the "
                        "summary reports the settled position error there "
                        "(gravity sag on loaded joints). 0 = plain triangle.")
    p.add_argument("--write-hz", type=float, default=100.0,
                   help="Sweep mode: sub-target write rate (matches INTERP_HZ).")
    p.add_argument("--port", default="/dev/ttyACM0")
    return p.parse_args()


def record_hold(bus: ServoBus, sid: int, boot_target: int,
                duration: float) -> list[tuple[float, int, int, int]]:
    """Frozen target; KeyboardInterrupt ends the recording, keeping the data."""
    rows = []
    t0 = time.monotonic()
    try:
        while (t := time.monotonic() - t0) < duration:
            pos, speed = bus.read_pos_speed(sid)
            rows.append((t, boot_target, pos, speed))
    except KeyboardInterrupt:
        print("\ninterrupted — summarizing what was recorded")
    return rows


def sweep_offset(t: float, amplitude: int, raw_per_s: float, dwell_s: float) -> int:
    """Triangle 0 -> +A -> -A -> 0 at slope raw_per_s, pausing dwell_s at each
    extreme so the steady-state (gravity-sag) tracking error can settle and be
    measured. dwell_s=0 gives a plain triangle."""
    ramp_quarter = amplitude / raw_per_s
    period = 4.0 * ramp_quarter + 2.0 * dwell_s
    u = t % period
    if u < ramp_quarter:                      # 0 -> +A
        return round(u * raw_per_s)
    u -= ramp_quarter
    if u < dwell_s:                           # dwell at +A
        return amplitude
    u -= dwell_s
    if u < 2.0 * ramp_quarter:                # +A -> -A
        return round(amplitude - u * raw_per_s)
    u -= 2.0 * ramp_quarter
    if u < dwell_s:                           # dwell at -A
        return -amplitude
    u -= dwell_s
    return round(-amplitude + u * raw_per_s)  # -A -> 0


def record_sweep(bus: ServoBus, joint_idx: int, watch_sid: int, watch_idx: int,
                 boot_raw: np.ndarray,
                 amplitude: int, raw_per_s: float, dwell_s: float, write_hz: float,
                 duration: float, speed: int, accel: int) -> list[tuple[float, int, int, int]]:
    """Sweep --joint, record --watch-joint. The logged target is the watched
    joint's commanded value, so tracking error stays meaningful both when
    watching the swept joint and when watching a held one (rattle test)."""
    rows = []
    targets = boot_raw.copy()
    write_dt = 1.0 / write_hz
    # Per-write step clamp: never jump more than the commanded sweep velocity
    # implies (+ slack), even if a slow read delays a write and the
    # time-indexed triangle skips ahead.
    max_step = int(np.ceil(raw_per_s * write_dt)) + 2
    t0 = time.monotonic()
    next_write = 0.0
    try:
        while (t := time.monotonic() - t0) < duration:
            if t >= next_write:
                want = boot_raw[joint_idx] + sweep_offset(t, amplitude, raw_per_s, dwell_s)
                step = int(np.clip(want - targets[joint_idx], -max_step, max_step))
                targets[joint_idx] += step
                bus.write_all(targets, speed, accel)
                next_write = t + write_dt
            pos, spd = bus.read_pos_speed(watch_sid)
            rows.append((time.monotonic() - t0, int(targets[watch_idx]), pos, spd))
    except KeyboardInterrupt:
        print("\ninterrupted — summarizing what was recorded")
    return rows


def summarize(rows: list[tuple[float, int, int, int]]) -> None:
    t = np.array([r[0] for r in rows])
    target = np.array([r[1] for r in rows], dtype=np.float64)
    pos = np.array([r[2] for r in rows], dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    rate = 1.0 / dt
    # Detrend with a 0.5 s moving average so slow sweep motion drops out and
    # only the oscillation remains.
    k = max(3, int(round(0.5 / dt)) | 1)
    if len(pos) < 3 * k:
        print(f"\nonly {len(rows)} samples — too short to analyze")
        return
    smooth = np.convolve(pos, np.ones(k) / k, mode="same")
    resid = (pos - smooth)[k:-k]
    err = pos - target
    print(f"\n{len(rows)} samples over {t[-1]:.1f} s  ->  {rate:.0f} Hz effective")
    print(f"tracking error:  mean {np.mean(err):+.1f} raw, rms {np.std(err):.2f} raw")
    print(f"oscillation:     rms {np.std(resid):.2f} raw, "
          f"p2p {np.ptp(resid):.1f} raw  (1 raw = 0.088°)")
    # Settled error: wherever the target sat still for >= 1 s (sweep dwells,
    # or the whole recording in hold mode), report how far the joint ended up
    # from it — steady-state sag the P-loop cannot close under load.
    run_start = 0
    settled = []
    for i in range(1, len(target) + 1):
        if i == len(target) or target[i] != target[run_start]:
            if t[i - 1] - t[run_start] >= 1.0:
                settled.append((target[run_start] - target[0], err[i - 1]))
            run_start = i
    if settled:
        lines = ", ".join(f"{e:+.0f} raw at offset {off:+.0f}" for off, e in settled)
        print(f"settled error:   {lines}")
    if len(resid) > 256:
        freqs = np.fft.rfftfreq(len(resid), dt)
        power = np.abs(np.fft.rfft(resid * np.hanning(len(resid)))) ** 2
        order = np.argsort(power[1:])[::-1][:3] + 1
        peaks = ", ".join(f"{freqs[i]:.1f} Hz" for i in order)
        print(f"dominant freqs:  {peaks}  (Nyquist {rate / 2:.0f} Hz)")


def main() -> int:
    args = parse_args()
    joint_idx = JOINT_NAMES.index(args.joint)
    watch_name = args.watch_joint if args.watch_joint is not None else args.joint
    watch_idx = JOINT_NAMES.index(watch_name)
    servo_ids = [JOINT_TO_SERVO[name][0] for name in JOINT_NAMES]
    sid = servo_ids[joint_idx]
    watch_sid = servo_ids[watch_idx]
    assert args.mode == "sweep" or args.watch_joint is None, \
        "--watch-joint only makes sense in sweep mode"

    kp = list(SERVO_POSITION_KP)
    dz = list(SERVO_POSITION_DEADZONE)
    if args.kp is not None:
        kp[joint_idx] = args.kp
    if args.deadzone is not None:
        dz[joint_idx] = args.deadzone

    print(f"probe {args.joint} (servo {sid})  mode={args.mode}  "
          f"kp={kp[joint_idx]} deadzone={dz[joint_idx]} "
          f"speed={args.speed} accel={args.accel}")
    if args.mode == "hold":
        print("holding boot pose under torque — tap the arm for a ring-down")
    elif watch_idx != joint_idx:
        print(f"sweeping {args.joint}, recording {watch_name}")

    bus = ServoBus(args.port, servo_ids)
    bus.connect()
    try:
        boot_raw = bus.read_all()
        bus.set_position_kp(kp)
        bus.set_position_deadzone(dz)
        bus.enable_torque_all()
        bus.write_all(boot_raw, args.speed, args.accel)
        time.sleep(0.5)  # let the hold settle before recording

        if args.mode == "hold":
            rows = record_hold(bus, sid, int(boot_raw[joint_idx]), args.duration)
        else:
            rows = record_sweep(bus, joint_idx, watch_sid, watch_idx, boot_raw,
                                args.sweep_range, args.sweep_raw_per_s,
                                args.dwell_s, args.write_hz, args.duration,
                                args.speed, args.accel)
        assert rows, "no samples recorded"

        tag = args.joint if watch_idx == joint_idx else f"{args.joint}_watch_{watch_name}"
        out = REPO_ROOT / "rollouts" / f"probe_{tag}_{args.mode}_{int(time.time())}.csv"
        with open(out, "w") as f:
            f.write(f"# joint={args.joint} watch={watch_name} mode={args.mode} "
                    f"kp={kp[joint_idx]} deadzone={dz[joint_idx]} "
                    f"speed={args.speed} accel={args.accel} "
                    f"sweep_range={args.sweep_range} "
                    f"sweep_raw_per_s={args.sweep_raw_per_s} "
                    f"dwell_s={args.dwell_s}\n")
            f.write("t_s,target_raw,pos_raw,speed_raw\n")
            for r in rows:
                f.write(f"{r[0]:.4f},{r[1]},{r[2]},{r[3]}\n")
        print(f"saved {out}")
        summarize(rows)
    finally:
        # Kp/deadzone are EEPROM-backed; always leave the standard values.
        bus.set_position_kp(SERVO_POSITION_KP)
        bus.set_position_deadzone(SERVO_POSITION_DEADZONE)
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
