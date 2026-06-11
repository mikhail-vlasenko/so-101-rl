"""Send a sysid trajectory to the real SO-101 arm and log measured positions.

For each trajectory the script:
  1. Reads the current pose and verifies it is inside the MuJoCo joint range.
  2. Smoothly ramps from current pose to traj[0] (homing, not logged).
  3. Settles for 0.5 s at traj[0] so transients die.
  4. Streams traj[i] at the sysid control rate, logging measured qpos per step.

Output CSV schema matches sysid.replay_sim so analyze.py can compare directly.

Safety:
  - --execute is OFF by default (dry-run reads positions only).
  - Per-step raw-delta clamp derived from action_scale (src.units).
  - Trajectories are validated against XML joint limits before connecting.
  - Ctrl-C triggers a clean torque-off via ServoBus.close().

Usage:
    python -m sysid.record_real --traj sweep_shoulder_pan
    python -m sysid.record_real --all --execute
"""

import argparse
import signal
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from real.twin.constants import (
    INTERP_HZ,
    SERVO_ACCEL,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from real.twin.control import clamp_raw_delta, stream_sub_targets
from real.twin.mapping import JointMaps, load_joint_maps, rad_to_raw, raw_to_rad
from real.twin.servo_io import ServoBus
from src.units import max_raw_delta_per_step
from sysid.io import OUT_DIR_REAL, REPO_ROOT, write_log
from sysid.trajectories import SYSID_DT, SYSID_HZ, TRAJECTORIES

DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
DEFAULT_CAL = REPO_ROOT.parent / "feetech-servo-sdk" / "calibration.json"
CONFIG_YAML = REPO_ROOT / "conf" / "config.yaml"
HOMING_S = 3.0
# Long enough for the servo's slow accel ramp to close the homing lag and for
# stiction to settle — the stretch_* trajectories start gravity-loaded, so an
# unsettled start would leak the homing transient into the first logged ticks.
SETTLE_S = 1.0

# Sub-target interpolation across each sysid tick, mirroring ArmLoop: the
# servo profile chases a fresh waypoint every few ms instead of one coarse
# 15 Hz step, so recorded dynamics match the deployment command shaping.
N_INTERP = max(1, int(round(INTERP_HZ / SYSID_HZ)))
SUB_DT = SYSID_DT / N_INTERP


def smooth_ramp(start: np.ndarray, end: np.ndarray, n_steps: int) -> np.ndarray:
    u = (1 - np.cos(np.linspace(0.0, np.pi, n_steps + 1)[1:])) / 2
    return start[None, :] + u[:, None] * (end - start)[None, :]


def stream(bus: ServoBus, targets: np.ndarray, jm: JointMaps, direction: np.ndarray,
           prev_raw: np.ndarray, execute: bool, max_raw_delta: int,
           log_pos: list[np.ndarray] | None) -> np.ndarray:
    """Send each target row at SYSID_DT period, optionally recording measured pos.

    Each tick is streamed as N_INTERP interpolated sub-targets (same shaping as
    ArmLoop.tick); stream_sub_targets paces the full SYSID_DT internally.
    Returns the last raw target written (for chaining)."""
    def write(raw: np.ndarray) -> None:
        if execute:
            bus.write_all(raw, SERVO_SPEED, SERVO_ACCEL)

    for i in range(len(targets)):
        target_raw = rad_to_raw(targets[i], jm, direction)
        target_raw = clamp_raw_delta(prev_raw, target_raw, max_raw_delta)
        stream_sub_targets(prev_raw, target_raw, N_INTERP, SUB_DT, write)
        prev_raw = target_raw

        raw = bus.read_all()
        if log_pos is not None:
            log_pos.append(raw_to_rad(raw, jm, direction))
    return prev_raw


def run_traj(bus: ServoBus, jm: JointMaps, direction: np.ndarray,
             name: str, traj: np.ndarray, execute: bool, prev_raw: np.ndarray,
             max_raw_delta: int) -> np.ndarray:
    raw_now = bus.read_all()
    qpos_now = raw_to_rad(raw_now, jm, direction)
    n_homing = int(round(HOMING_S * SYSID_HZ))
    homing = smooth_ramp(qpos_now, traj[0], n_homing)
    prev_raw = stream(bus, homing, jm, direction, prev_raw, execute, max_raw_delta,
                      log_pos=None)

    n_settle = max(1, int(round(SETTLE_S * SYSID_HZ)))
    settle = np.tile(traj[0], (n_settle, 1))
    prev_raw = stream(bus, settle, jm, direction, prev_raw, execute, max_raw_delta,
                      log_pos=None)

    log_pos: list[np.ndarray] = []
    prev_raw = stream(bus, traj, jm, direction, prev_raw, execute, max_raw_delta,
                      log_pos=log_pos)
    pos = np.stack(log_pos)
    out = OUT_DIR_REAL / f"{name}.csv"
    write_log(out, target=traj, pos=pos, dt=SYSID_DT)
    print(f"real {name}: {len(traj)} steps → {out.relative_to(REPO_ROOT)}")
    return prev_raw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default=None,
                   help="Name of trajectory to record; omit if --all is set.")
    p.add_argument("--all", action="store_true",
                   help="Record every trajectory back-to-back.")
    p.add_argument("--execute", action="store_true",
                   help="Actually send servo commands. Default: dry-run (read-only).")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    args = p.parse_args()

    if bool(args.traj) == bool(args.all):
        raise SystemExit("specify exactly one of --traj <name> or --all")

    if args.all:
        names = sorted(TRAJECTORIES.keys())
    else:
        if args.traj not in TRAJECTORIES:
            raise SystemExit(
                f"unknown trajectory {args.traj!r}; "
                f"available: {sorted(TRAJECTORIES)}"
            )
        names = [args.traj]

    model = mujoco.MjModel.from_xml_path(args.xml)
    jm = load_joint_maps(model, Path(args.cal))
    # Same per-tick raw clamp the policy rollouts use, derived from the
    # configured action_scale.
    max_raw_delta = max_raw_delta_per_step(float(OmegaConf.load(CONFIG_YAML)["action_scale"]))
    direction = np.ones(len(jm.items), dtype=np.int8)
    xml_low, xml_high = jm.xml_low(), jm.xml_high()
    for name in names:
        t = TRAJECTORIES[name]
        if not (np.all(t >= xml_low - 1e-6) and np.all(t <= xml_high + 1e-6)):
            raise SystemExit(f"trajectory {name!r} exceeds joint limits")

    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()

    stopped = {"flag": False}
    def stop(_sig, _frame):
        stopped["flag"] = True
    signal.signal(signal.SIGINT, stop)

    try:
        raw0 = bus.read_all()
        qpos0 = raw_to_rad(raw0, jm, direction)
        slack = 0.05
        if not (np.all(qpos0 >= xml_low - slack) and np.all(qpos0 <= xml_high + slack)):
            raise SystemExit(
                f"ABORT: initial qpos outside joint range: {qpos0.tolist()}"
            )
        print(f"start qpos={qpos0.round(3).tolist()}  execute={args.execute}")

        if args.execute:
            bus.set_position_kp(SERVO_POSITION_KP)
            bus.enable_torque_all()

        prev_raw = raw0.copy()
        for name in names:
            if stopped["flag"]:
                print("stopped by user")
                break
            prev_raw = run_traj(bus, jm, direction, name,
                                TRAJECTORIES[name], args.execute, prev_raw,
                                max_raw_delta)
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
