"""SO-101 digital-twin tool — entry point.

Mirrors the real arm into MuJoCo's passive viewer and lets the user verify the
servo-raw <-> MuJoCo-rad mapping interactively, with a DearPyGui side panel.

Usage:
    python -m real.twin.digital_twin --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .constants import MAX_RAW_DELTA_PER_STEP, SERVO_ACCEL, SERVO_SPEED
from .gamepad import gamepad_worker
from .gui import CONTROL, MIRROR, TwinState, run as run_gui
from .mapping import JointMaps, load_joint_maps, rad_to_raw, raw_to_norm
from .servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
DEFAULT_CAL = REPO_ROOT.parent / "feetech-servo-sdk" / "calibration.json"

READ_HZ = 50.0
RENDER_HZ = 60.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    return p.parse_args()


def print_mapping_header(jm: JointMaps) -> None:
    print(f"\n{'joint':<14} {'servo_id':>8} {'cal_key':<14} "
          f"{'cal_min':>8} {'cal_max':>8} {'cal_mid':>8} "
          f"{'xml_low':>9} {'xml_high':>9}")
    print("-" * 88)
    for j in jm.items:
        flag = "" if abs(j.cal_mid - 2048) < 20 else "  <- midpoint != 2048"
        print(f"{j.name:<14} {j.servo_id:>8} {j.cal_key:<14} "
              f"{j.cal_min:>8} {j.cal_max:>8} {j.cal_mid:>8.0f} "
              f"{j.xml_low:>+9.3f} {j.xml_high:>+9.3f}{flag}")
    print()


def servo_worker(state: TwinState, bus: ServoBus, jm: JointMaps) -> None:
    period = 1.0 / READ_HZ
    next_t = time.monotonic()
    last_log_t = time.monotonic()
    reads_since_log = 0
    while not state.stop:
        with state.bus_lock:
            raw = bus.read_all()
        with state.lock:
            state.latest_raw_read = raw
            mode = state.mode
            targets_rad = state.targets_rad.copy()
            prev = state.prev_raw_target.copy()

        if mode == CONTROL:
            with state.lock:
                direction = state.direction.copy()
            target_raw = rad_to_raw(targets_rad, jm, direction)
            delta = np.clip(target_raw - prev, -MAX_RAW_DELTA_PER_STEP, MAX_RAW_DELTA_PER_STEP)
            target_raw = (prev + delta).astype(np.int64)
            with state.bus_lock:
                bus.write_all(target_raw, SERVO_SPEED, SERVO_ACCEL)
            with state.lock:
                state.prev_raw_target = target_raw

        reads_since_log += 1
        now = time.monotonic()
        if now - last_log_t >= 0.5:
            with state.lock:
                state.read_hz = reads_since_log / (now - last_log_t)
            last_log_t = now
            reads_since_log = 0

        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()


def make_enter_control(state: TwinState, bus: ServoBus, jm: JointMaps):
    def fn() -> None:
        with state.lock:
            raw = state.latest_raw_read.copy()
            direction = state.direction.copy()
        norm = raw_to_norm(raw, jm.cal_min(), jm.cal_max(), direction)
        current_rad = jm.xml_low() + norm * (jm.xml_high() - jm.xml_low())
        with state.lock:
            state.targets_rad = current_rad
            state.prev_raw_target = raw
        with state.bus_lock:
            bus.enable_torque_all()
        with state.lock:
            state.mode = CONTROL
    return fn


def make_enter_mirror(state: TwinState, bus: ServoBus):
    def fn() -> None:
        with state.lock:
            state.mode = MIRROR
        with state.bus_lock:
            bus.disable_torque_all()
    return fn


def make_estop(state: TwinState, bus: ServoBus):
    def fn() -> None:
        with state.lock:
            state.mode = MIRROR
            state.stop = True
        with state.bus_lock:
            bus.disable_torque_all()
    return fn


def print_exit_summary(state: TwinState, jm: JointMaps) -> None:
    with state.lock:
        direction = state.direction.copy()
        raw = state.latest_raw_read.copy()
    norm = raw_to_norm(raw, jm.cal_min(), jm.cal_max(), direction)
    print("\n=== exit summary ===")
    print(f"direction (in JOINT_NAMES order): {direction.tolist()}")
    print(f"last raw read:                    {raw.tolist()}")
    print(f"last norm:                        {[f'{x:+.3f}' for x in norm]}")
    print()


def main() -> int:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    print_mapping_header(jm)

    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()

    state = TwinState(n=len(jm.items))
    qposadr = jm.qposadr()

    # Worker thread populates state.latest_raw_read; do one synchronous read
    # first so the viewer opens at the right pose.
    bus.disable_torque_all()
    initial_raw = bus.read_all()
    state.latest_raw_read = initial_raw
    state.prev_raw_target = initial_raw.copy()

    def sigint_handler(_signum, _frame) -> None:
        state.stop = True
    signal.signal(signal.SIGINT, sigint_handler)

    enter_control = make_enter_control(state, bus, jm)
    enter_mirror = make_enter_mirror(state, bus)
    estop = make_estop(state, bus)

    def gamepad_mode_toggle() -> None:
        with state.lock:
            cur = state.mode
        if cur == CONTROL:
            enter_mirror()
        else:
            enter_control()

    def viewer_loop() -> None:
        period = 1.0 / RENDER_HZ
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not state.stop:
                with state.lock:
                    raw = state.latest_raw_read.copy()
                    direction = state.direction.copy()
                norm = raw_to_norm(raw, jm.cal_min(), jm.cal_max(), direction)
                rad = jm.xml_low() + norm * (jm.xml_high() - jm.xml_low())
                data.qpos[qposadr] = rad
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(period)
        state.stop = True

    worker = threading.Thread(target=servo_worker, args=(state, bus, jm),
                              name="servo-worker", daemon=True)
    viewer_thread = threading.Thread(target=viewer_loop, name="mj-viewer", daemon=True)
    gamepad_thread = threading.Thread(
        target=gamepad_worker, args=(state, jm, gamepad_mode_toggle, estop),
        name="gamepad", daemon=True,
    )

    worker.start()
    viewer_thread.start()
    gamepad_thread.start()

    try:
        # Tk MUST run on the main thread.
        run_gui(state, jm, enter_control, enter_mirror, estop)
    finally:
        state.stop = True
        worker.join(timeout=1.0)
        viewer_thread.join(timeout=1.5)
        with state.bus_lock:
            bus.close()
        print_exit_summary(state, jm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
