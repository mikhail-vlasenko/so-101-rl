"""DualSense control for the SO-101 digital twin.

A reader thread translates evdev events into a `GamepadState`; an integrator
thread (~50 Hz) maps stick/trigger/bumper deflections to per-joint angular
velocities and integrates them into `TwinState.targets_rad` while the twin is
in CONTROL mode. Create toggles MIRROR<->CONTROL; PS triggers e-stop.

Mapping (JOINT_NAMES order):
    LX  -> shoulder_pan       (left = negative)
    LY  -> shoulder_lift      (stick up = lift up)
    RY  -> elbow_flex         (stick down = elbow up)
    R1/R2 -> wrist_flex       (R2 up, R1 down, R2 analog)
    RX  -> wrist_roll
    L1/L2 -> gripper          (L1 open, L2 close, L2 analog)

Run as a daemon thread; if no DualSense is present the worker logs and exits
without affecting the rest of the twin.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from evdev import InputDevice, ecodes, list_devices

from .gui import CONTROL, TwinState
from .mapping import JointMaps

# DualSense via hid-playstation reports sticks as unsigned bytes 0..255
# centered at ~128, NOT signed shorts. Normalize as (v - center) / half.
STICK_CENTER = 128.0
STICK_HALF = 127.0
TRIG_FULL = 255.0      # ABS_Z/RZ range 0..255
STICK_DEADZONE = 0.12
TRIG_DEADZONE = 0.05
INTEGRATOR_HZ = 50.0

# Per-joint max angular speed (rad/s) at full stick/trigger deflection.
# Order matches JOINT_NAMES: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
MAX_SPEED_RAD_S = np.array([0.9, 0.75, 0.9, 1.0, 1.75, 1.25], dtype=np.float64)

PAN, LIFT, ELBOW, WRIST_FLEX, WRIST_ROLL, GRIPPER = 0, 1, 2, 3, 4, 5


def find_dualsense() -> InputDevice | None:
    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name.lower()
        if "dualsense" in name and "touchpad" not in name and "motion" not in name:
            return dev
    return None


def _deadzone(v: float, dz: float) -> float:
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


@dataclass
class GamepadState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    l2: float = 0.0
    r2: float = 0.0
    l1: bool = False
    r1: bool = False
    mode_toggle_pending: bool = False
    estop_pending: bool = False


def _reader_loop(dev: InputDevice, gp: GamepadState, lock: threading.Lock,
                 stop_fn: Callable[[], bool]) -> None:
    for event in dev.read_loop():
        if stop_fn():
            return
        if event.type == ecodes.EV_KEY:
            if event.code == ecodes.BTN_TL:
                with lock:
                    gp.l1 = bool(event.value)
            elif event.code == ecodes.BTN_TR:
                with lock:
                    gp.r1 = bool(event.value)
            elif event.code == ecodes.BTN_SELECT and event.value == 1:
                with lock:
                    gp.mode_toggle_pending = True
            elif event.code == ecodes.BTN_MODE and event.value == 1:
                with lock:
                    gp.estop_pending = True
        elif event.type == ecodes.EV_ABS:
            v = event.value
            if event.code == ecodes.ABS_X:
                with lock:
                    gp.lx = _deadzone((v - STICK_CENTER) / STICK_HALF, STICK_DEADZONE)
            elif event.code == ecodes.ABS_Y:
                with lock:
                    gp.ly = _deadzone((v - STICK_CENTER) / STICK_HALF, STICK_DEADZONE)
            elif event.code == ecodes.ABS_RX:
                with lock:
                    gp.rx = _deadzone((v - STICK_CENTER) / STICK_HALF, STICK_DEADZONE)
            elif event.code == ecodes.ABS_RY:
                with lock:
                    gp.ry = _deadzone((v - STICK_CENTER) / STICK_HALF, STICK_DEADZONE)
            elif event.code == ecodes.ABS_Z:
                with lock:
                    gp.l2 = _deadzone(max(0.0, v / TRIG_FULL), TRIG_DEADZONE)
            elif event.code == ecodes.ABS_RZ:
                with lock:
                    gp.r2 = _deadzone(max(0.0, v / TRIG_FULL), TRIG_DEADZONE)


def gamepad_worker(state: TwinState, jm: JointMaps,
                   on_mode_toggle: Callable[[], None],
                   on_estop: Callable[[], None]) -> None:
    """Run the gamepad reader + integrator. Daemon-thread entry point."""
    dev = find_dualsense()
    if dev is None:
        print("[gamepad] no DualSense found; controller input disabled")
        return
    print(f"[gamepad] connected: {dev.name} ({dev.path})")

    gp = GamepadState()
    gp_lock = threading.Lock()
    reader = threading.Thread(
        target=_reader_loop, args=(dev, gp, gp_lock, lambda: state.stop),
        name="gamepad-reader", daemon=True,
    )
    reader.start()

    xml_low = jm.xml_low()
    xml_high = jm.xml_high()
    period = 1.0 / INTEGRATOR_HZ
    next_t = time.monotonic()
    last_t = time.monotonic()

    while not state.stop:
        now = time.monotonic()
        dt = now - last_t
        last_t = now

        with gp_lock:
            lx, ly, rx, ry = gp.lx, gp.ly, gp.rx, gp.ry
            l2, r2 = gp.l2, gp.r2
            l1, r1 = gp.l1, gp.r1
            toggle = gp.mode_toggle_pending
            estop = gp.estop_pending
            gp.mode_toggle_pending = False
            gp.estop_pending = False

        if estop:
            on_estop()
        elif toggle:
            on_mode_toggle()

        with state.lock:
            mode = state.mode
            tgt = state.targets_rad.copy()

        if mode == CONTROL:
            vel = np.zeros(6, dtype=np.float64)
            vel[PAN]        = -lx
            vel[LIFT]       = -ly  # stick up (ly<0) -> lift up
            vel[ELBOW]      = ry
            vel[WRIST_FLEX] = r2 - (1.0 if r1 else 0.0)
            vel[WRIST_ROLL] = rx
            vel[GRIPPER]    = (1.0 if l1 else 0.0) - l2
            tgt = tgt + vel * MAX_SPEED_RAD_S * dt
            tgt = np.clip(tgt, xml_low, xml_high)
            with state.lock:
                if state.mode == CONTROL:
                    state.targets_rad = tgt

        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()
