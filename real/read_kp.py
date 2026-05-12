"""Read the position-loop Kp (Feetech SMS-STS addr 21) from every SO-101 servo.

Usage:
    python -m real.read_kp                 # default port /dev/ttyACM0
    python -m real.read_kp --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse

from .twin.servo_io import ADDR_POSITION_KP, ServoBus

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    args = p.parse_args()

    bus = ServoBus(args.port, list(range(1, 7)))
    bus.connect()
    try:
        print(f"{'id':>3}  {'joint':<14} {'Kp (addr 21)':>14}")
        print("-" * 36)
        for sid, name in zip(bus.servo_ids, JOINT_NAMES):
            kp, comm, err = bus.packet_handler.read1ByteTxRx(sid, ADDR_POSITION_KP)
            if comm != 0:
                raise RuntimeError(f"servo {sid} read comm error: {comm}")
            if err != 0:
                raise RuntimeError(f"servo {sid} read packet error: {err}")
            print(f"{sid:>3}  {name:<14} {kp:>14}")
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
