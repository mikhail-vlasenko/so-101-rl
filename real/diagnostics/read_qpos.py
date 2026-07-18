"""Read and print the SO-101 follower's current joint state (raw / norm / rad).

Torque stays off, so the arm can be posed by hand and queried. Useful for
discovering contact-relevant poses (e.g. near the table) to seed calibration.

Usage:
    python -m real.diagnostics.read_qpos                 # default port /dev/ttyACM0
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from real.twin.mapping import (
    JOINT_NAMES,
    load_joint_maps,
    raw_to_norm,
    raw_to_rad,
)
from real.twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
DEFAULT_CAL = REPO_ROOT / "real" / "follower_calibration.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    direction = np.ones(6, dtype=np.int8)  # follower: no inversions

    bus = ServoBus(args.port, jm.servo_ids())
    bus.connect()
    try:
        raw = bus.read_all()
    finally:
        bus.close()  # torque stays off

    norm = raw_to_norm(raw, jm.cal_min(), jm.cal_max(), direction)
    rad = raw_to_rad(raw, jm, direction)

    data.qpos[jm.qposadr()] = rad
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    ee = data.site_xpos[ee_id].copy()

    print(f"{'joint':<14} {'raw':>6} {'norm':>7} {'rad':>8} {'deg':>8}")
    print("-" * 46)
    for i, name in enumerate(JOINT_NAMES):
        print(f"{name:<14} {int(raw[i]):>6} {norm[i]:>7.3f} "
              f"{rad[i]:>8.3f} {np.degrees(rad[i]):>8.2f}")
    print(f"\nqpos (rad):  {np.round(rad, 4).tolist()}")
    print(f"gripperframe (base frame): {np.round(ee, 4).tolist()} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
