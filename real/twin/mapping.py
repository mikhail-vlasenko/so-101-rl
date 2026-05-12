"""Transparent rad <-> raw mapping for the SO-101 digital-twin tool.

Built from scratch on purpose — the existing `real/arm_bridge.py` is suspected
of having wrong direction signs and/or mismatched calibration semantics, and
this tool exists to debug that. Do not import from `real/arm_bridge.py`.

Mapping convention (linear, two-point):
    frac = (raw - cal_min) / (cal_max - cal_min)
    if direction == -1: frac = 1 - frac
    rad  = xml_low + frac * (xml_high - xml_low)

`calibration.json` stores software-defined raw min/max chosen so the midpoint
should land at ~2048 after the Feetech recenter procedure. Those values are
NOT measurements at mechanical limits — see the project memory.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

JOINT_TO_SERVO: dict[str, tuple[int, str]] = {
    "shoulder_pan":  (1, "base"),
    "shoulder_lift": (2, "shoulder"),
    "elbow_flex":    (3, "elbow"),
    "wrist_flex":    (4, "wrist_pitch"),
    "wrist_roll":    (5, "wrist_roll"),
    "gripper":       (6, "gripper"),
}


@dataclass
class JointMap:
    name: str
    servo_id: int
    cal_key: str
    cal_min: int
    cal_max: int
    xml_low: float
    xml_high: float
    qposadr: int

    @property
    def cal_mid(self) -> float:
        return 0.5 * (self.cal_min + self.cal_max)


@dataclass
class JointMaps:
    items: list[JointMap]

    def __post_init__(self) -> None:
        assert [jm.name for jm in self.items] == JOINT_NAMES

    def servo_ids(self) -> list[int]:
        return [jm.servo_id for jm in self.items]

    def qposadr(self) -> np.ndarray:
        return np.array([jm.qposadr for jm in self.items], dtype=np.int64)

    def cal_min(self) -> np.ndarray:
        return np.array([jm.cal_min for jm in self.items], dtype=np.float64)

    def cal_max(self) -> np.ndarray:
        return np.array([jm.cal_max for jm in self.items], dtype=np.float64)

    def xml_low(self) -> np.ndarray:
        return np.array([jm.xml_low for jm in self.items], dtype=np.float64)

    def xml_high(self) -> np.ndarray:
        return np.array([jm.xml_high for jm in self.items], dtype=np.float64)


def load_joint_maps(model: mujoco.MjModel, cal_path: Path) -> JointMaps:
    with open(cal_path) as f:
        cal = json.load(f)
    items: list[JointMap] = []
    for name in JOINT_NAMES:
        servo_id, cal_key = JOINT_TO_SERVO[name]
        if cal_key not in cal:
            raise KeyError(f"{cal_path} missing entry {cal_key!r} for joint {name!r}")
        entry = cal[cal_key]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name!r} not found in MuJoCo model")
        qposadr = int(model.jnt_qposadr[jid])
        xml_low, xml_high = (float(x) for x in model.jnt_range[jid])
        items.append(JointMap(
            name=name, servo_id=servo_id, cal_key=cal_key,
            cal_min=int(entry["min"]), cal_max=int(entry["max"]),
            xml_low=xml_low, xml_high=xml_high, qposadr=qposadr,
        ))
    return JointMaps(items=items)


def raw_to_norm(raw: np.ndarray, cal_min: np.ndarray, cal_max: np.ndarray,
                direction: np.ndarray) -> np.ndarray:
    frac = (raw.astype(np.float64) - cal_min) / (cal_max - cal_min)
    return np.where(direction == -1, 1.0 - frac, frac)


def raw_to_rad(raw: np.ndarray, jm: JointMaps, direction: np.ndarray) -> np.ndarray:
    frac = raw_to_norm(raw, jm.cal_min(), jm.cal_max(), direction)
    return jm.xml_low() + frac * (jm.xml_high() - jm.xml_low())


def rad_to_raw(rad: np.ndarray, jm: JointMaps, direction: np.ndarray) -> np.ndarray:
    xml_low, xml_high = jm.xml_low(), jm.xml_high()
    cal_min, cal_max = jm.cal_min(), jm.cal_max()
    frac = (rad - xml_low) / (xml_high - xml_low)
    frac = np.where(direction == -1, 1.0 - frac, frac)
    raw = cal_min + frac * (cal_max - cal_min)
    bounds_lo = np.minimum(cal_min, cal_max)
    bounds_hi = np.maximum(cal_min, cal_max)
    return np.clip(raw, bounds_lo, bounds_hi).round().astype(np.int64)
