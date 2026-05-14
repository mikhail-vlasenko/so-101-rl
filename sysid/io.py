"""CSV I/O for sysid logs. Schema is shared between real and sim recorders so
analyze.py can load either with the same code path.

Columns: step, t (s), target_<joint>... (rad), pos_<joint>... (rad)
"""

import csv
from pathlib import Path

import numpy as np

from base_env import JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "sysid_logs"
OUT_DIR_REAL = OUT_DIR / "real"
OUT_DIR_SIM = OUT_DIR / "sim"
OUT_DIR_PLOTS = OUT_DIR / "plots"


def csv_header() -> list[str]:
    return (["step", "t_s"]
            + [f"target_{n}" for n in JOINT_NAMES]
            + [f"pos_{n}" for n in JOINT_NAMES])


def write_log(out_path: Path, target: np.ndarray, pos: np.ndarray, dt: float) -> None:
    assert target.shape == pos.shape and target.shape[1] == 6, \
        f"shape mismatch target={target.shape} pos={pos.shape}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        for i in range(len(target)):
            w.writerow([i, f"{i * dt:.4f}",
                        *(f"{v:.6f}" for v in target[i]),
                        *(f"{v:.6f}" for v in pos[i])])


def read_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t, target, pos) arrays. target/pos shape (T, 6)."""
    t_list: list[float] = []
    target_list: list[list[float]] = []
    pos_list: list[list[float]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            t_list.append(float(row["t_s"]))
            target_list.append([float(row[f"target_{n}"]) for n in JOINT_NAMES])
            pos_list.append([float(row[f"pos_{n}"]) for n in JOINT_NAMES])
    return (np.array(t_list, dtype=np.float64),
            np.array(target_list, dtype=np.float64),
            np.array(pos_list, dtype=np.float64))
