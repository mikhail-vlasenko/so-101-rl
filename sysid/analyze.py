"""Compare real-arm and sim sysid logs: per-trajectory plots + RMSE report.

Reads paired CSVs from sysid_logs/real/<name>.csv and sysid_logs/sim/<name>.csv,
writes:
  - sysid_logs/plots/<name>.png   (6-joint overlay: target / real / sim)
  - sysid_logs/report.tsv         (per-(traj, joint) RMSE table + ALL aggregate)

Usage:
    python -m sysid.analyze
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base_env import JOINT_NAMES
from sysid.io import (
    OUT_DIR,
    OUT_DIR_PLOTS,
    OUT_DIR_REAL,
    OUT_DIR_SIM,
    REPO_ROOT,
    read_log,
)


def plot_pair(name: str, t_r: np.ndarray, target: np.ndarray, pos_r: np.ndarray,
              t_s: np.ndarray, pos_s: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(f"{name}: real (solid) vs sim (dashed)")
    for j, joint in enumerate(JOINT_NAMES):
        ax = axes[j // 3, j % 3]
        ax.plot(t_r, target[:, j], color="k", lw=0.7, alpha=0.4, label="target")
        ax.plot(t_r, pos_r[:, j], color="C0", lw=1.2, label="real")
        ax.plot(t_s, pos_s[:, j], color="C1", lw=1.2, ls="--", label="sim")
        ax.set_title(joint)
        ax.set_ylabel("rad")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="best")
    for ax in axes[1, :]:
        ax.set_xlabel("t [s]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def per_joint_metrics(target: np.ndarray, pos_r: np.ndarray,
                      pos_s: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for j, joint in enumerate(JOINT_NAMES):
        out[joint] = {
            "rmse_real_vs_sim":   float(np.sqrt(np.mean((pos_r[:, j] - pos_s[:, j]) ** 2))),
            "rmse_real_vs_target": float(np.sqrt(np.mean((pos_r[:, j] - target[:, j]) ** 2))),
            "rmse_sim_vs_target":  float(np.sqrt(np.mean((pos_s[:, j] - target[:, j]) ** 2))),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation; write the report.tsv only.")
    args = p.parse_args()

    if not OUT_DIR_REAL.exists():
        raise SystemExit(f"no real logs in {OUT_DIR_REAL}; run sysid.record_real first")
    if not OUT_DIR_SIM.exists():
        raise SystemExit(f"no sim logs in {OUT_DIR_SIM}; run sysid.replay_sim first")

    paired: list[str] = []
    for real_path in sorted(OUT_DIR_REAL.glob("*.csv")):
        if (OUT_DIR_SIM / real_path.name).exists():
            paired.append(real_path.stem)
        else:
            print(f"skip {real_path.stem}: no sim CSV")

    if not paired:
        raise SystemExit("no real/sim pairs to compare")

    rows: list[dict] = []
    per_joint_sq: dict[str, list[float]] = {j: [] for j in JOINT_NAMES}
    for name in paired:
        t_r, target_r, pos_r = read_log(OUT_DIR_REAL / f"{name}.csv")
        t_s, target_s, pos_s = read_log(OUT_DIR_SIM / f"{name}.csv")
        assert target_r.shape == target_s.shape, (
            f"{name}: target shape mismatch real={target_r.shape} sim={target_s.shape}"
        )
        assert np.allclose(target_r, target_s, atol=1e-4), (
            f"{name}: real and sim CSVs disagree on the target sequence"
        )
        if not args.no_plots:
            plot_pair(name, t_r, target_r, pos_r, t_s, pos_s, OUT_DIR_PLOTS / f"{name}.png")
        metrics = per_joint_metrics(target_r, pos_r, pos_s)
        for joint in JOINT_NAMES:
            rows.append({"traj": name, "joint": joint, **metrics[joint]})
            per_joint_sq[joint].append(metrics[joint]["rmse_real_vs_sim"] ** 2)

    for joint in JOINT_NAMES:
        rows.append({
            "traj": "ALL",
            "joint": joint,
            "rmse_real_vs_sim":  float(np.sqrt(np.mean(per_joint_sq[joint]))),
            "rmse_real_vs_target": float("nan"),
            "rmse_sim_vs_target":  float("nan"),
        })

    report = OUT_DIR / "report.tsv"
    fields = ["traj", "joint", "rmse_real_vs_sim",
              "rmse_real_vs_target", "rmse_sim_vs_target"]
    with report.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"wrote {report.relative_to(REPO_ROOT)} ({len(paired)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
