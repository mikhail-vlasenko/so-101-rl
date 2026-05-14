"""Two-stage curriculum lift training.

Stage 1: train at --warmup-scale for --warmup-min minutes (faster commands,
         easier credit assignment).
Stage 2: resume at --final-scale for --final-min minutes (the deployment
         target — slower per-step delta).

Wraps train.py — both stages produce their own W&B run. Stage 1's
final_model.zip is copied aside before stage 2 starts, since stage 2
overwrites it.

Usage:
    python scripts/curriculum_train.py
    python scripts/curriculum_train.py --warmup-min 30 --final-min 10
    python scripts/curriculum_train.py --warmup-scale 0.06 --final-scale 0.03
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = REPO_ROOT / "train.py"
STAGE1_CKPT = REPO_ROOT / "logs" / "ppo_lift" / "curriculum_stage1.zip"
FINAL_MODEL = REPO_ROOT / "logs" / "ppo_lift" / "final_model.zip"


def run_stage(label: str, overrides: list[str]) -> None:
    cmd = ["conda", "run", "-n", "mujoco_env", "python", str(TRAIN), "env=lift", *overrides]
    print(f"\n===== {label} =====")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup-scale", type=float, default=0.05)
    p.add_argument("--warmup-min", type=float, default=20)
    p.add_argument("--final-scale", type=float, default=0.035)
    p.add_argument("--final-min", type=float, default=5)
    args = p.parse_args()

    run_stage(
        f"Stage 1: action_scale={args.warmup_scale} for {args.warmup_min} min",
        [f"action_scale={args.warmup_scale}", f"train.time_limit_minutes={args.warmup_min}"],
    )

    STAGE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FINAL_MODEL, STAGE1_CKPT)
    print(f"Saved stage-1 checkpoint to {STAGE1_CKPT}")

    run_stage(
        f"Stage 2: action_scale={args.final_scale} for {args.final_min} min (resume)",
        [
            f"action_scale={args.final_scale}",
            f"train.time_limit_minutes={args.final_min}",
            f"resume={STAGE1_CKPT}",
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
