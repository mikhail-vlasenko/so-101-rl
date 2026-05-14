"""Fetch W&B metrics for a run, save to CSV, and print recent averages.

Usage:
    python fetch_wandb.py                       # most recent run, last-minute summary
    python fetch_wandb.py <run_id>              # specific run
    python fetch_wandb.py --trajectory          # also print per-minute trajectory
    python fetch_wandb.py <run_id> -t           # both
"""

import argparse
import os
import sys

import wandb
import pandas as pd


ENTITY = "mvlasenko"
PROJECT = "robot-arm"
METRIC = "rollout/lift/mean_max_cube_height"
CSV_DIR = "wandb"  # written here to keep the repo root clean (already gitignored)


def fetch_run(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")

    history = run.scan_history()
    rows = list(history)
    df = pd.DataFrame(rows)
    return run, df


def print_per_minute_trajectory(metric_df: pd.DataFrame, metric: str) -> None:
    """Print the first metric value observed in each elapsed-minute bucket
    since the run started — mirrors the awk one-liner against output.log."""
    if "_timestamp" not in metric_df.columns:
        print(f"\nNo _timestamp column; cannot bucket by minute.")
        return
    t0 = metric_df["_timestamp"].iloc[0]
    minute = ((metric_df["_timestamp"] - t0) // 60).astype(int)
    first_per_minute = metric_df.groupby(minute)[metric].first()
    print(f"\n{metric} — per-minute trajectory ({len(first_per_minute)} buckets):")
    for m, v in first_per_minute.items():
        print(f"  {m:>3d}min: {v:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id", nargs="?", default=None,
                   help="W&B run id (default: most recent run in the project)")
    p.add_argument("-t", "--trajectory", action="store_true",
                   help="Also print per-minute metric trajectory.")
    args = p.parse_args()

    api = wandb.Api()
    if args.run_id is None:
        runs = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=1)
        run_id = runs[0].id
        print(f"Using most recent run: {run_id}")
    else:
        run_id = args.run_id

    run_path = f"{ENTITY}/{PROJECT}/{run_id}"
    run, df = fetch_run(run_path)

    # Save to CSV under wandb/ so the repo root stays clean.
    os.makedirs(CSV_DIR, exist_ok=True)
    out_file = os.path.join(CSV_DIR, f"metrics_{run.id}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")

    if METRIC not in df.columns:
        print(f"\nMetric '{METRIC}' not found in run.")
        print(f"Available columns containing 'lift', 'cube', or 'height':")
        for c in sorted(df.columns):
            if any(s in c.lower() for s in ("lift", "cube", "height")):
                print(f"  {c}")
        sys.exit(1)

    # Filter to rows that have the metric logged
    metric_df = df[df[METRIC].notna()].copy()
    if "_timestamp" in metric_df.columns:
        metric_df = metric_df.sort_values("_timestamp")
        last_ts = metric_df["_timestamp"].iloc[-1]
        last_minute = metric_df[metric_df["_timestamp"] >= last_ts - 60]
    else:
        last_minute = metric_df.tail(10)

    avg = last_minute[METRIC].mean()
    print(f"\n{METRIC} — last minute ({len(last_minute)} points):")
    print(f"  Average: {avg:.4f}")
    print(f"  Min:     {last_minute[METRIC].min():.4f}")
    print(f"  Max:     {last_minute[METRIC].max():.4f}")

    if args.trajectory:
        print_per_minute_trajectory(metric_df, METRIC)


if __name__ == "__main__":
    main()
