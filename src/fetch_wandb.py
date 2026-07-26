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
# Primary objective: maximize the stage-1 lift success rate reached within the
# run's budget — the from-scratch policy has to learn the grasp-and-lift skill at
# all before anything about *how fast* it lifts is meaningful. The metrics below
# form a ladder for the regime where success is still ~0: ever_grasped moves
# before success does, and mean_max_cube_height before that. mean_ep_length is
# reported last and is a diagnostic here: it is pinned at max_steps until the
# policy starts terminating episodes, and only becomes an objective once success
# is high (that was the previous refine loop's target).
METRIC = "rollout/lift/success_rate"             # PRIMARY, higher is better
LADDER_METRICS = (
    ("rollout/lift/ever_grasped", "higher=better, leads success"),
    ("rollout/lift/grasp_ratio", "higher=better, grasp stability"),
    ("rollout/lift/mean_max_cube_height", "higher=better, leads grasp"),
    ("rollout/lift/mean_ep_length", "diagnostic; pinned at max_steps until success>0"),
)
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
        print(f"Available columns containing 'lift', 'ep_length', or 'success':")
        for c in sorted(df.columns):
            if any(s in c.lower() for s in ("lift", "ep_length", "success", "height")):
                print(f"  {c}")
        sys.exit(1)

    for metric, label in ((METRIC, "PRIMARY, higher=better"),) + LADDER_METRICS:
        if metric not in df.columns:
            print(f"\n{metric}: not logged in this run.")
            continue
        metric_df = df[df[metric].notna()].copy()
        if "_timestamp" in metric_df.columns:
            metric_df = metric_df.sort_values("_timestamp")
            last_ts = metric_df["_timestamp"].iloc[-1]
            last_minute = metric_df[metric_df["_timestamp"] >= last_ts - 60]
        else:
            last_minute = metric_df.tail(10)
        avg = last_minute[metric].mean()
        print(f"\n{metric} ({label}) — last minute ({len(last_minute)} points):")
        print(f"  Average: {avg:.4f}")
        print(f"  Min:     {last_minute[metric].min():.4f}")
        print(f"  Max:     {last_minute[metric].max():.4f}")
        # A takeoff run can peak and then collapse; the last minute alone would
        # read that as "never learned".
        print(f"  Peak over run: {metric_df[metric].max():.4f}")
        if args.trajectory:
            print_per_minute_trajectory(metric_df, metric)


if __name__ == "__main__":
    main()
