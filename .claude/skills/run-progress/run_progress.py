#!/usr/bin/env python3
"""Compact status report for a SB3 training run log.

Usage:
    python run_progress.py [LOG_PATH]

If LOG_PATH is omitted, picks the newest /tmp/train_*.log.
"""

import datetime as dt
import glob
import os
import re
import sys


KEYS = [
    # display name, regex key (after "rollout/" or "time/" or "train/" prefix)
    ("completion", r"rollout/completion_rate"),
    ("ring", r"rollout/ring_contact_ratio"),
    ("floor", r"rollout/floor_contact_ratio"),
    ("cube_h", r"rollout/mean_max_cube_height"),
    ("return", r"rollout/mean_return"),
    ("xy_prog", r"rollout/mean_xy_progress"),
    ("xy_regr", r"rollout/mean_xy_regress"),
    ("ep", r"rollout/episodes"),
    ("ep_rew", r"rollout/ep_rew_mean"),
    ("steps", r"time/total_timesteps"),
    ("elapsed_s", r"time/time_elapsed"),
    ("fps", r"time/fps"),
    ("ev", r"train/explained_variance"),
    ("std", r"train/std"),
    ("lr", r"train/learning_rate"),
]


def pick_log(arg: str | None) -> str:
    if arg:
        return arg
    candidates = sorted(glob.glob("/tmp/train_*.log"), key=os.path.getmtime, reverse=True)
    if not candidates:
        sys.exit("No /tmp/train_*.log files found and no path given")
    return candidates[0]


def parse_latest_block(text: str) -> dict[str, float]:
    """Return the last row seen for each metric from SB3's rollout tables.

    SB3's printed format strips the section prefix on row keys (so `rollout/completion_rate`
    in the logger appears as just `completion_rate` in the printed table). We track the
    surrounding section header (`| rollout/  |`, `| time/  |`, etc.) and rebuild the full
    key, so values from rollout/ vs eval/ don't shadow each other.
    """
    # Section header has the form `| rollout/   |   |` (two columns, value empty).
    section_re = re.compile(r"^\|\s+([a-z_]+)/\s+\|\s*\|$")
    row_re = re.compile(r"^\|\s+([\w]+)\s+\|\s+([-+0-9.eE]+)\s+\|$", re.MULTILINE)
    prefixed_row_re = re.compile(r"^\|\s+([\w]+/[\w]+)\s+\|\s+([-+0-9.eE]+)\s+\|$", re.MULTILINE)

    # Walk the file linearly so we know which section each row belongs to.
    section = ""
    latest: dict[str, float] = {}
    for line in text.splitlines():
        sec_m = section_re.match(line)
        if sec_m:
            section = sec_m.group(1) + "/"
            continue
        # Already-prefixed rows (e.g., `pickplace/completion_rate`) — keep verbatim.
        pre_m = prefixed_row_re.match(line)
        if pre_m:
            try:
                latest[pre_m.group(1)] = float(pre_m.group(2))
            except ValueError:
                pass
            continue
        row_m = row_re.match(line)
        if row_m:
            key, val = row_m.group(1), row_m.group(2)
            try:
                latest[section + key] = float(val)
            except ValueError:
                pass
    return latest


def detect_status(text: str) -> tuple[str, str | None]:
    if "Time limit reached" in text:
        return "DONE (time-limit)", None
    if "Model saved to" in text:
        return "DONE", None
    for marker in ("Traceback", "Killed", "OOM", "MemoryError"):
        if marker in text:
            tail = "\n".join(text.splitlines()[-15:])
            return "FAILED", tail
    if re.search(r"\bError\b", text):
        tail = "\n".join(text.splitlines()[-15:])
        return "FAILED?", tail
    return "RUNNING", None


def fmt(n: float, kind: str) -> str:
    if kind == "int":
        return f"{int(n):>5d}"
    if kind == "steps":
        return f"{n / 1e6:5.2f}M"
    if kind == "elapsed":
        return f"{n / 60:5.1f}m"
    if kind == "fps":
        return f"{int(n):>5d}"
    if kind == "lr":
        return f"{n:.1e}"
    return f"{n:.3f}"


def main() -> None:
    path = pick_log(sys.argv[1] if len(sys.argv) > 1 else None)
    with open(path) as f:
        text = f.read()
    metrics = parse_latest_block(text)
    status, err_tail = detect_status(text)
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")

    print(f"[ {path} @ {mtime} ]")

    elapsed = metrics.get("time/time_elapsed", 0.0)
    steps = metrics.get("time/total_timesteps", 0.0)
    fps = metrics.get("time/fps", 0.0)
    ep = metrics.get("rollout/episodes", 0.0)
    print(f"elapsed={fmt(elapsed, 'elapsed')}  steps={fmt(steps, 'steps')}  fps={fmt(fps, 'fps')}  ep={int(ep):>5d}")

    comp = metrics.get("rollout/completion_rate")
    ring = metrics.get("rollout/ring_contact_ratio")
    floor = metrics.get("rollout/floor_contact_ratio")
    cube_h = metrics.get("rollout/mean_max_cube_height")
    if comp is not None:
        print(f"completion={comp:.3f}  ring={ring:.3f}  floor={floor:.3f}  cube_h={cube_h:.3f}")

    rew = metrics.get("rollout/mean_return")
    xyp = metrics.get("rollout/mean_xy_progress")
    xyr = metrics.get("rollout/mean_xy_regress")
    if rew is not None:
        print(f"return={rew:.2f}  xy_prog={xyp:.2f}  xy_regr={xyr:.2f}")

    ev = metrics.get("train/explained_variance")
    std = metrics.get("train/std")
    lr = metrics.get("train/learning_rate")
    if ev is not None:
        print(f"ev={ev:.2f}  std={std:.3f}  lr={fmt(lr, 'lr')}")

    print(f"status: {status}")
    if err_tail:
        print("--- last 15 lines ---")
        print(err_tail)


if __name__ == "__main__":
    main()
