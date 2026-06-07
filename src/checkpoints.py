"""Centralized checkpoint resolution shared by eval and the real-arm rollouts.

A model argument is resolved the same way everywhere:
    "latest" → newest-mtime .zip in <log_dir>/checkpoints/
    "best"   → <log_dir>/best_model.zip
    anything else → treated as an explicit path to a .zip
"""

import os


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Return the most recently created checkpoint file in checkpoint_dir."""
    zips = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not zips:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(
        (os.path.join(checkpoint_dir, f) for f in zips),
        key=os.path.getmtime,
    )


def resolve_model_path(model_arg: str, log_dir: str) -> str:
    if model_arg == "latest":
        return find_latest_checkpoint(os.path.join(log_dir, "checkpoints"))
    if model_arg == "best":
        return os.path.join(log_dir, "best_model.zip")
    return model_arg
