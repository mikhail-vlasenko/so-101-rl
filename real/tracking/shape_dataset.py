"""Shared IO and SAM-mask cache for recorded sponge stereo datasets."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

from real.calib.extrinsics import pos_quat_to_mat


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LIFT_CONFIG_PATH = REPO_ROOT / "conf" / "env" / "lift.yaml"


def load_workspace_bounds():
    """Configured lift workspace xy bounds used by dense stereo."""
    with LIFT_CONFIG_PATH.open() as stream:
        task = yaml.safe_load(stream)["lift_env"]
    return (np.asarray(task["cube_low"], dtype=np.float64),
            np.asarray(task["cube_high"], dtype=np.float64))


class CausalMeanPosition:
    """Suppress position jitter without using future samples."""

    def __init__(self, window_s):
        self.window_s = float(window_s)
        if self.window_s <= 0.0:
            raise ValueError("causal mean window must be positive")
        self._times = []
        self._positions = []

    def update(self, t, position):
        t = float(t)
        self._times.append(t)
        self._positions.append(np.asarray(position, dtype=np.float64).copy())
        while len(self._times) > 1 and self._times[0] < t - self.window_s:
            self._times.pop(0)
            self._positions.pop(0)
        return np.mean(np.stack(self._positions), axis=0)

    def clear(self):
        """Discard samples across a tracking loss."""
        self._times.clear()
        self._positions.clear()


def load_dataset(dataset_dir: Path):
    with (dataset_dir / "meta.yaml").open() as stream:
        meta = yaml.safe_load(stream)
    with (dataset_dir / "index.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    assert records, f"empty dataset index in {dataset_dir}"
    return records, meta


def compute_masks(dataset_dir: Path, records, meta, prompt, sam2_model,
                  reprompt_after=30):
    """Compute or resume the per-camera SAM mask cache."""
    from real.tracking.sam_seg import MaskTracker, find_text_mask, load_sam3

    mask_dir = dataset_dir / "masks" / sam2_model
    todo = [(rec, name) for rec in records for name in meta["cameras"]
            if not (mask_dir / rec["frame"][name]).with_suffix(".png").exists()]
    total = len(records) * len(meta["cameras"])
    if not todo:
        print(f"using {total} cached SAM masks from {mask_dir}")
        return mask_dir
    cached = total - len(todo)
    print(f"computing {len(todo)} SAM masks ({cached} cached) into {mask_dir}; "
          f"loading SAM3 + SAM2 {sam2_model} on CUDA ...", flush=True)
    sam3 = load_sam3()
    tracker = MaskTracker(sam2_model)
    with tqdm(total=total, initial=cached, unit="mask", desc="SAM masks",
              dynamic_ncols=True) as progress:
        for name in meta["cameras"]:
            (mask_dir / "frames").mkdir(parents=True, exist_ok=True)
            progress.set_postfix(camera=name)
            primed = False
            empty_run = 0
            retry_in = 0
            for rec in records:
                out_path = (mask_dir / rec["frame"][name]).with_suffix(".png")
                if out_path.exists():
                    continue
                frame = cv2.imread(str(dataset_dir / rec["frame"][name]))
                assert frame is not None, rec["frame"][name]
                if primed and empty_run < reprompt_after:
                    mask = tracker.track(frame)
                elif retry_in > 0:
                    retry_in -= 1
                    mask = np.zeros(frame.shape[:2], dtype=bool)
                else:
                    found = find_text_mask(sam3, frame, prompt)
                    if found is None:
                        primed = False
                        retry_in = reprompt_after
                        mask = np.zeros(frame.shape[:2], dtype=bool)
                    else:
                        mask, score = found
                        tracker.prime(frame, mask)
                        primed = True
                        progress.write(f"  {name}: (re)prompted at k={rec['k']}, "
                                       f"score {score:.2f}")
                empty_run = 0 if mask.any() else empty_run + 1
                if not cv2.imwrite(str(out_path), mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"failed to write mask {out_path}")
                progress.update()
    return mask_dir


def load_mask(mask_dir: Path, rec, name):
    mask = cv2.imread(
        str((mask_dir / rec["frame"][name]).with_suffix(".png")),
        cv2.IMREAD_GRAYSCALE,
    )
    assert mask is not None, rec["frame"][name]
    return mask > 127


def gt_pose(rec):
    """Evaluation-only tag GT ``(center, rotation)`` or ``None``."""
    fused = rec["body"]["fused"]
    if fused is None:
        return None
    transform = pos_quat_to_mat(fused["pos"], fused["quat"])
    return transform[:3, 3], transform[:3, :3]
