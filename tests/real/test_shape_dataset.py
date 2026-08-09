"""Offline shape-dataset mask caching and progress contracts."""

from pathlib import Path

import cv2
import numpy as np

from real.tracking import sam_seg, shape_dataset


class _Progress:
    instances = []

    def __init__(self, total, initial, unit, desc, dynamic_ncols):
        self.total = total
        self.initial = initial
        self.updated = 0
        self.messages = []
        _Progress.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_postfix(self, camera):
        pass

    def write(self, message):
        self.messages.append(message)

    def update(self):
        self.updated += 1


class _Tracker:
    def __init__(self, model):
        self.mask = None

    def prime(self, frame, mask):
        self.mask = mask

    def track(self, frame):
        return self.mask


def _dataset(tmp_path: Path):
    records = []
    cameras = ["main", "aux"]
    frames = tmp_path / "frames"
    frames.mkdir()
    for k in (1, 2):
        paths = {}
        for camera in cameras:
            relative = f"frames/{camera}_{k:06d}.jpg"
            assert cv2.imwrite(str(tmp_path / relative),
                               np.full((8, 10, 3), k, dtype=np.uint8))
            paths[camera] = relative
        records.append({"k": k, "frame": paths})
    return records, {"cameras": cameras}


def test_compute_masks_reports_progress_and_reuses_cache(tmp_path, monkeypatch,
                                                         capsys):
    records, meta = _dataset(tmp_path)
    mask = np.ones((8, 10), dtype=bool)
    monkeypatch.setattr(sam_seg, "load_sam3", lambda: object())
    monkeypatch.setattr(sam_seg, "find_text_mask",
                        lambda model, frame, prompt: (mask, 0.9))
    monkeypatch.setattr(sam_seg, "MaskTracker", _Tracker)
    monkeypatch.setattr(shape_dataset, "tqdm", _Progress)
    _Progress.instances.clear()

    mask_dir = shape_dataset.compute_masks(
        tmp_path, records, meta, "sponge", "tiny")

    progress = _Progress.instances[0]
    assert progress.total == 4
    assert progress.initial == 0
    assert progress.updated == 4
    assert len(list(mask_dir.glob("frames/*.png"))) == 4

    def fail_load():
        raise AssertionError("loaded SAM despite a complete mask cache")

    monkeypatch.setattr(sam_seg, "load_sam3", fail_load)
    assert shape_dataset.compute_masks(
        tmp_path, records, meta, "sponge", "tiny") == mask_dir
    assert "using 4 cached SAM masks" in capsys.readouterr().out
