"""Isolated persistent Fast-FoundationStereo inference worker.

This file is launched in the dedicated ``fast_foundation_stereo`` conda
environment. It loads one official serialized model, processes every JSON job
with that resident model, and writes left disparity plus a flipped/swapped
right-to-left disparity for deterministic consistency filtering. It imports no
MuJoCo project dependencies; the parent evaluator owns geometry and metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def _tensor(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.as_tensor(rgb, device="cuda").float()[None].permute(0, 3, 1, 2)


def _forward(model, InputPadder, left_bgr, right_bgr, valid_iters):
    left = _tensor(left_bgr)
    right = _tensor(right_bgr)
    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.float16):
        disparity = model.forward(
            left, right, iters=valid_iters, test_mode=True,
            optimize_build_volume="pytorch1")
    disparity = padder.unpad(disparity.float())
    return disparity.cpu().numpy().reshape(left_bgr.shape[:2]).clip(0.0, None)


def main():
    parser = argparse.ArgumentParser(description="Persistent Fast-FoundationStereo worker")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-iters", type=int, required=True)
    parser.add_argument("--max-disp", type=int, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from core.utils.utils import InputPadder

    checkpoint = repo / "weights" / args.checkpoint / "model_best_bp2_serialize.pth"
    if not checkpoint.exists():
        raise RuntimeError(f"missing official checkpoint {checkpoint}")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line]
    if not jobs:
        return
    for index, job in enumerate(jobs, 1):
        left = cv2.imread(job["left"])
        right = cv2.imread(job["right"])
        if left is None or right is None:
            raise RuntimeError(f"failed to read job images: {job}")
        if left.shape != right.shape:
            raise RuntimeError(f"stereo image shapes differ: {left.shape} vs {right.shape}")
        torch.cuda.synchronize()
        started = time.perf_counter()
        disparity = _forward(model, InputPadder, left, right, args.valid_iters)
        reverse_flipped = _forward(
            model, InputPadder,
            np.ascontiguousarray(right[:, ::-1]),
            np.ascontiguousarray(left[:, ::-1]),
            args.valid_iters)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0
        right_disparity = -np.ascontiguousarray(reverse_flipped[:, ::-1])
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            disparity=disparity.astype(np.float16),
            right_disparity=right_disparity.astype(np.float16),
            inference_ms=np.float32(inference_ms),
        )
        if index % 25 == 0 or index == len(jobs):
            print(f"Fast-FoundationStereo {index}/{len(jobs)}  "
                  f"last={inference_ms:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
