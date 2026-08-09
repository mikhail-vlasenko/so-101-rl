"""Evaluate official Fast-FoundationStereo after the Stage 2 SGBM gate fails.

This parent runs in ``mujoco_env`` and owns the exact same cached rectification,
SAM masks, Q reprojection, base-frame filtering and tag-GT metrics as the SGBM
evaluation. Neural disparity runs in a persistent subprocess inside the pinned
``fast_foundation_stereo`` conda environment; each candidate loads once and
writes resumable disparity files. A flipped/swapped second forward pass supplies
right disparity because the official model exposes no confidence map.

Run after ``real.tracking.eval_dense_stereo`` has prepared the dataset:
    conda run --no-capture-output -n mujoco_env python -m \
        real.tracking.eval_fast_foundation --dataset datasets/sponge_<stamp>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from real.tracking.dense_stereo import (
    CALIBRATION_PATH,
    CloudResult,
    FastFoundationCandidate,
    StereoPreprocessor,
    T_base_rectified_main,
    disparity_range_from_depths,
    disparity_to_cloud,
    fast_foundation_backend_key,
    left_right_validity,
    load_config,
    voxel_downsample,
)
from real.tracking.eval_dense_stereo import (
    _anchor_transform,
    _cache_image_path,
    _cache_root,
    _development_indices,
    _flatten,
    _load_prepared,
    _selection_score,
    _write_yaml,
    evaluate_results,
    in_workspace_windows,
    rectified_workspace_depths,
    relabel_static,
    split_windows,
    static_windows,
    validate_recorded_camera_placement,
    visible_windows,
)
from real.tracking.eval_estimator import load_dataset
from real.tracking.record_shapes import load_workspace_bounds


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKER_PATH = REPO_ROOT / "real" / "tracking" / "fast_foundation_worker.py"
DEFAULT_FAST_ROOT = Path(
    "/home/mikhail/.cache/mujoco_training/Fast-FoundationStereo")
FAST_ENV = "fast_foundation_stereo"


def max_disparity_for_range(minimum: int, count: int, multiple: int) -> int:
    high = minimum + count
    return int(math.ceil(high / multiple)) * multiple


def _backend_root(root: Path, config, candidate: FastFoundationCandidate,
                  max_disparity: int) -> Path:
    key = fast_foundation_backend_key(config, candidate, max_disparity)
    return root / "backends" / f"fast_foundation_{key}"


def disparity_path(backend_root: Path, source: str, record: dict) -> Path:
    return backend_root / "disparities" / source / f"{record['k']:06d}.npz"


def cloud_path(backend_root: Path, source: str, record: dict) -> Path:
    return backend_root / "clouds" / source / f"{record['k']:06d}.npz"


def ensure_disparities(records: list[dict], indices: list[int], root: Path,
                       source: str, config,
                       candidate: FastFoundationCandidate,
                       max_disparity: int, fast_root: Path,
                       conda_env: str):
    backend_root = _backend_root(root, config, candidate, max_disparity)
    jobs = []
    for index in indices:
        record = records[index]
        output = disparity_path(backend_root, source, record)
        if output.exists():
            continue
        jobs.append({
            "left": str(_cache_image_path(
                root, source, record, "main", "jpg").resolve()),
            "right": str(_cache_image_path(
                root, source, record, "aux", "jpg").resolve()),
            "output": str(output.resolve()),
        })
    if not jobs:
        print(f"using {len(indices)} cached Fast-FoundationStereo disparities "
              f"for {source}")
        return backend_root
    jobs_path = backend_root / f"jobs_{source}.jsonl"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w") as stream:
        for job in jobs:
            stream.write(json.dumps(job) + "\n")
    print(f"computing {len(jobs)} Fast-FoundationStereo pairs: "
          f"{candidate.checkpoint}, iters={candidate.valid_iters}, "
          f"max_disp={max_disparity}", flush=True)
    subprocess.run([
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", str(WORKER_PATH),
        "--repo", str(fast_root),
        "--checkpoint", candidate.checkpoint,
        "--valid-iters", str(candidate.valid_iters),
        "--max-disp", str(max_disparity),
        "--jobs", str(jobs_path),
    ], check=True)
    return backend_root


def fast_frame_result(root: Path, backend_root: Path, source: str,
                      record: dict, preprocessor: StereoPreprocessor,
                      workspace_xy: tuple[np.ndarray, np.ndarray]):
    cached_cloud = cloud_path(backend_root, source, record)
    if cached_cloud.exists():
        with np.load(cached_cloud) as cached:
            return {
                "points": cached["points"],
                "confidence": cached["confidence"],
                "left_mask_pixels": int(cached["left_mask_pixels"]),
                "correspondence_rejected": int(cached["correspondence_rejected"]),
                "stereo_ms": float(cached["stereo_ms"]),
                "filter_ms": float(cached["filter_ms"]),
            }
    path = disparity_path(backend_root, source, record)
    if not path.exists():
        raise RuntimeError(f"missing neural disparity {path}")
    with np.load(path) as disparity_file:
        disparity = disparity_file["disparity"].astype(np.float32)
        reverse = disparity_file["right_disparity"].astype(np.float32)
        stereo_ms = float(disparity_file["inference_ms"])
    _, masks = _load_prepared(root, record, source)
    main = _anchor_transform(record, "main")
    started = time.perf_counter()
    cloud = disparity_to_cloud(
        disparity, reverse, masks["main"], masks["aux"],
        preprocessor.geometry.Q,
        T_base_rectified_main(main, preprocessor.rectification),
        preprocessor.config, workspace_xy)
    points, confidence = voxel_downsample(
        cloud.points_base, cloud.confidence,
        preprocessor.config.voxel_size_m)
    cloud = CloudResult(points, confidence, cloud.left_mask_pixels,
                        cloud.correspondence_rejected)
    filter_ms = (time.perf_counter() - started) * 1000.0
    cached_cloud.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cached_cloud,
        points=cloud.points_base.astype(np.float32),
        confidence=cloud.confidence.astype(np.float32),
        left_mask_pixels=np.int32(cloud.left_mask_pixels),
        correspondence_rejected=np.int32(cloud.correspondence_rejected),
        stereo_ms=np.float32(stereo_ms),
        filter_ms=np.float32(filter_ms),
    )
    return {
        "points": cloud.points_base,
        "confidence": cloud.confidence,
        "left_mask_pixels": cloud.left_mask_pixels,
        "correspondence_rejected": cloud.correspondence_rejected,
        "stereo_ms": stereo_ms,
        "filter_ms": filter_ms,
    }


def evaluate_fast(records, indices, root, source, config, candidate,
                  max_disparity, fast_root, conda_env, preprocessor,
                  half_extents, workspace_xy):
    backend_root = ensure_disparities(
        records, indices, root, source, config, candidate, max_disparity,
        fast_root, conda_env)

    def result_for_index(index):
        return fast_frame_result(
            root, backend_root, source, records[index], preprocessor,
            workspace_xy)

    return evaluate_results(
        records, indices, result_for_index, source, half_extents, workspace_xy)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fast-FoundationStereo on the Stage 2 dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--sam2-model", choices=("tiny", "base+"), default="tiny")
    parser.add_argument("--fast-root", type=Path, default=DEFAULT_FAST_ROOT)
    parser.add_argument("--conda-env", default=FAST_ENV)
    parser.add_argument("--max-development-frames", type=int, default=0,
                        help="smoke-test cap; does not write acceptance report")
    parser.add_argument("--diagnose-frame", type=int, default=0,
                        help="print staged filter counts for one cached frame")
    parser.add_argument("--diagnose-source", choices=("inpainted", "raw"),
                        default="inpainted")
    args = parser.parse_args()

    config = load_config()
    records, meta = load_dataset(args.dataset)
    preprocessor = StereoPreprocessor(args.calibration, config)
    movement_mm, movement_deg = validate_recorded_camera_placement(
        records, preprocessor)
    print(f"camera placement vs calibrated anchor: {movement_mm:.2f} mm / "
          f"{movement_deg:.3f} deg")
    root = _cache_root(
        args.dataset, args.calibration, config, args.sam2_model)
    if not (root / "manifest.yaml").exists():
        raise RuntimeError(
            "prepared dense-stereo cache is missing; run "
            "real.tracking.eval_dense_stereo first")
    if not args.fast_root.exists():
        raise RuntimeError(f"Fast-FoundationStereo checkout missing: {args.fast_root}")

    labels = relabel_static(records, config)
    windows = static_windows(labels, config.min_static_window_frames)
    workspace_xy = load_workspace_bounds()
    windows = in_workspace_windows(records, windows, *workspace_xy)
    development_windows, held_out_windows = split_windows(
        windows, config.held_out_fraction)
    visible_development = visible_windows(
        root, records, development_windows, config.min_mask_area_px)
    visible_held_out = visible_windows(
        root, records, held_out_windows, config.min_mask_area_px)
    development_indices = _development_indices(
        visible_development, config.development_frame_stride)
    if args.max_development_frames:
        development_indices = development_indices[:args.max_development_frames]
    held_out_indices = _flatten(visible_held_out)
    half_extents = np.asarray(meta["half_extents"], dtype=np.float64)
    depths = rectified_workspace_depths(
        records, _flatten(development_windows), preprocessor, half_extents)
    minimum, count = disparity_range_from_depths(
        depths, preprocessor.rectification, config)
    max_disparity = max_disparity_for_range(
        minimum, count, config.fast_max_disp_multiple)
    print(f"Fast-FoundationStereo max_disp={max_disparity}; "
          f"development={len(development_indices)}, held_out={len(held_out_indices)}")
    if args.diagnose_frame:
        matching = [record for record in records
                    if int(record["k"]) == args.diagnose_frame]
        if len(matching) != 1:
            raise RuntimeError(f"frame k={args.diagnose_frame} not found")
        candidate = config.fast_candidates[-1]
        record = matching[0]
        ensure_disparities(
            records, [records.index(record)], root, args.diagnose_source,
            config, candidate, max_disparity, args.fast_root, args.conda_env)
        backend_root = _backend_root(root, config, candidate, max_disparity)
        with np.load(disparity_path(
                backend_root, args.diagnose_source, record)) as disparity_file:
            disparity = disparity_file["disparity"].astype(np.float32)
            reverse = disparity_file["right_disparity"].astype(np.float32)
        _, masks = _load_prepared(root, record, args.diagnose_source)
        lr_valid, residual, right_x = left_right_validity(
            disparity, reverse, config.lr_max_error_px)
        yy, _ = np.indices(disparity.shape)
        inside = (right_x >= 0) & (right_x < disparity.shape[1])
        right_hit = np.zeros_like(masks["main"])
        right_hit[inside] = masks["aux"][yy[inside], right_x[inside]]
        positive = masks["main"] & (disparity > 0.0)
        paired = positive & right_hit
        finite_residual = residual[paired & np.isfinite(residual)]
        result = fast_frame_result(
            root, backend_root, args.diagnose_source, record,
            preprocessor, workspace_xy)
        print(f"frame {args.diagnose_frame} staged counts:")
        print(f"  left mask: {int(masks['main'].sum())}")
        print(f"  positive disparity: {int(positive.sum())}")
        print(f"  reprojection in right mask: {int(paired.sum())}")
        print(f"  left-right consistent <= {config.lr_max_error_px:.1f}px: "
              f"{int(np.count_nonzero(paired & lr_valid))}")
        if finite_residual.size:
            print("  LR residual px p50/p90/p95/p99: "
                  + np.array2string(
                      np.percentile(finite_residual, (50, 90, 95, 99)),
                      precision=2))
        print(f"  final workspace/MAD/voxel points: {len(result['points'])}")
        return

    development_reports = []
    for index, candidate in enumerate(config.fast_candidates, 1):
        print(f"candidate {index}/{len(config.fast_candidates)}: {candidate}")
        report = evaluate_fast(
            records, development_indices, root, "inpainted", config,
            candidate, max_disparity, args.fast_root, args.conda_env,
            preprocessor, half_extents, workspace_xy)
        development_reports.append({
            "candidate": asdict(candidate),
            "report": report,
        })
        print(yaml.safe_dump(report, sort_keys=False).rstrip())
    if args.max_development_frames:
        print("smoke test complete; acceptance report intentionally not written")
        return
    winner_index = min(
        range(len(development_reports)),
        key=lambda i: _selection_score(development_reports[i]["report"]))
    winner = config.fast_candidates[winner_index]
    print(f"frozen development winner: {winner}")
    held_inpainted = evaluate_fast(
        records, held_out_indices, root, "inpainted", config, winner,
        max_disparity, args.fast_root, args.conda_env, preprocessor,
        half_extents, workspace_xy)
    held_raw = evaluate_fast(
        records, held_out_indices, root, "raw", config, winner,
        max_disparity, args.fast_root, args.conda_env, preprocessor,
        half_extents, workspace_xy)
    gates = {
        "median_surface_error": held_inpainted["surface_error_mm"]["median"] <= 3.0,
        "p95_surface_error": held_inpainted["surface_error_mm"]["p95"] <= 8.0,
        "catastrophic_outliers": (
            held_inpainted["catastrophic_outside_fraction"] < 0.01),
        "static_valid_points": (
            held_inpainted["frames_at_least_128_fraction"] >= 0.95),
        "latency": held_inpainted["latency_ms"]["p95"] <= 100.0,
        "slice_valid_points": all(
            item["median_valid_points"] >= 64
            for item in held_inpainted["slices"].values()),
    }
    final = {
        "backend": "fast_foundation_stereo",
        "max_disparity": max_disparity,
        "frozen_candidate": asdict(winner),
        "development": development_reports,
        "held_out_tag_inpainted": held_inpainted,
        "held_out_raw_tag": held_raw,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }
    report_path = root / "fast_foundation_report.yaml"
    _write_yaml(report_path, final)
    print(yaml.safe_dump(final, sort_keys=False))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
