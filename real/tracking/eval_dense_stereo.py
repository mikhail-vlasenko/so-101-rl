"""Offline Stage 2 dense-stereo feasibility gate on a tagged sponge dataset.

The command never touches policy code. It recomputes a jitter-tolerant static
label from recorded tag GT, validates that the recorded camera placement still
matches the stereo calibration's tag-derived reference, caches rectified raw,
tag-inpainted and SAM-mask inputs under a calibration/configuration fingerprint,
and tunes a small configured StereoSGBM grid on whole development placements.
Only after the winning candidate is frozen does it evaluate held-out placement
windows and compare tag-inpainted against raw-tag imagery.

Run:
    conda run --no-capture-output -n mujoco_env python -m \
        real.tracking.eval_dense_stereo --dataset datasets/sponge_<stamp>

The SAM masks are shared with ``eval_estimator`` and resume from its cache. Use
``--prepare-only`` to build masks/rectified inputs without running the grid.
``--max-frames`` is a smoke-test aid and cannot produce an acceptance result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from real.calib.align_stereo_rig import relative_pose_change
from real.calib.calibrate_stereo import load_limits as load_calibration_limits
from real.calib.extrinsics import average_transforms, mat_inv, pos_quat_to_mat
from real.tracking.dense_stereo import (
    CALIBRATION_PATH,
    CloudResult,
    DenseStereoConfig,
    SGBMCandidate,
    StereoPreprocessor,
    T_base_rectified_main,
    calibration_fingerprint,
    disparity_range_from_depths,
    disparity_to_cloud,
    inpaint_sponge_tags,
    load_config,
    preprocessing_key,
    sgbm_backend_key,
    sgbm_disparities,
    voxel_downsample,
)
from real.tracking.eval_estimator import compute_masks, gt_pose, load_dataset, load_mask
from real.tracking.record_shapes import CausalMeanPosition, load_workspace_bounds
from src.shape_obs import STATIC_DWELL_S, VISIBLE_FRACTION_MIN, is_static


def relabel_static(records: list[dict], config: DenseStereoConfig) -> list[bool]:
    """Recompute capture labels from denoised GT without modifying raw records."""
    position_filter = CausalMeanPosition(config.static_mean_window_s)
    history_t: list[float] = []
    history_p: list[np.ndarray] = []
    labels = []
    for record in records:
        pose = gt_pose(record)
        if pose is None:
            labels.append(False)
            position_filter = CausalMeanPosition(config.static_mean_window_s)
            history_t.clear()
            history_p.clear()
            continue
        t = float(np.mean([record["t"][name] for name in record["t"]]))
        position = position_filter.update(t, pose[0])
        history_t.append(t)
        history_p.append(position)
        cutoff = t - 2.0 * STATIC_DWELL_S
        while len(history_t) > 2 and history_t[0] < cutoff:
            history_t.pop(0)
            history_p.pop(0)
        labels.append(bool(is_static(history_t, history_p)))
    return labels


def static_windows(labels: list[bool], min_frames: int) -> list[list[int]]:
    windows = []
    current = []
    for index, label in enumerate(labels):
        if label:
            current.append(index)
        elif current:
            if len(current) >= min_frames:
                windows.append(current)
            current = []
    if len(current) >= min_frames:
        windows.append(current)
    return windows


def split_windows(windows: list[list[int]], held_out_fraction: float):
    if len(windows) < 4:
        raise RuntimeError(
            f"need at least four static placement windows, found {len(windows)}")
    held_count = max(1, int(round(len(windows) * held_out_fraction)))
    held_ids = set(np.linspace(0, len(windows) - 1, held_count + 2)[1:-1]
                   .round().astype(int).tolist())
    development = [window for i, window in enumerate(windows) if i not in held_ids]
    held_out = [window for i, window in enumerate(windows) if i in held_ids]
    return development, held_out


def in_workspace_windows(records: list[dict], windows: list[list[int]],
                         low: np.ndarray, high: np.ndarray) -> list[list[int]]:
    """Keep whole static placements whose median GT center is deployable."""
    accepted = []
    for window in windows:
        centers = [gt_pose(records[index])[0][:2] for index in window
                   if gt_pose(records[index]) is not None]
        if not centers:
            continue
        center = np.median(np.stack(centers), axis=0)
        if np.all(center >= low) and np.all(center <= high):
            accepted.append(window)
    return accepted


def _anchor_transform(record: dict, name: str) -> np.ndarray | None:
    anchor = record["T_base_cam"][name]
    if anchor is None:
        return None
    return pos_quat_to_mat(anchor["pos"], anchor["quat"])


def validate_recorded_camera_placement(records: list[dict], preprocessor: StereoPreprocessor):
    reference = preprocessor.rectification.anchor_reference_T_aux_main
    if reference is None:
        raise RuntimeError(
            "stereo calibration has no table-anchor reference; record one with "
            "real.calib.align_stereo_rig before dense evaluation")
    relative = []
    for record in records:
        main = _anchor_transform(record, "main")
        aux = _anchor_transform(record, "aux")
        if main is not None and aux is not None:
            relative.append(mat_inv(aux) @ main)
    if not relative:
        raise RuntimeError("dataset contains no frame with both camera anchors")
    measured = average_transforms(relative)
    movement_mm, movement_deg = relative_pose_change(measured, reference)
    limits = load_calibration_limits()
    if (movement_mm > limits.camera_movement_warning_translation_mm
            or movement_deg > limits.camera_movement_warning_rotation_deg):
        raise RuntimeError(
            "dataset camera placement does not match its stereo calibration "
            f"({movement_mm:.2f} mm / {movement_deg:.3f} deg). Lay the "
            "checkerboard flat in both views, rerun `python -m "
            "real.calib.calibrate_stereo`, remove it, and record the anchor "
            "reference before capturing again.")
    return movement_mm, movement_deg


def _cache_root(dataset_dir: Path, calibration_path: Path,
                config: DenseStereoConfig, sam2_model: str) -> Path:
    return (dataset_dir / "dense_stereo"
            / calibration_fingerprint(calibration_path)[:16]
            / f"pre_{preprocessing_key(config)}"
            / f"sam_{sam2_model}")


def _cache_image_path(root: Path, kind: str, record: dict, name: str,
                      suffix: str) -> Path:
    return root / "rectified" / kind / f"{name}_{record['k']:06d}.{suffix}"


def prepare_record(dataset_dir: Path, record: dict, mask_dir: Path,
                   preprocessor: StereoPreprocessor, root: Path):
    outputs = {}
    for name in ("main", "aux"):
        raw_path = _cache_image_path(root, "raw", record, name, "jpg")
        inpainted_path = _cache_image_path(root, "inpainted", record, name, "jpg")
        mask_path = _cache_image_path(root, "masks", record, name, "png")
        for path in (raw_path, inpainted_path, mask_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not (raw_path.exists() and inpainted_path.exists() and mask_path.exists()):
            frame = cv2.imread(str(dataset_dir / record["frame"][name]))
            if frame is None:
                raise RuntimeError(f"failed to read {record['frame'][name]}")
            if not raw_path.exists():
                raw = preprocessor.rectify_image(name, frame)
                if not cv2.imwrite(str(raw_path), raw,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to write {raw_path}")
            if not inpainted_path.exists():
                inpainted, _ = inpaint_sponge_tags(
                    frame, record["tags"][name],
                    preprocessor.config.tag_inpaint_dilate_px,
                    preprocessor.config.tag_inpaint_radius_px)
                inpainted = preprocessor.rectify_image(name, inpainted)
                if not cv2.imwrite(str(inpainted_path), inpainted,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to write {inpainted_path}")
            if not mask_path.exists():
                mask = preprocessor.rectify_mask(
                    name, load_mask(mask_dir, record, name))
                if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"failed to write {mask_path}")
        outputs[name] = {
            "raw": raw_path,
            "inpainted": inpainted_path,
            "mask": mask_path,
        }
    return outputs


def prepare_inputs(dataset_dir: Path, records: list[dict], mask_dir: Path,
                   preprocessor: StereoPreprocessor, root: Path):
    started = time.monotonic()
    total = len(records)
    for index, record in enumerate(records, 1):
        prepare_record(dataset_dir, record, mask_dir, preprocessor, root)
        if index % 100 == 0 or index == total:
            elapsed = time.monotonic() - started
            rate = index / max(elapsed, 1e-9)
            eta = (total - index) / max(rate, 1e-9)
            print(f"rectified cache {index}/{total}  {rate:.1f} pair/s  "
                  f"ETA {eta:.0f}s", flush=True)


def _load_prepared(root: Path, record: dict, source: str):
    images = {}
    masks = {}
    for name in ("main", "aux"):
        image_path = _cache_image_path(root, source, record, name, "jpg")
        mask_path = _cache_image_path(root, "masks", record, name, "png")
        images[name] = cv2.imread(str(image_path))
        masks[name] = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if images[name] is None or masks[name] is None:
            raise RuntimeError(f"missing prepared input for frame {record['k']}")
        masks[name] = masks[name] > 127
    return images, masks


def visible_window_indices(root: Path, records: list[dict], window: list[int],
                           min_mask_area_px: int) -> list[int]:
    """Static frames whose two mask areas are near that placement's maxima."""
    areas = {name: [] for name in ("main", "aux")}
    for index in window:
        record = records[index]
        for name in areas:
            path = _cache_image_path(root, "masks", record, name, "png")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"missing prepared mask {path}")
            areas[name].append(int(np.count_nonzero(mask)))
    maxima = {name: max(values) for name, values in areas.items()}
    if any(value < min_mask_area_px for value in maxima.values()):
        return []
    return [
        index for offset, index in enumerate(window)
        if all(areas[name][offset] >= VISIBLE_FRACTION_MIN * maxima[name]
               for name in areas)
    ]


def visible_windows(root: Path, records: list[dict], windows: list[list[int]],
                    min_mask_area_px: int) -> list[list[int]]:
    return [visible for window in windows
            if (visible := visible_window_indices(
                root, records, window, min_mask_area_px))]


def rectified_workspace_depths(records: list[dict], indices: list[int],
                               preprocessor: StereoPreprocessor,
                               half_extents: np.ndarray) -> np.ndarray:
    depths = []
    radius = float(np.linalg.norm(half_extents))
    for index in indices:
        record = records[index]
        pose = gt_pose(record)
        main = _anchor_transform(record, "main")
        if pose is None or main is None:
            continue
        T_base_rectified = T_base_rectified_main(main, preprocessor.rectification)
        center_rectified = (pose[0] - T_base_rectified[:3, 3]) \
            @ T_base_rectified[:3, :3]
        depths.extend((center_rectified[2] - radius,
                       center_rectified[2] + radius))
    return np.asarray(depths, dtype=np.float64)


def box_signed_distance(points_base: np.ndarray, center: np.ndarray,
                        rotation: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    local = (np.asarray(points_base) - center) @ rotation
    q = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _frame_slice(record: dict, workspace_low: np.ndarray,
                 workspace_high: np.ndarray) -> str:
    center, rotation = gt_pose(record)
    face = "xyz"[int(np.argmax(np.abs(rotation[2, :])))]
    axis = rotation[:, 0] if face != "x" else rotation[:, 1]
    yaw = int(((np.arctan2(axis[1], axis[0]) + np.pi) / (np.pi / 2))) % 4
    cell = np.clip(((center[:2] - workspace_low)
                    / (workspace_high - workspace_low) * 2).astype(int), 0, 1)
    return f"{face}/yaw{yaw}/region{cell[0]}{cell[1]}"


def _backend_path(root: Path, candidate: SGBMCandidate,
                  minimum: int, count: int, source: str, record: dict) -> Path:
    key = sgbm_backend_key(load_config(), candidate)
    return (root / "backends" / f"sgbm_{key}_d{minimum}_n{count}" / source
            / f"{record['k']:06d}.npz")


def run_frame(root: Path, record: dict, source: str,
              candidate: SGBMCandidate, minimum: int, count: int,
              preprocessor: StereoPreprocessor,
              workspace_xy: tuple[np.ndarray, np.ndarray]):
    cache_path = _backend_path(root, candidate, minimum, count, source, record)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {
                "points": cached["points"],
                "confidence": cached["confidence"],
                "disparity": cached["disparity"].astype(np.float32),
                "left_mask_pixels": int(cached["left_mask_pixels"]),
                "correspondence_rejected": int(cached["correspondence_rejected"]),
                "stereo_ms": float(cached["stereo_ms"]),
                "filter_ms": float(cached["filter_ms"]),
            }
    images, masks = _load_prepared(root, record, source)
    started = time.perf_counter()
    disparity, reverse = sgbm_disparities(
        images["main"], images["aux"], candidate, minimum, count)
    stereo_ms = (time.perf_counter() - started) * 1000.0
    main = _anchor_transform(record, "main")
    if main is None:
        cloud = CloudResult(np.empty((0, 3)), np.empty(0),
                            int(masks["main"].sum()), 0)
        filter_ms = 0.0
    else:
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        points=cloud.points_base.astype(np.float32),
        confidence=cloud.confidence.astype(np.float32),
        disparity=disparity.astype(np.float16),
        left_mask_pixels=np.int32(cloud.left_mask_pixels),
        correspondence_rejected=np.int32(cloud.correspondence_rejected),
        stereo_ms=np.float32(stereo_ms),
        filter_ms=np.float32(filter_ms),
    )
    return {
        "points": cloud.points_base,
        "confidence": cloud.confidence,
        "disparity": disparity,
        "left_mask_pixels": cloud.left_mask_pixels,
        "correspondence_rejected": cloud.correspondence_rejected,
        "stereo_ms": stereo_ms,
        "filter_ms": filter_ms,
    }


def evaluate_candidate(records: list[dict], indices: list[int], root: Path,
                       source: str, candidate: SGBMCandidate,
                       minimum: int, count: int,
                       preprocessor: StereoPreprocessor,
                       half_extents: np.ndarray,
                       workspace_xy: tuple[np.ndarray, np.ndarray]):
    def result_for_index(index):
        return run_frame(
            root, records[index], source, candidate, minimum, count,
            preprocessor, workspace_xy)

    return evaluate_results(
        records, indices, result_for_index, source, half_extents, workspace_xy)


def evaluate_results(records: list[dict], indices: list[int], result_for_index,
                     label: str, half_extents: np.ndarray,
                     workspace_xy: tuple[np.ndarray, np.ndarray]):
    """Backend-independent tag-GT aggregation over frozen frame results."""
    absolute_errors = []
    outside_catastrophic = 0
    total_points = 0
    frame_counts = []
    rejected = []
    latencies = []
    slices: dict[str, list[int]] = {}
    per_frame = []
    for progress, index in enumerate(indices, 1):
        record = records[index]
        result = result_for_index(index)
        center, rotation = gt_pose(record)
        signed = box_signed_distance(
            result["points"], center, rotation, half_extents)
        absolute_errors.append(np.abs(signed))
        outside_catastrophic += int(np.count_nonzero(signed > 0.02))
        total_points += signed.size
        frame_counts.append(signed.size)
        rejected.append(result["correspondence_rejected"]
                        / max(result["left_mask_pixels"], 1))
        latencies.append(result["stereo_ms"] + result["filter_ms"])
        slice_name = _frame_slice(record, *workspace_xy)
        slices.setdefault(slice_name, []).append(signed.size)
        per_frame.append({
            "k": int(record["k"]),
            "slice": slice_name,
            "valid_points": int(signed.size),
            "left_mask_pixels": int(result["left_mask_pixels"]),
        })
        if progress % 50 == 0:
            print(f"  {label} {progress}/{len(indices)}", flush=True)
    errors = (np.concatenate(absolute_errors) if absolute_errors
              else np.empty(0, dtype=np.float64))
    counts_array = np.asarray(frame_counts)
    slice_report = {
        name: {
            "frames": len(values),
            "median_valid_points": float(np.median(values)),
            "min_valid_points": int(np.min(values)),
        }
        for name, values in sorted(slices.items())
    }
    return {
        "frames": len(indices),
        "points": int(total_points),
        "surface_error_mm": {
            "median": float(np.median(errors) * 1000.0) if errors.size else math.inf,
            "rms": float(np.sqrt(np.mean(errors ** 2)) * 1000.0)
            if errors.size else math.inf,
            "p95": float(np.percentile(errors, 95) * 1000.0)
            if errors.size else math.inf,
        },
        "catastrophic_outside_fraction": (
            outside_catastrophic / max(total_points, 1)),
        "frames_at_least_128_fraction": float(np.mean(counts_array >= 128))
        if counts_array.size else 0.0,
        "valid_points": {
            "median": float(np.median(counts_array)) if counts_array.size else 0.0,
            "p05": float(np.percentile(counts_array, 5)) if counts_array.size else 0.0,
            "min": int(np.min(counts_array)) if counts_array.size else 0,
        },
        "correspondence_rejected_fraction": float(np.mean(rejected)),
        "latency_ms": {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "slices": slice_report,
        "worst_frames": sorted(
            per_frame, key=lambda item: (item["valid_points"], item["k"]))[:12],
    }


def _selection_score(report: dict):
    error = report["surface_error_mm"]
    return (
        max(error["median"] / 3.0, error["p95"] / 8.0,
            report["catastrophic_outside_fraction"] / 0.01,
            (1.0 - report["frames_at_least_128_fraction"]) / 0.05),
        error["p95"],
        -report["valid_points"]["median"],
    )


def _flatten(windows: list[list[int]]) -> list[int]:
    return [index for window in windows for index in window]


def _development_indices(windows: list[list[int]], stride: int) -> list[int]:
    return [index for window in windows for index in window[::stride]]


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the offline dense-stereo feasibility gate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--prompt", default="sponge")
    parser.add_argument("--sam2-model", choices=("tiny", "base+"), default="tiny")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="validate placement/split without loading SAM")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="smoke-test cap; disables acceptance output")
    args = parser.parse_args()

    config = load_config()
    records, meta = load_dataset(args.dataset)
    preprocessor = StereoPreprocessor(args.calibration, config)
    movement_mm, movement_deg = validate_recorded_camera_placement(
        records, preprocessor)
    print(f"camera placement vs calibrated anchor: {movement_mm:.2f} mm / "
          f"{movement_deg:.3f} deg")

    labels = relabel_static(records, config)
    windows = static_windows(labels, config.min_static_window_frames)
    workspace_xy = load_workspace_bounds()
    in_bounds = in_workspace_windows(records, windows, *workspace_xy)
    development_windows, held_out_windows = split_windows(
        in_bounds, config.held_out_fraction)
    print(f"static relabel: {sum(labels)}/{len(labels)} frames, "
          f"{len(windows)} placement windows, {len(in_bounds)} in workspace; "
          f"{len(development_windows)} development / "
          f"{len(held_out_windows)} held out")
    if args.analyze_only:
        return

    mask_dir = compute_masks(
        args.dataset, records, meta, args.prompt, args.sam2_model)
    root = _cache_root(args.dataset, args.calibration, config, args.sam2_model)
    selected_records = records if args.max_frames == 0 else records[:args.max_frames]
    prepare_inputs(args.dataset, selected_records, mask_dir, preprocessor, root)
    manifest = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "calibration": str(args.calibration),
        "calibration_sha256": calibration_fingerprint(args.calibration),
        "configuration": asdict(config),
        "processed_image_size": list(preprocessor.geometry.image_size),
        "processed_pad_top": preprocessor.geometry.pad_top,
        "static_frames": int(sum(labels)),
        "static_windows": len(windows),
        "development_windows": len(development_windows),
        "held_out_windows": len(held_out_windows),
    }
    _write_yaml(root / "manifest.yaml", manifest)
    if args.prepare_only or args.max_frames:
        print(f"prepared cache at {root}")
        return

    visible_development_windows = visible_windows(
        root, records, development_windows, config.min_mask_area_px)
    visible_held_out_windows = visible_windows(
        root, records, held_out_windows, config.min_mask_area_px)
    development_indices = _development_indices(
        visible_development_windows, config.development_frame_stride)
    held_out_indices = _flatten(visible_held_out_windows)
    print(
        f"visibility gate: development {len(development_indices)} sampled frames "
        f"from {len(visible_development_windows)}/{len(development_windows)} "
        f"placements; held out {len(held_out_indices)}/"
        f"{len(_flatten(held_out_windows))} frames from "
        f"{len(visible_held_out_windows)}/{len(held_out_windows)} placements")
    if not development_indices or not held_out_indices:
        raise RuntimeError("visibility gate left no development or held-out frames")
    half_extents = np.asarray(meta["half_extents"], dtype=np.float64)
    depths = rectified_workspace_depths(
        records, _flatten(development_windows), preprocessor, half_extents)
    minimum, count = disparity_range_from_depths(
        depths, preprocessor.rectification, config)
    print(f"derived SGBM disparity interval [{minimum}, {minimum + count}) "
          f"from rectified workspace depth {depths.min():.3f}-{depths.max():.3f} m")
    development_reports = []
    for index, candidate in enumerate(config.sgbm_candidates, 1):
        print(f"candidate {index}/{len(config.sgbm_candidates)}: {candidate}")
        report = evaluate_candidate(
            records, development_indices, root, "inpainted", candidate,
            minimum, count, preprocessor, half_extents, workspace_xy)
        development_reports.append({
            "candidate": asdict(candidate),
            "report": report,
        })
        print(yaml.safe_dump(report, sort_keys=False).rstrip())
    winner_index = min(
        range(len(development_reports)),
        key=lambda i: _selection_score(development_reports[i]["report"]))
    winner = config.sgbm_candidates[winner_index]
    print(f"frozen development winner: {winner}")

    held_inpainted = evaluate_candidate(
        records, held_out_indices, root, "inpainted", winner,
        minimum, count, preprocessor, half_extents, workspace_xy)
    held_raw = evaluate_candidate(
        records, held_out_indices, root, "raw", winner,
        minimum, count, preprocessor, half_extents, workspace_xy)
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
        "backend": "opencv_stereo_sgbm",
        "disparity": {"min": minimum, "num": count},
        "frozen_candidate": asdict(winner),
        "development": development_reports,
        "held_out_tag_inpainted": held_inpainted,
        "held_out_raw_tag": held_raw,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }
    _write_yaml(root / "sgbm_report.yaml", final)
    print(yaml.safe_dump(final, sort_keys=False))
    print(f"report: {root / 'sgbm_report.yaml'}")
    if not final["passes_all_gates"]:
        print("StereoSGBM did not pass every gate; Fast-FoundationStereo is next.")


if __name__ == "__main__":
    main()
