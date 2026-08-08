"""Calibrate the current relative pose of the independently mounted C922 pair.

Both cameras observe the same stationary, flat checkerboard. Their intrinsics
are already calibrated, so many board orientations are unnecessary here: this
tool averages repeated subpixel detections, solves each camera's pose against
the common metric board, derives ``T_aux_main``, and constructs OpenCV stereo
rectification matrices. It refuses to overwrite the output unless camera
reprojection and rectified vertical correspondence pass configured gates.

The cameras are on independent stands, so this artifact is a placement
snapshot, not a permanent factory calibration. Re-run it whenever either stand
moves materially. Small motion during an episode is tracked separately through
the two-tag table-board camera EMAs.

The checkerboard specification comes from the two intrinsic YAMLs and must
match exactly. Lay the print flat, keep the complete inner-corner grid visible
in both cameras, then run:

    conda run -n mujoco_env python -m real.calib.calibrate_stereo
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from real.calib.calibrate_camera import detect, object_points
from real.calib.extrinsics import mat_inv, rt_to_mat
from real.vision.camera import SERIALS
from real.vision.intrinsics import intrinsics_path
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"
OUTPUT_PATH = REPO_ROOT / "real" / "vision" / "stereo_calibration.yaml"


@dataclass(frozen=True)
class StereoCalibrationLimits:
    capture_frames: int
    min_detected_pairs: int
    max_reprojection_rmse_px: float
    max_rectified_vertical_p95_px: float
    camera_movement_warning_translation_mm: float
    camera_movement_warning_rotation_deg: float


@dataclass(frozen=True)
class StereoCalibrationResult:
    T_aux_main: np.ndarray
    rectification_rotations: dict[str, np.ndarray]
    projection_matrices: dict[str, np.ndarray]
    Q: np.ndarray
    valid_rois: dict[str, tuple[int, int, int, int]]
    reprojection_rmse_px: dict[str, float]
    temporal_corner_rmse_px: dict[str, float]
    vertical_mean_px: float
    vertical_p95_px: float
    vertical_max_px: float


@dataclass(frozen=True)
class StereoRectification:
    image_size: tuple[int, int]
    T_aux_main: np.ndarray
    rotations: dict[str, np.ndarray]
    projections: dict[str, np.ndarray]
    Q: np.ndarray
    valid_rois: dict[str, tuple[int, int, int, int]]
    anchor_reference_T_aux_main: np.ndarray | None


def load_stereo_rectification(path: Path = OUTPUT_PATH) -> StereoRectification:
    """Load the accepted placement snapshot needed to rectify live frames."""
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if data["schema_version"] != 1:
        raise RuntimeError(
            f"unsupported stereo calibration schema {data['schema_version']}")
    for name in CAMERA_NAMES:
        source = data["source_intrinsics"][name]
        intrinsic_path = REPO_ROOT / source["path"]
        fingerprint = hashlib.sha256(intrinsic_path.read_bytes()).hexdigest()
        if fingerprint != source["sha256"]:
            raise RuntimeError(
                f"{intrinsic_path} changed after stereo calibration; recalibrate stereo")
        if source["serial"] != SERIALS[name]:
            raise RuntimeError(
                f"stereo calibration {name} serial {source['serial']} does not match "
                f"configured {SERIALS[name]}")
    anchor_reference = data.get("anchor_reference")
    return StereoRectification(
        image_size=(int(data["image_width"]), int(data["image_height"])),
        T_aux_main=np.asarray(data["T_aux_main"], dtype=np.float64),
        rotations={
            name: np.asarray(data["rectification_rotation"][name], dtype=np.float64)
            for name in CAMERA_NAMES
        },
        projections={
            name: np.asarray(data["projection_matrix"][name], dtype=np.float64)
            for name in CAMERA_NAMES
        },
        Q=np.asarray(data["Q"], dtype=np.float64),
        valid_rois={
            name: tuple(int(v) for v in data["valid_roi"][name])
            for name in CAMERA_NAMES
        },
        anchor_reference_T_aux_main=(
            None if anchor_reference is None else
            np.asarray(anchor_reference["T_aux_main"], dtype=np.float64)
        ),
    )


def save_anchor_reference(path: Path, T_aux_main: np.ndarray) -> None:
    """Attach the tag-derived known-good placement after removing the checker."""
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if data["schema_version"] != 1:
        raise RuntimeError(
            f"unsupported stereo calibration schema {data['schema_version']}")
    data["anchor_reference"] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "T_aux_main": np.asarray(T_aux_main, dtype=np.float64).tolist(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    temporary.replace(path)


def load_limits(path: Path = CONFIG_PATH) -> StereoCalibrationLimits:
    with path.open() as stream:
        cfg = yaml.safe_load(stream)["stereo_calibration"]
    limits = StereoCalibrationLimits(
        capture_frames=int(cfg["capture_frames"]),
        min_detected_pairs=int(cfg["min_detected_pairs"]),
        max_reprojection_rmse_px=float(cfg["max_reprojection_rmse_px"]),
        max_rectified_vertical_p95_px=float(
            cfg["max_rectified_vertical_p95_px"]),
        camera_movement_warning_translation_mm=float(
            cfg["camera_movement_warning_translation_mm"]),
        camera_movement_warning_rotation_deg=float(
            cfg["camera_movement_warning_rotation_deg"]),
    )
    assert 0 < limits.min_detected_pairs <= limits.capture_frames
    assert limits.max_reprojection_rmse_px > 0.0
    assert limits.max_rectified_vertical_p95_px > 0.0
    assert limits.camera_movement_warning_translation_mm > 0.0
    assert limits.camera_movement_warning_rotation_deg > 0.0
    return limits


def _pose_from_mean_corners(object_pts: np.ndarray, samples: list[np.ndarray],
                            camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_corners = np.mean(np.stack(samples), axis=0).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts.astype(np.float64), mean_corners,
        camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("stationary checkerboard solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(
        object_pts.astype(np.float64), mean_corners,
        camera_matrix, dist_coeffs, rvec, tvec)
    return rvec.reshape(3), tvec.reshape(3)


def _canonicalize_corners(corners: np.ndarray) -> np.ndarray:
    """Choose the same 180-degree checkerboard ordering in both image views."""
    points = np.asarray(corners)
    if float(points[0, 0].sum()) > float(points[-1, 0].sum()):
        return points[::-1].copy()
    return points.copy()


def _reprojection_rmse(object_pts: np.ndarray, samples: list[np.ndarray],
                       camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                       rvec: np.ndarray, tvec: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(
        object_pts, rvec, tvec, camera_matrix, dist_coeffs)
    mean_corners = np.mean(np.stack(samples), axis=0).reshape(-1, 2)
    residuals = mean_corners - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def _temporal_corner_rmse(samples: list[np.ndarray]) -> float:
    stacked = np.stack(samples).reshape(len(samples), -1, 2)
    residuals = stacked - np.mean(stacked, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=2))))


def solve_stationary_board(
        object_pts: np.ndarray,
        image_samples: dict[str, list[np.ndarray]],
        camera_matrices: dict[str, np.ndarray],
        dist_coeffs: dict[str, np.ndarray],
        image_size: tuple[int, int]) -> StereoCalibrationResult:
    """Solve current relative geometry and its rectification quality."""
    if set(image_samples) != set(CAMERA_NAMES):
        raise ValueError(f"image_samples must contain {CAMERA_NAMES}")
    sample_counts = {name: len(image_samples[name]) for name in CAMERA_NAMES}
    if len(set(sample_counts.values())) != 1 or next(iter(sample_counts.values())) == 0:
        raise ValueError(f"paired non-empty samples required, got {sample_counts}")

    image_samples = {
        name: [_canonicalize_corners(corners) for corners in image_samples[name]]
        for name in CAMERA_NAMES
    }
    poses = {}
    reprojection = {}
    temporal = {}
    for name in CAMERA_NAMES:
        rvec, tvec = _pose_from_mean_corners(
            object_pts, image_samples[name], camera_matrices[name],
            dist_coeffs[name])
        poses[name] = rt_to_mat(rvec, tvec)
        reprojection[name] = _reprojection_rmse(
            object_pts, image_samples[name], camera_matrices[name],
            dist_coeffs[name], rvec, tvec)
        temporal[name] = _temporal_corner_rmse(image_samples[name])

    T_aux_main = poses["aux"] @ mat_inv(poses["main"])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera_matrices["main"], dist_coeffs["main"],
        camera_matrices["aux"], dist_coeffs["aux"], image_size,
        T_aux_main[:3, :3], T_aux_main[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)

    rectified = {"main": [], "aux": []}
    for name, rotation, projection in (
            ("main", R1, P1), ("aux", R2, P2)):
        for corners in image_samples[name]:
            rectified[name].append(cv2.undistortPoints(
                corners.astype(np.float64), camera_matrices[name],
                dist_coeffs[name], R=rotation, P=projection).reshape(-1, 2))
    vertical = np.abs(
        np.stack(rectified["main"])[:, :, 1]
        - np.stack(rectified["aux"])[:, :, 1]
    ).reshape(-1)
    return StereoCalibrationResult(
        T_aux_main=T_aux_main,
        rectification_rotations={"main": R1, "aux": R2},
        projection_matrices={"main": P1, "aux": P2},
        Q=Q,
        valid_rois={"main": tuple(int(v) for v in roi1),
                    "aux": tuple(int(v) for v in roi2)},
        reprojection_rmse_px=reprojection,
        temporal_corner_rmse_px=temporal,
        vertical_mean_px=float(np.mean(vertical)),
        vertical_p95_px=float(np.percentile(vertical, 95)),
        vertical_max_px=float(np.max(vertical)),
    )


def calibration_failures(result: StereoCalibrationResult,
                         limits: StereoCalibrationLimits) -> tuple[str, ...]:
    failures = []
    for name in CAMERA_NAMES:
        error = result.reprojection_rmse_px[name]
        if error > limits.max_reprojection_rmse_px:
            failures.append(
                f"{name} reprojection RMSE {error:.3f} px exceeds "
                f"{limits.max_reprojection_rmse_px:.3f} px")
    if result.vertical_p95_px > limits.max_rectified_vertical_p95_px:
        failures.append(
            f"rectified vertical p95 {result.vertical_p95_px:.3f} px exceeds "
            f"{limits.max_rectified_vertical_p95_px:.3f} px")
    return tuple(failures)


def _intrinsics_metadata() -> tuple[dict[str, dict], tuple[int, int],
                                    tuple[int, int], float]:
    metadata = {}
    for name in CAMERA_NAMES:
        path = Path(intrinsics_path(name))
        with path.open() as stream:
            data = yaml.safe_load(stream)
        if data["serial"] != SERIALS[name]:
            raise RuntimeError(
                f"{path} belongs to serial {data['serial']}, expected {SERIALS[name]}")
        metadata[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "serial": data["serial"],
            "focus_absolute": int(data["focus_absolute"]),
            "pattern_inner_corners": tuple(int(v) for v in data["pattern_inner_corners"]),
            "square_size_m": float(data["square_size_m"]),
            "image_size": (int(data["image_width"]), int(data["image_height"])),
        }
    patterns = {metadata[name]["pattern_inner_corners"] for name in CAMERA_NAMES}
    square_sizes = {metadata[name]["square_size_m"] for name in CAMERA_NAMES}
    image_sizes = {metadata[name]["image_size"] for name in CAMERA_NAMES}
    if len(patterns) != 1 or len(square_sizes) != 1 or len(image_sizes) != 1:
        raise RuntimeError(
            "stereo cameras must share checkerboard geometry and image size; "
            f"patterns={patterns}, square_sizes={square_sizes}, image_sizes={image_sizes}")
    return metadata, image_sizes.pop(), patterns.pop(), square_sizes.pop()


def _save(path: Path, result: StereoCalibrationResult,
          metadata: dict[str, dict], image_size: tuple[int, int],
          pattern: tuple[int, int], square_size_m: float,
          detected_pairs: int) -> None:
    data = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_kind": "stationary_planar_board_extrinsics",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "pattern_inner_corners": list(pattern),
        "square_size_m": square_size_m,
        "detected_pairs": detected_pairs,
        "source_intrinsics": {
            name: {
                "path": metadata[name]["path"],
                "sha256": metadata[name]["sha256"],
                "serial": metadata[name]["serial"],
                "focus_absolute": metadata[name]["focus_absolute"],
            }
            for name in CAMERA_NAMES
        },
        "T_aux_main": result.T_aux_main.tolist(),
        "rectification_rotation": {
            name: result.rectification_rotations[name].tolist()
            for name in CAMERA_NAMES
        },
        "projection_matrix": {
            name: result.projection_matrices[name].tolist()
            for name in CAMERA_NAMES
        },
        "Q": result.Q.tolist(),
        "valid_roi": {name: list(result.valid_rois[name]) for name in CAMERA_NAMES},
        "quality": {
            "reprojection_rmse_px": result.reprojection_rmse_px,
            "temporal_corner_rmse_px": result.temporal_corner_rmse_px,
            "rectified_vertical_mean_px": result.vertical_mean_px,
            "rectified_vertical_p95_px": result.vertical_p95_px,
            "rectified_vertical_max_px": result.vertical_max_px,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    temporary.replace(path)


def main() -> None:
    limits = load_limits()
    parser = argparse.ArgumentParser(
        description="Snapshot stereo calibration from a stationary checkerboard")
    parser.add_argument("--frames", type=int, default=limits.capture_frames,
                        help="capture loops to average")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="stereo calibration YAML")
    parser.add_argument("--save-frames", type=Path, default=None,
                        help="directory for final checkerboard overlays")
    args = parser.parse_args()
    if args.frames < limits.min_detected_pairs:
        parser.error(
            f"--frames must be >= configured min_detected_pairs "
            f"({limits.min_detected_pairs})")

    metadata, image_size, pattern, square_size_m = _intrinsics_metadata()
    board_points = object_points(pattern)
    if not np.isclose(square_size_m, np.linalg.norm(board_points[1] - board_points[0])):
        raise RuntimeError(
            "intrinsic checker square size disagrees with calibrate_camera.object_points")

    caps, mats, dists = {}, {}, {}
    for name in CAMERA_NAMES:
        caps[name], mats[name], dists[name] = open_rig_camera(name)
    samples = {name: [] for name in CAMERA_NAMES}
    last_frames = {}
    last_corners = {}
    try:
        for _ in range(args.frames):
            pair = {}
            frames = {}
            for name in CAMERA_NAMES:
                ok, frame = caps[name].read()
                if not ok:
                    raise RuntimeError(f"camera read failed on '{name}'")
                frames[name] = frame
                pair[name] = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
            if all(pair[name] is not None for name in CAMERA_NAMES):
                for name in CAMERA_NAMES:
                    samples[name].append(pair[name])
                last_frames, last_corners = frames, pair
    finally:
        for cap in caps.values():
            cap.release()

    detected_pairs = len(samples["main"])
    print(f"detected complete checkerboard in {detected_pairs}/{args.frames} pairs")
    if detected_pairs < limits.min_detected_pairs:
        raise RuntimeError(
            f"only {detected_pairs} complete pairs; need "
            f"{limits.min_detected_pairs}. Keep the whole checkerboard visible in both views")

    result = solve_stationary_board(
        board_points, samples, mats, dists, image_size)
    print("reprojection RMSE: " + ", ".join(
        f"{name}={result.reprojection_rmse_px[name]:.3f} px"
        for name in CAMERA_NAMES))
    print("temporal corner RMSE: " + ", ".join(
        f"{name}={result.temporal_corner_rmse_px[name]:.3f} px"
        for name in CAMERA_NAMES))
    print(
        f"rectified vertical residual: mean={result.vertical_mean_px:.3f} px, "
        f"p95={result.vertical_p95_px:.3f} px, max={result.vertical_max_px:.3f} px")
    print(f"baseline={np.linalg.norm(result.T_aux_main[:3, 3]) * 1000.0:.2f} mm")

    failures = calibration_failures(result, limits)
    if failures:
        raise RuntimeError("stereo calibration rejected:\n  - " + "\n  - ".join(failures))
    _save(args.output, result, metadata, image_size, pattern, square_size_m,
          detected_pairs)
    print(f"wrote {args.output}")

    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)
        for name in CAMERA_NAMES:
            view = last_frames[name].copy()
            cv2.drawChessboardCorners(view, pattern, last_corners[name], True)
            target = args.save_frames / f"stereo_calibrate_{name}.jpg"
            if not cv2.imwrite(str(target), view):
                raise RuntimeError(f"failed to write {target}")
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
