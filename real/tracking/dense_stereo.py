"""Shared dense-stereo preprocessing, SGBM matching and metric cloud geometry.

Stage 2 evaluates stereo matchers offline, but every backend must consume the
same rectified/inpainted images and pass through this module's mask,
left-right-consistency, base-frame and outlier filters. Coordinates returned by
OpenCV's Q matrix are in the rectified main-camera frame; ``T_base_rectified``
is the only conversion into the arm base frame.

The cache identity includes the accepted stereo-calibration bytes and every
preprocessing/matcher setting. Changing either creates a new cache rather than
silently mixing incompatible disparity or cloud files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from real.calib.calibrate_stereo import StereoRectification, load_stereo_rectification
from real.marker_spec import SPONGE_TAG_IDS
from real.vision.intrinsics import intrinsics_path
from real.vision.pose import load_intrinsics
from real.vision.stereo_rig import CAMERA_NAMES


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"
CALIBRATION_PATH = REPO_ROOT / "real" / "vision" / "stereo_calibration.yaml"


@dataclass(frozen=True)
class SGBMCandidate:
    block_size: int
    uniqueness_ratio: int
    speckle_window_size: int
    speckle_range: int
    disp12_max_diff: int


@dataclass(frozen=True)
class FastFoundationCandidate:
    checkpoint: str
    valid_iters: int


@dataclass(frozen=True)
class DenseStereoConfig:
    processing_scale: float
    processing_height_multiple: int
    tag_inpaint_dilate_px: int
    tag_inpaint_radius_px: int
    static_mean_window_s: float
    held_out_fraction: float
    min_static_window_frames: int
    development_frame_stride: int
    disparity_margin_px: int
    lr_max_error_px: float
    point_sample_count: int
    voxel_size_m: float
    workspace_z_m: tuple[float, float]
    depth_mad_scale: float
    depth_mad_floor_m: float
    min_mask_area_px: int
    sgbm_candidates: tuple[SGBMCandidate, ...]
    fast_max_disp_multiple: int
    fast_candidates: tuple[FastFoundationCandidate, ...]


@dataclass(frozen=True)
class ProcessingGeometry:
    image_size: tuple[int, int]
    pad_top: int
    Q: np.ndarray


@dataclass(frozen=True)
class CloudResult:
    points_base: np.ndarray
    confidence: np.ndarray
    left_mask_pixels: int
    correspondence_rejected: int


def load_config(path: Path = CONFIG_PATH) -> DenseStereoConfig:
    with path.open() as stream:
        data = yaml.safe_load(stream)["dense_stereo_feasibility"]
    candidates = tuple(SGBMCandidate(**entry) for entry in data["sgbm_candidates"])
    fast_candidates = tuple(
        FastFoundationCandidate(**entry) for entry in data["fast_candidates"])
    config = DenseStereoConfig(
        processing_scale=float(data["processing_scale"]),
        processing_height_multiple=int(data["processing_height_multiple"]),
        tag_inpaint_dilate_px=int(data["tag_inpaint_dilate_px"]),
        tag_inpaint_radius_px=int(data["tag_inpaint_radius_px"]),
        static_mean_window_s=float(data["static_mean_window_s"]),
        held_out_fraction=float(data["held_out_fraction"]),
        min_static_window_frames=int(data["min_static_window_frames"]),
        development_frame_stride=int(data["development_frame_stride"]),
        disparity_margin_px=int(data["disparity_margin_px"]),
        lr_max_error_px=float(data["lr_max_error_px"]),
        point_sample_count=int(data["point_sample_count"]),
        voxel_size_m=float(data["voxel_size_m"]),
        workspace_z_m=tuple(float(v) for v in data["workspace_z_m"]),
        depth_mad_scale=float(data["depth_mad_scale"]),
        depth_mad_floor_m=float(data["depth_mad_floor_m"]),
        min_mask_area_px=int(data["min_mask_area_px"]),
        sgbm_candidates=candidates,
        fast_max_disp_multiple=int(data["fast_max_disp_multiple"]),
        fast_candidates=fast_candidates,
    )
    assert 0.0 < config.processing_scale <= 1.0
    assert config.processing_height_multiple > 0
    assert 0.0 < config.held_out_fraction < 0.5
    assert config.min_static_window_frames >= 2
    assert config.development_frame_stride >= 1
    assert config.disparity_margin_px >= 0
    assert config.lr_max_error_px > 0.0
    assert config.point_sample_count > 0 and config.voxel_size_m > 0.0
    assert config.workspace_z_m[0] < config.workspace_z_m[1]
    assert config.depth_mad_scale > 0.0 and config.depth_mad_floor_m > 0.0
    assert config.min_mask_area_px > 0
    assert candidates
    for candidate in candidates:
        assert candidate.block_size >= 3 and candidate.block_size % 2 == 1
    assert config.fast_max_disp_multiple > 0
    assert fast_candidates
    for candidate in fast_candidates:
        assert candidate.checkpoint and candidate.valid_iters > 0
    return config


def calibration_fingerprint(path: Path = CALIBRATION_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocessing_key(config: DenseStereoConfig) -> str:
    payload = {
        "processing_scale": config.processing_scale,
        "processing_height_multiple": config.processing_height_multiple,
        "tag_inpaint_dilate_px": config.tag_inpaint_dilate_px,
        "tag_inpaint_radius_px": config.tag_inpaint_radius_px,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def sgbm_backend_key(config: DenseStereoConfig,
                     candidate: SGBMCandidate) -> str:
    payload = {
        "candidate": asdict(candidate),
        "preprocessing": preprocessing_key(config),
        "lr_max_error_px": config.lr_max_error_px,
        "voxel_size_m": config.voxel_size_m,
        "workspace_z_m": config.workspace_z_m,
        "depth_mad_scale": config.depth_mad_scale,
        "depth_mad_floor_m": config.depth_mad_floor_m,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def fast_foundation_backend_key(config: DenseStereoConfig,
                                candidate: FastFoundationCandidate,
                                max_disparity: int) -> str:
    payload = {
        "candidate": asdict(candidate),
        "max_disparity": max_disparity,
        "preprocessing": preprocessing_key(config),
        "lr_max_error_px": config.lr_max_error_px,
        "voxel_size_m": config.voxel_size_m,
        "workspace_z_m": config.workspace_z_m,
        "depth_mad_scale": config.depth_mad_scale,
        "depth_mad_floor_m": config.depth_mad_floor_m,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _processed_geometry(rectification: StereoRectification,
                        config: DenseStereoConfig) -> ProcessingGeometry:
    width, height = rectification.image_size
    scaled_width = int(round(width * config.processing_scale))
    scaled_height = int(round(height * config.processing_scale))
    multiple = config.processing_height_multiple
    output_height = ((scaled_height + multiple - 1) // multiple) * multiple
    pad_top = (output_height - scaled_height) // 2

    P_main = rectification.projections["main"]
    P_aux = rectification.projections["aux"]
    scale = config.processing_scale
    fx = float(P_main[0, 0]) * scale
    cx_main = float(P_main[0, 2]) * scale
    cx_aux = float(P_aux[0, 2]) * scale
    cy = float(P_main[1, 2]) * scale + pad_top
    tx = float(P_aux[0, 3] / P_aux[0, 0])
    Q = np.array([
        [1.0, 0.0, 0.0, -cx_main],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, fx],
        [0.0, 0.0, -1.0 / tx, (cx_main - cx_aux) / tx],
    ], dtype=np.float64)
    return ProcessingGeometry((scaled_width, output_height), pad_top, Q)


class StereoPreprocessor:
    """Accepted calibration -> deterministic rectified processing tensors."""

    def __init__(self, calibration_path: Path = CALIBRATION_PATH,
                 config: DenseStereoConfig | None = None):
        self.calibration_path = Path(calibration_path)
        self.config = load_config() if config is None else config
        self.rectification = load_stereo_rectification(self.calibration_path)
        self.geometry = _processed_geometry(self.rectification, self.config)
        self.maps = {}
        for name in CAMERA_NAMES:
            matrix, distortion = load_intrinsics(intrinsics_path(name))
            self.maps[name] = cv2.initUndistortRectifyMap(
                matrix, distortion, self.rectification.rotations[name],
                self.rectification.projections[name],
                self.rectification.image_size, cv2.CV_32FC1)

    def _resize_pad(self, image: np.ndarray, interpolation: int,
                    border_value=0) -> np.ndarray:
        width, output_height = self.geometry.image_size
        scaled_height = int(round(self.rectification.image_size[1]
                                  * self.config.processing_scale))
        resized = cv2.resize(image, (width, scaled_height), interpolation=interpolation)
        bottom = output_height - scaled_height - self.geometry.pad_top
        return cv2.copyMakeBorder(
            resized, self.geometry.pad_top, bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=border_value)

    def rectify_image(self, name: str, image: np.ndarray) -> np.ndarray:
        expected = self.rectification.image_size[::-1]
        if image.shape[:2] != expected:
            raise RuntimeError(
                f"{name} frame size {image.shape[1]}x{image.shape[0]} does not "
                f"match stereo calibration {expected[1]}x{expected[0]}")
        rectified = cv2.remap(image, *self.maps[name], cv2.INTER_LINEAR)
        return self._resize_pad(rectified, cv2.INTER_AREA)

    def rectify_mask(self, name: str, mask: np.ndarray) -> np.ndarray:
        expected = self.rectification.image_size[::-1]
        if mask.shape != expected:
            raise RuntimeError(
                f"{name} mask size {mask.shape[::-1]} does not match stereo "
                f"calibration {expected[::-1]}")
        rectified = cv2.remap(mask.astype(np.uint8), *self.maps[name],
                             cv2.INTER_NEAREST)
        return self._resize_pad(rectified, cv2.INTER_NEAREST) > 0


def inpaint_sponge_tags(image: np.ndarray, tags: dict,
                        dilate_px: int, radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove recorded sponge tags while leaving table anchors untouched."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for tag_id, observation in tags.items():
        if int(tag_id) not in SPONGE_TAG_IDS:
            continue
        polygon = np.rint(np.asarray(observation["corners"])).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
    if dilate_px > 0 and mask.any():
        size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    if not mask.any():
        return image.copy(), mask.astype(bool)
    return cv2.inpaint(image, mask, radius_px, cv2.INPAINT_TELEA), mask > 0


def disparity_range_from_depths(depths_m: np.ndarray,
                                rectification: StereoRectification,
                                config: DenseStereoConfig) -> tuple[int, int]:
    """Derive an SGBM range from rectified workspace depths, with margin."""
    depths = np.asarray(depths_m, dtype=np.float64)
    depths = depths[np.isfinite(depths) & (depths > 0.0)]
    if depths.size == 0:
        raise RuntimeError("no positive rectified workspace depths")
    P_main = rectification.projections["main"]
    P_aux = rectification.projections["aux"]
    scale = config.processing_scale
    focal_baseline = -float(P_aux[0, 3]) * scale
    principal_delta = float(P_main[0, 2] - P_aux[0, 2]) * scale
    disparities = focal_baseline / depths + principal_delta
    low = int(np.floor(disparities.min())) - config.disparity_margin_px
    high = int(np.ceil(disparities.max())) + config.disparity_margin_px
    minimum = max(0, low)
    count = int(np.ceil((high - minimum + 1) / 16.0)) * 16
    if count <= 0:
        raise RuntimeError(f"invalid derived disparity interval [{minimum}, {high}]")
    return minimum, count


def make_sgbm(candidate: SGBMCandidate, min_disparity: int,
              num_disparities: int, right: bool = False):
    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError("num_disparities must be a positive multiple of 16")
    minimum = -(min_disparity + num_disparities) if right else min_disparity
    block = candidate.block_size
    return cv2.StereoSGBM.create(
        minDisparity=minimum,
        numDisparities=num_disparities,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=candidate.disp12_max_diff,
        uniquenessRatio=candidate.uniqueness_ratio,
        speckleWindowSize=candidate.speckle_window_size,
        speckleRange=candidate.speckle_range,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def sgbm_disparities(left_bgr: np.ndarray, right_bgr: np.ndarray,
                     candidate: SGBMCandidate, min_disparity: int,
                     num_disparities: int) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    forward = make_sgbm(candidate, min_disparity, num_disparities)
    reverse = make_sgbm(candidate, min_disparity, num_disparities, right=True)
    left_disparity = forward.compute(left, right).astype(np.float32) / 16.0
    right_disparity = reverse.compute(right, left).astype(np.float32) / 16.0
    return left_disparity, right_disparity


def left_right_validity(left_disparity: np.ndarray,
                        right_disparity: np.ndarray,
                        max_error_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validity, residual and corresponding right-image x coordinate."""
    height, width = left_disparity.shape
    yy, xx = np.indices((height, width))
    right_x_float = xx.astype(np.float32) - left_disparity
    right_x = np.rint(right_x_float).astype(np.int32)
    inside = (right_x >= 0) & (right_x < width) & (left_disparity > 0.0)
    sampled = np.full_like(left_disparity, np.nan)
    sampled[inside] = right_disparity[yy[inside], right_x[inside]]
    residual = np.abs(left_disparity + sampled)
    valid = inside & np.isfinite(residual) & (residual <= max_error_px)
    return valid, residual, right_x


def T_base_rectified_main(T_base_main: np.ndarray,
                          rectification: StereoRectification) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rectification.rotations["main"].T
    return np.asarray(T_base_main, dtype=np.float64) @ transform


def disparity_to_cloud(left_disparity: np.ndarray,
                       right_disparity: np.ndarray,
                       left_mask: np.ndarray,
                       right_mask: np.ndarray,
                       Q: np.ndarray,
                       T_base_rectified: np.ndarray,
                       config: DenseStereoConfig,
                       workspace_xy: tuple[np.ndarray, np.ndarray]) -> CloudResult:
    valid_lr, residual, right_x = left_right_validity(
        left_disparity, right_disparity, config.lr_max_error_px)
    yy, _ = np.indices(left_disparity.shape)
    right_hit = np.zeros_like(left_mask, dtype=bool)
    inside = (right_x >= 0) & (right_x < right_mask.shape[1])
    right_hit[inside] = right_mask[yy[inside], right_x[inside]]
    candidate = left_mask & (left_disparity > 0.0)
    valid = candidate & right_hit & valid_lr
    left_pixels = int(left_mask.sum())
    rejected = int(np.count_nonzero(candidate & ~(right_hit & valid_lr)))
    if not valid.any():
        return CloudResult(np.empty((0, 3)), np.empty(0), left_pixels, rejected)

    points_rectified = cv2.reprojectImageTo3D(left_disparity, Q)[valid]
    finite = np.isfinite(points_rectified).all(axis=1)
    points_rectified = points_rectified[finite]
    depths = points_rectified[:, 2]
    lr_residual = residual[valid][finite]
    points_base = (points_rectified @ T_base_rectified[:3, :3].T
                   + T_base_rectified[:3, 3])
    low_xy, high_xy = workspace_xy
    keep = ((points_base[:, 0] >= low_xy[0])
            & (points_base[:, 0] <= high_xy[0])
            & (points_base[:, 1] >= low_xy[1])
            & (points_base[:, 1] <= high_xy[1])
            & (points_base[:, 2] >= config.workspace_z_m[0])
            & (points_base[:, 2] <= config.workspace_z_m[1]))
    points_base = points_base[keep]
    depths = depths[keep]
    lr_residual = lr_residual[keep]
    if points_base.shape[0] == 0:
        return CloudResult(points_base, np.empty(0), left_pixels, rejected)

    median = float(np.median(depths))
    mad = float(np.median(np.abs(depths - median)))
    threshold = max(config.depth_mad_scale * 1.4826 * mad,
                    config.depth_mad_floor_m)
    depth_keep = np.abs(depths - median) <= threshold
    points_base = points_base[depth_keep]
    lr_residual = lr_residual[depth_keep]
    confidence = np.clip(1.0 - lr_residual / config.lr_max_error_px, 0.0, 1.0)
    return CloudResult(points_base, confidence.astype(np.float32),
                       left_pixels, rejected)


def voxel_downsample(points: np.ndarray, confidence: np.ndarray,
                     voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if points.shape[0] != confidence.shape[0]:
        raise ValueError("points and confidence lengths differ")
    if points.shape[0] == 0:
        return points, confidence
    voxels = np.floor(points / voxel_size_m).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    first.sort()
    return points[first], confidence[first]


def sample_cloud(points: np.ndarray, confidence: np.ndarray, count: int):
    """Deterministically sample/pad the Stage 2 cloud tensor contract."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    output = np.zeros((count, 5), dtype=np.float32)
    if points.shape[0] == 0:
        return output, np.zeros(3, dtype=np.float64)
    center = np.median(points, axis=0)
    if points.shape[0] <= count:
        chosen = np.arange(points.shape[0])
    else:
        chosen = np.linspace(0, points.shape[0] - 1, count).round().astype(int)
    n = chosen.size
    output[:n, :3] = (points[chosen] - center).astype(np.float32)
    output[:n, 3] = confidence[chosen]
    output[:n, 4] = 1.0
    return output, center
