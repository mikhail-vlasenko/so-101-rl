"""Mechanically align the rigid C922 stereo pair before checkerboard calibration.

Each camera observes the fixed two-tag table board, which gives its live pose in
the arm base frame. A rolling window then checks the contracts dense rectified
stereo needs before calibration: horizontal baseline, matched camera height,
nearly parallel optical frames, stable anchoring, shared configured-workspace
coverage and border margin for the later rectification crop.

The optional viewer draws the configured lift/pickplace cube workspace as a 3-D
wire box in both raw images. Green means that camera has enough margin; red and
the per-camera instruction say which way to aim it. The header reports the
auxiliary camera's pitch/yaw/roll relative to main, so all three should converge
to zero while the mount is adjusted. Measurements use only the most recent
window, so stale poses age out after a camera is moved.

This is a pre-calibration mechanical check. Passing it does not replace shared
checkerboard stereo calibration or the rectified vertical-residual acceptance
test in `.claude/plans/dense_stereo_pointnet.md`.

Run:
    conda run -n mujoco_env python -m real.calib.align_stereo_rig
    conda run -n mujoco_env python -m real.calib.align_stereo_rig --gui

With ``--gui``, q/Esc evaluates the current rolling window and exits. Without a
viewer, the configured finite capture is evaluated directly. A failed gate
raises with every required adjustment rather than silently returning success.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from real.calib.extrinsics import (
    average_transforms,
    mat_inv,
    transform_spread,
)
from real.calib.calibrate_stereo import (
    load_limits as load_stereo_calibration_limits,
    load_stereo_rectification,
    save_anchor_reference,
)
from real.calib.table_anchor import TableAnchorTracker, load_table_anchor_limits
from real.marker_spec import TABLE_TAG_IDS
from real.vision.detect import make_detector
from real.vision.overlay import (
    GREEN,
    RED,
    TABLE_BLUE,
    OverlayLine,
    OverlaySpan,
    StereoViewer,
    TagStyle,
    annotate_tags,
)
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"
ENV_CONFIG_PATHS = (
    REPO_ROOT / "conf" / "env" / "lift.yaml",
    REPO_ROOT / "conf" / "env" / "pickplace.yaml",
)

MARGIN_NAMES = ("left", "right", "top", "bottom")
RELATIVE_AXIS_NAMES = ("pitch", "yaw", "roll")
BOX_EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)


@dataclass(frozen=True)
class AlignmentLimits:
    capture_frames: int
    sample_window: int
    min_samples: int
    baseline_min_m: float
    baseline_max_m: float
    max_height_delta_m: float
    max_relative_axis_deg: float
    min_workspace_margin_px: float
    workspace_z_low_m: float
    workspace_z_high_m: float
    min_pair_detection_fraction: float
    max_translation_spread_mean_mm: float
    max_rotation_spread_mean_deg: float


@dataclass(frozen=True)
class CameraCoverage:
    pixels: np.ndarray
    margins_px: np.ndarray
    all_in_front: bool


@dataclass(frozen=True)
class AlignmentReport:
    baseline_m: float
    aux_minus_main_height_m: float
    relative_euler_deg: np.ndarray
    coverage: dict[str, CameraCoverage]
    spread: dict[str, tuple[float, float]]
    pair_detection_fraction: float
    sample_count: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def load_alignment_limits(path: Path = CONFIG_PATH) -> AlignmentLimits:
    """Load every acceptance threshold from the Hydra config source of truth."""
    with path.open() as stream:
        cfg = yaml.safe_load(stream)["stereo_alignment"]
    limits = AlignmentLimits(
        capture_frames=int(cfg["capture_frames"]),
        sample_window=int(cfg["sample_window"]),
        min_samples=int(cfg["min_samples"]),
        baseline_min_m=float(cfg["baseline_min_m"]),
        baseline_max_m=float(cfg["baseline_max_m"]),
        max_height_delta_m=float(cfg["max_height_delta_m"]),
        max_relative_axis_deg=float(cfg["max_relative_axis_deg"]),
        min_workspace_margin_px=float(cfg["min_workspace_margin_px"]),
        workspace_z_low_m=float(cfg["workspace_z_low_m"]),
        workspace_z_high_m=float(cfg["workspace_z_high_m"]),
        min_pair_detection_fraction=float(cfg["min_pair_detection_fraction"]),
        max_translation_spread_mean_mm=float(
            cfg["max_translation_spread_mean_mm"]),
        max_rotation_spread_mean_deg=float(
            cfg["max_rotation_spread_mean_deg"]),
    )
    assert 0 < limits.min_samples <= limits.sample_window
    assert 0.0 < limits.baseline_min_m < limits.baseline_max_m
    assert limits.workspace_z_low_m < limits.workspace_z_high_m
    assert 0.0 < limits.min_pair_detection_fraction <= 1.0
    return limits


def load_workspace_corners(limits: AlignmentLimits) -> np.ndarray:
    """Eight base-frame corners of the union of configured task spawn boxes."""
    lows, highs = [], []
    for path in ENV_CONFIG_PATHS:
        with path.open() as stream:
            cfg = yaml.safe_load(stream)
        task_key = "lift_env" if path.stem == "lift" else "pickplace_env"
        task = cfg[task_key]
        lows.append(np.asarray(task["cube_low"], dtype=np.float64))
        highs.append(np.asarray(task["cube_high"], dtype=np.float64))
    low_xy = np.min(np.stack(lows), axis=0)
    high_xy = np.max(np.stack(highs), axis=0)
    return np.array([
        [x, y, z]
        for x in (low_xy[0], high_xy[0])
        for y in (low_xy[1], high_xy[1])
        for z in (limits.workspace_z_low_m, limits.workspace_z_high_m)
    ], dtype=np.float64)


def project_workspace(points_base: np.ndarray, T_base_cam: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                      image_size: tuple[int, int]) -> CameraCoverage:
    """Project base-frame workspace points and return L/R/T/B image margins."""
    T_cam_base = mat_inv(T_base_cam)
    rvec, _ = cv2.Rodrigues(T_cam_base[:3, :3])
    tvec = T_cam_base[:3, 3]
    pixels, _ = cv2.projectPoints(
        np.asarray(points_base, dtype=np.float64), rvec, tvec,
        camera_matrix, dist_coeffs)
    pixels = pixels.reshape(-1, 2)
    points_cam = (
        np.asarray(points_base, dtype=np.float64) @ T_cam_base[:3, :3].T
        + tvec
    )
    width, height = image_size
    margins = np.array([
        pixels[:, 0].min(),
        width - 1 - pixels[:, 0].max(),
        pixels[:, 1].min(),
        height - 1 - pixels[:, 1].max(),
    ])
    return CameraCoverage(pixels, margins, bool(np.all(points_cam[:, 2] > 0.0)))


def project_rectified_workspace(
        points_base: np.ndarray, T_base_cam: np.ndarray,
        rectification_rotation: np.ndarray, projection_matrix: np.ndarray,
        valid_roi: tuple[int, int, int, int]) -> CameraCoverage:
    """Project base points into a rectified image and measure valid-ROI margins."""
    T_cam_base = mat_inv(T_base_cam)
    points_cam = (
        np.asarray(points_base, dtype=np.float64) @ T_cam_base[:3, :3].T
        + T_cam_base[:3, 3]
    )
    points_rectified = points_cam @ rectification_rotation.T
    normalized = points_rectified[:, :2] / points_rectified[:, 2, None]
    K_rectified = projection_matrix[:, :3]
    pixels = np.column_stack([
        K_rectified[0, 0] * normalized[:, 0] + K_rectified[0, 2],
        K_rectified[1, 1] * normalized[:, 1] + K_rectified[1, 2],
    ])
    x, y, width, height = valid_roi
    margins = np.array([
        pixels[:, 0].min() - x,
        x + width - 1 - pixels[:, 0].max(),
        pixels[:, 1].min() - y,
        y + height - 1 - pixels[:, 1].max(),
    ])
    return CameraCoverage(
        pixels, margins, bool(np.all(points_rectified[:, 2] > 0.0)))


def relative_pose_change(T_current: np.ndarray,
                         T_reference: np.ndarray) -> tuple[float, float]:
    """Return translation-mm and rotation-deg change from a known-good pose."""
    delta = T_current @ mat_inv(T_reference)
    translation_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
    rotation_deg = float(np.linalg.norm(
        Rotation.from_matrix(delta[:3, :3]).as_rotvec()) * 180.0 / np.pi)
    return translation_mm, rotation_deg


def camera_movement_warning(
        movement_mm: float, movement_deg: float,
        translation_limit_mm: float, rotation_limit_deg: float) -> str | None:
    """Return the operator instruction when tag poses show calibration drift."""
    if movement_mm <= translation_limit_mm and movement_deg <= rotation_limit_deg:
        return None
    return (
        "WARNING: cameras moved relative to their stereo calibration "
        f"({movement_mm:.2f} mm / {movement_deg:.3f} deg). Lay the "
        "checkerboard flat in both views and rerun "
        "`python -m real.calib.calibrate_stereo` before dense stereo."
    )


def evaluate_alignment(
        transforms: dict[str, np.ndarray],
        coverage: dict[str, CameraCoverage],
        spread: dict[str, tuple[float, float]],
        pair_detection_fraction: float,
        sample_count: int,
        limits: AlignmentLimits) -> AlignmentReport:
    """Evaluate measured geometry against every mechanical acceptance gate."""
    T_main = transforms["main"]
    T_aux = transforms["aux"]
    baseline_m = float(np.linalg.norm(T_aux[:3, 3] - T_main[:3, 3]))
    height_delta_m = float(T_aux[2, 3] - T_main[2, 3])
    R_main_aux = T_main[:3, :3].T @ T_aux[:3, :3]
    relative_euler_deg = Rotation.from_matrix(R_main_aux).as_euler(
        "xyz", degrees=True)

    failures = []
    if not limits.baseline_min_m <= baseline_m <= limits.baseline_max_m:
        failures.append(
            f"baseline {baseline_m * 1000.0:.1f} mm is outside "
            f"{limits.baseline_min_m * 1000.0:.0f}-"
            f"{limits.baseline_max_m * 1000.0:.0f} mm")
    if abs(height_delta_m) > limits.max_height_delta_m:
        action = "lower" if height_delta_m > 0.0 else "raise"
        failures.append(
            f"{action} aux: height delta is {height_delta_m * 1000.0:+.1f} mm "
            f"(limit +/-{limits.max_height_delta_m * 1000.0:.0f} mm)")
    for name, value in zip(RELATIVE_AXIS_NAMES, relative_euler_deg):
        if abs(value) > limits.max_relative_axis_deg:
            failures.append(
                f"reduce aux-vs-main {name} {value:+.1f} deg toward 0 "
                f"(limit +/-{limits.max_relative_axis_deg:.1f} deg)")

    for camera in CAMERA_NAMES:
        camera_coverage = coverage[camera]
        if not camera_coverage.all_in_front:
            failures.append(f"{camera}: workspace extends behind the camera")
        for margin_name, value in zip(MARGIN_NAMES, camera_coverage.margins_px):
            if value < limits.min_workspace_margin_px:
                failures.append(
                    f"{camera}: {margin_name} workspace margin {value:.0f} px "
                    f"is below {limits.min_workspace_margin_px:.0f} px")
        translation_mm, rotation_deg = spread[camera]
        if translation_mm > limits.max_translation_spread_mean_mm:
            failures.append(
                f"{camera}: table-anchor translation spread {translation_mm:.2f} mm "
                f"exceeds {limits.max_translation_spread_mean_mm:.2f} mm")
        if rotation_deg > limits.max_rotation_spread_mean_deg:
            failures.append(
                f"{camera}: table-anchor rotation spread {rotation_deg:.3f} deg "
                f"exceeds {limits.max_rotation_spread_mean_deg:.3f} deg")

    if pair_detection_fraction < limits.min_pair_detection_fraction:
        failures.append(
            f"paired two-tag-board visibility {pair_detection_fraction:.0%} is below "
            f"{limits.min_pair_detection_fraction:.0%}")
    if sample_count < limits.min_samples:
        failures.append(
            f"only {sample_count} paired pose samples; need {limits.min_samples}")

    return AlignmentReport(
        baseline_m=baseline_m,
        aux_minus_main_height_m=height_delta_m,
        relative_euler_deg=relative_euler_deg,
        coverage=coverage,
        spread=spread,
        pair_detection_fraction=pair_detection_fraction,
        sample_count=sample_count,
        failures=tuple(failures),
    )


def coverage_guidance(coverage: CameraCoverage, minimum_margin_px: float,
                      image_size: tuple[int, int]) -> str:
    """Turn failed border margins into a concise physical aiming instruction."""
    left, right, top, bottom = coverage.margins_px
    width, height = image_size
    horizontal_span = width - 1 - left - right
    vertical_span = height - 1 - top - bottom
    actions = []
    if horizontal_span > width - 1 - 2.0 * minimum_margin_px:
        actions.append("move back: workspace too wide")
    elif left < minimum_margin_px:
        actions.append("aim left")
    elif right < minimum_margin_px:
        actions.append("aim right")
    if vertical_span > height - 1 - 2.0 * minimum_margin_px:
        actions.append("move back: workspace too tall")
    elif top < minimum_margin_px:
        actions.append("tilt up")
    elif bottom < minimum_margin_px:
        actions.append("tilt down")
    return ", ".join(actions) if actions else "workspace margin OK"


def format_report(report: AlignmentReport, limits: AlignmentLimits) -> str:
    """Human-readable result shared by terminal output and failure traceback."""
    pitch, yaw, roll = report.relative_euler_deg
    lines = [
        f"STEREO ALIGNMENT: {'PASS' if report.passed else 'ADJUST'}",
        f"  paired samples {report.sample_count}, recent visibility "
        f"{report.pair_detection_fraction:.0%}",
        f"  baseline {report.baseline_m * 1000.0:.1f} mm; "
        f"aux-main height {report.aux_minus_main_height_m * 1000.0:+.1f} mm",
        f"  aux vs main: pitch {pitch:+.2f} deg, yaw {yaw:+.2f} deg, "
        f"roll {roll:+.2f} deg",
    ]
    for camera in CAMERA_NAMES:
        margins = "/".join(f"{value:.0f}" for value in report.coverage[camera].margins_px)
        trans_mm, rot_deg = report.spread[camera]
        lines.append(
            f"  {camera}: margins L/R/T/B {margins} px; anchor spread "
            f"{trans_mm:.2f} mm / {rot_deg:.3f} deg")
    if report.failures:
        lines.append("required adjustments:")
        lines.extend(f"  - {failure}" for failure in report.failures)
    else:
        lines.append(
            f"  all raw-image margins >= {limits.min_workspace_margin_px:.0f} px; "
            "ready for checkerboard stereo calibration")
    return "\n".join(lines)


def _draw_workspace(frame: np.ndarray, coverage: CameraCoverage,
                    passed: bool) -> np.ndarray:
    view = frame.copy()
    color = GREEN if passed else RED
    points = np.rint(coverage.pixels).astype(np.int32)
    for a, b in BOX_EDGES:
        cv2.line(view, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(view, tuple(point), 4, color, -1, cv2.LINE_AA)
    return view


def _current_measurement(measurements: deque,
                         mats: dict[str, np.ndarray],
                         dists: dict[str, np.ndarray],
                         image_sizes: dict[str, tuple[int, int]],
                         workspace: np.ndarray,
                         limits: AlignmentLimits) -> tuple[
                             dict[str, np.ndarray], AlignmentReport] | None:
    samples = [measurement for measurement in measurements
               if measurement is not None]
    if not samples:
        return None
    transforms = {
        camera: average_transforms([sample[camera] for sample in samples])
        for camera in CAMERA_NAMES
    }
    coverage = {
        camera: project_workspace(
            workspace, transforms[camera], mats[camera], dists[camera],
            image_sizes[camera])
        for camera in CAMERA_NAMES
    }
    spread = {}
    for camera in CAMERA_NAMES:
        trans_mm, rot_deg = transform_spread(
            [sample[camera] for sample in samples], transforms[camera])
        spread[camera] = (float(trans_mm.mean()), float(rot_deg.mean()))
    report = evaluate_alignment(
        transforms, coverage, spread,
        float(np.mean([measurement is not None for measurement in measurements])),
        len(samples), limits)
    return transforms, report


def _viewer_lines(report: AlignmentReport | None,
                  limits: AlignmentLimits,
                  image_sizes: dict[str, tuple[int, int]],
                  table_visible: dict[str, bool]) -> tuple[
                      list[OverlayLine], dict[str, list[OverlayLine]]]:
    if report is None:
        return [
            OverlayLine("collecting two-tag board poses", TABLE_BLUE),
            OverlayLine("keep both complete anchors visible in both views", TABLE_BLUE),
        ], {
            camera: [OverlayLine(
                f"{camera}: " + ("both table tags visible" if table_visible[camera] else
                                  "ANCHOR PAIR INCOMPLETE - show both full tags"),
                GREEN if table_visible[camera] else RED)]
            for camera in CAMERA_NAMES
        }
    status_color = GREEN if report.passed else RED
    pitch, yaw, roll = report.relative_euler_deg
    baseline_ok = limits.baseline_min_m <= report.baseline_m <= limits.baseline_max_m
    height_ok = abs(report.aux_minus_main_height_m) <= limits.max_height_delta_m
    pitch_ok = abs(pitch) <= limits.max_relative_axis_deg
    yaw_ok = abs(yaw) <= limits.max_relative_axis_deg
    roll_ok = abs(roll) <= limits.max_relative_axis_deg
    header = [
        OverlayLine("PASS - ready to calibrate" if report.passed else
                    "ADJUST - all measurements must turn green", status_color),
        OverlayLine(spans=(
            OverlaySpan(f"baseline {report.baseline_m * 1000.0:.1f} mm  ",
                        GREEN if baseline_ok else RED),
            OverlaySpan(
                f"height {report.aux_minus_main_height_m * 1000.0:+.1f} mm  ",
                GREEN if height_ok else RED),
            OverlaySpan(f"P {pitch:+.1f}  ", GREEN if pitch_ok else RED),
            OverlaySpan(f"Y {yaw:+.1f}  ", GREEN if yaw_ok else RED),
            OverlaySpan(f"R {roll:+.1f} deg", GREEN if roll_ok else RED),
        )),
    ]
    camera_lines = {}
    for camera in CAMERA_NAMES:
        camera_coverage = report.coverage[camera]
        margin_ok = (
            camera_coverage.all_in_front
            and np.all(camera_coverage.margins_px >= limits.min_workspace_margin_px)
        )
        margins = "/".join(f"{value:.0f}" for value in camera_coverage.margins_px)
        camera_lines[camera] = [
            OverlayLine(
                "both table anchors visible" if table_visible[camera] else
                "ANCHOR PAIR INCOMPLETE - show both full tags",
                GREEN if table_visible[camera] else RED),
            OverlayLine(f"{camera} workspace L/R/T/B {margins} px",
                        GREEN if margin_ok else RED),
            OverlayLine(coverage_guidance(
                camera_coverage, limits.min_workspace_margin_px,
                image_sizes[camera]), GREEN if margin_ok else RED),
        ]
    return header, camera_lines


def main() -> None:
    limits = load_alignment_limits()
    parser = argparse.ArgumentParser(
        description="Check and live-align the rigid C922 stereo mount")
    parser.add_argument(
        "--frames", type=int, default=None,
        help=f"capture loops without GUI (default {limits.capture_frames}); "
             "with GUI, omit or use 0 to run until q")
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--gui", action="store_true",
                        help="live alignment overlay; q/Esc evaluates and exits")
    parser.add_argument("--save-frames", default=None,
                        help="directory for the final annotated frame pair")
    parser.add_argument(
        "--stereo-calibration", type=Path, default=None,
        help="also validate rectified workspace coverage against this YAML")
    parser.add_argument(
        "--record-stereo-anchor-reference", action="store_true",
        help="record the current two-tag relative pose as known-good after calibration")
    args = parser.parse_args()

    frame_limit = args.frames
    if frame_limit is None:
        frame_limit = 0 if args.gui else limits.capture_frames
    if frame_limit < 0 or (frame_limit == 0 and not args.gui):
        parser.error("--frames must be positive, or 0 only with --gui")
    if args.record_stereo_anchor_reference and args.stereo_calibration is None:
        parser.error("--record-stereo-anchor-reference requires --stereo-calibration")

    workspace = load_workspace_corners(limits)
    detector = make_detector(args.family)
    caps, mats, dists, anchors = {}, {}, {}, {}
    raw_anchor_limits = replace(load_table_anchor_limits(), ema_alpha=1.0)
    for camera in CAMERA_NAMES:
        caps[camera], mats[camera], dists[camera] = open_rig_camera(camera)
        anchors[camera] = TableAnchorTracker(
            mats[camera], dists[camera], limits=raw_anchor_limits)

    measurements = deque(maxlen=limits.sample_window)
    seen_counts = {camera: 0 for camera in CAMERA_NAMES}
    viewer = StereoViewer("stereo rig alignment") if args.gui else None
    last_views = {}
    image_sizes = {}
    loops = 0
    measurement = None
    try:
        while frame_limit == 0 or loops < frame_limit:
            frames, poses, table_dets = {}, {}, {}
            for camera in CAMERA_NAMES:
                ok, frame = caps[camera].read()
                if not ok:
                    raise RuntimeError(f"camera read failed on '{camera}'")
                frames[camera] = frame
                image_sizes[camera] = (frame.shape[1], frame.shape[0])
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                by_id = {d.id: d for d in detector.detect(gray)}
                table_dets[camera] = {
                    tag: by_id[tag] for tag in TABLE_TAG_IDS if tag in by_id}
                if anchors[camera].observe(table_dets[camera]):
                    seen_counts[camera] += 1
                    poses[camera] = anchors[camera].value()
            paired = len(poses) == len(CAMERA_NAMES)
            measurements.append(poses if paired else None)
            loops += 1

            measurement = _current_measurement(
                measurements, mats, dists, image_sizes, workspace, limits)
            report = None if measurement is None else measurement[1]
            table_visible = {
                camera: len(table_dets[camera]) == len(TABLE_TAG_IDS)
                for camera in CAMERA_NAMES
            }
            header, camera_lines = _viewer_lines(
                report, limits, image_sizes, table_visible)
            views = {}
            for camera in CAMERA_NAMES:
                if report is None:
                    view = frames[camera].copy()
                else:
                    margins_ok = (
                        report.coverage[camera].all_in_front
                        and np.all(report.coverage[camera].margins_px
                                   >= limits.min_workspace_margin_px)
                    )
                    view = _draw_workspace(
                        frames[camera], report.coverage[camera], margins_ok)
                annotate_tags(view, table_dets[camera].values(), {
                    tag: TagStyle(f"table anchor {tag}", TABLE_BLUE)
                    for tag in TABLE_TAG_IDS
                })
                views[camera] = view
            last_views = views

            if viewer is not None:
                key = viewer.show(views, camera_lines, header)
                if key in (27, ord("q")):
                    break
    finally:
        for camera in CAMERA_NAMES:
            caps[camera].release()
        detector.close()
        if viewer is not None:
            viewer.close()

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for camera, view in last_views.items():
            path = os.path.join(args.save_frames, f"stereo_align_{camera}.jpg")
            if not cv2.imwrite(path, view):
                raise RuntimeError(f"failed to write {path}")
            print(f"wrote {path}")

    if measurement is None:
        visibility = ", ".join(
            f"{camera}={seen_counts[camera]}/{loops}"
            for camera in CAMERA_NAMES)
        raise RuntimeError(
            "no paired two-tag-board pose was measured; per-camera acceptances: "
            f"{visibility}. Put both complete table tags inside both frames")
    report = measurement[1]
    summary = format_report(report, limits)
    print(f"\n{summary}", flush=True)
    if not report.passed:
        raise RuntimeError("stereo rig alignment failed; see required adjustments above")

    if args.stereo_calibration is None:
        return
    rectification = load_stereo_rectification(args.stereo_calibration)
    transforms = measurement[0]
    rectified_coverage = {
        camera: project_rectified_workspace(
            workspace, transforms[camera], rectification.rotations[camera],
            rectification.projections[camera], rectification.valid_rois[camera])
        for camera in CAMERA_NAMES
    }
    print("\nRECTIFIED WORKSPACE:")
    failures = []
    for camera in CAMERA_NAMES:
        coverage = rectified_coverage[camera]
        margins = "/".join(f"{value:.0f}" for value in coverage.margins_px)
        print(f"  {camera}: valid-ROI margins L/R/T/B {margins} px")
        if not coverage.all_in_front:
            failures.append(f"{camera}: workspace extends behind rectified camera")
        for margin_name, value in zip(MARGIN_NAMES, coverage.margins_px):
            if value < 0.0:
                failures.append(
                    f"{camera}: rectified {margin_name} workspace margin "
                    f"{value:.0f} px is outside the valid ROI")
    T_aux_main_live = mat_inv(transforms["aux"]) @ transforms["main"]
    delta_translation_mm, delta_rotation_deg = relative_pose_change(
        T_aux_main_live, rectification.T_aux_main)
    print(f"  live anchor vs checkerboard relative pose: "
          f"{delta_translation_mm:.2f} mm / {delta_rotation_deg:.3f} deg")
    if args.record_stereo_anchor_reference:
        save_anchor_reference(args.stereo_calibration, T_aux_main_live)
        rectification = load_stereo_rectification(args.stereo_calibration)
        print(f"  recorded current table-anchor pose in {args.stereo_calibration}")
    reference = rectification.anchor_reference_T_aux_main
    if reference is None:
        print(
            "WARNING: stereo calibration has no table-anchor movement reference. "
            "Remove the checkerboard, expose both table tags, and rerun this check "
            "with --record-stereo-anchor-reference.")
    else:
        movement_mm, movement_deg = relative_pose_change(T_aux_main_live, reference)
        print(f"  camera movement since calibration: "
              f"{movement_mm:.2f} mm / {movement_deg:.3f} deg")
        stereo_limits = load_stereo_calibration_limits()
        warning = camera_movement_warning(
            movement_mm, movement_deg,
            stereo_limits.camera_movement_warning_translation_mm,
            stereo_limits.camera_movement_warning_rotation_deg)
        if warning is not None:
            print(warning)
    if failures:
        raise RuntimeError(
            "rectified workspace coverage failed:\n  - " + "\n  - ".join(failures))


if __name__ == "__main__":
    main()
