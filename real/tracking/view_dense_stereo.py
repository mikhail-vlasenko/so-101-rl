"""Inspect the frozen Stage 2 StereoSGBM result on synchronized dataset frames.

The OpenCV window shows rectified tag-inpainted/raw camera images, disparity,
left-right confidence, and orthographic views of the actual filtered point
cloud against the evaluation-only tag-GT cuboid. An optional MuJoCo window
shows the same cloud and GT box in 3D. This is an offline diagnostic: it never
opens the cameras and the sponge tags never enter stereo inference.

Run:
    conda run --no-capture-output -n mujoco_env python -m \
      real.tracking.view_dense_stereo --dataset datasets/sponge_<stamp>

Controls: A/left and D/right step, Space plays/pauses, Q/Esc quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import yaml

from real.calib.extrinsics import mat_to_pos_quat
from real.tracking.dense_stereo import (
    CALIBRATION_PATH,
    SGBMCandidate,
    StereoPreprocessor,
    left_right_validity,
    load_config,
    sgbm_disparities,
)
from real.tracking.eval_dense_stereo import (
    _cache_root,
    _flatten,
    _load_prepared,
    in_workspace_windows,
    relabel_static,
    run_frame,
    split_windows,
    static_windows,
    visible_windows,
)
from real.tracking.shape_dataset import gt_pose, load_dataset
from real.tracking.record_shapes import load_workspace_bounds


WINDOW_NAME = "frozen StereoSGBM diagnostic"
POINT_RADIUS_M = 0.0015
GT_COLOR = (80, 230, 80)


def make_view_model(half_extents):
    """Minimal table scene for the evaluation-only tag-GT box and cloud."""
    hx, hy, hz = np.asarray(half_extents, dtype=np.float64)
    xml = f"""
    <mujoco model="dense_stereo_comparison">
      <visual><global azimuth="150" elevation="-25"/></visual>
      <worldbody>
        <light pos="0 0 2" dir="0 0 -1" directional="true"/>
        <geom name="table" type="plane" size="1 1 0.02" rgba="0.25 0.3 0.35 1"/>
        <body name="tag_gt" pos="0 0 -1">
          <freejoint name="tag_gt_joint"/>
          <geom name="tag_gt_box" type="box" size="{hx} {hy} {hz}"
                contype="0" conaffinity="0" rgba="0.1 1 0.1 0.22"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tag_gt_joint")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tag_gt_box")
    assert joint >= 0 and geom >= 0
    return model, data, int(model.jnt_qposadr[joint]), geom


def confidence_map(disparity: np.ndarray, reverse: np.ndarray,
                   left_mask: np.ndarray, right_mask: np.ndarray,
                   max_error_px: float) -> np.ndarray:
    """Per-left-pixel deterministic confidence used by the SGBM cloud."""
    valid, residual, right_x = left_right_validity(
        disparity, reverse, max_error_px)
    yy, _ = np.indices(disparity.shape)
    inside = (right_x >= 0) & (right_x < right_mask.shape[1])
    right_hit = np.zeros_like(left_mask, dtype=bool)
    right_hit[inside] = right_mask[yy[inside], right_x[inside]]
    accepted = left_mask & right_hit & valid
    confidence = np.zeros_like(disparity, dtype=np.float32)
    confidence[accepted] = np.clip(
        1.0 - residual[accepted] / max_error_px, 0.0, 1.0)
    return confidence


def _colorize_scalar(values: np.ndarray, low: float, high: float,
                     valid: np.ndarray) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    color = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _annotate(panel: np.ndarray, label: str, detail: str = "") -> np.ndarray:
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)
    if detail:
        size = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.putText(output, detail, (output.shape[1] - size[0] - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1,
                    cv2.LINE_AA)
    return output


def _masked_image(image: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)
    return _annotate(output, label, f"mask {int(mask.sum())} px")


def _orthographic_panel(points: np.ndarray, confidence: np.ndarray,
                        center: np.ndarray, rotation: np.ndarray,
                        half_extents: np.ndarray, axes: tuple[int, int],
                        bounds: tuple[tuple[float, float], tuple[float, float]],
                        size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 24, dtype=np.uint8)

    def pixels(values):
        x = (values[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
        y = (values[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0])
        return np.column_stack((
            np.rint(x * (width - 1)),
            np.rint((1.0 - y) * (height - 1)),
        )).astype(int)

    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    corners = signs * half_extents
    corners = corners @ rotation.T + center
    corner_pixels = pixels(corners[:, axes])
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(signs[first] != signs[second]) == 1:
                cv2.line(panel, tuple(corner_pixels[first]),
                         tuple(corner_pixels[second]), GT_COLOR, 2, cv2.LINE_AA)

    if points.shape[0]:
        point_pixels = pixels(points[:, axes])
        colors = cv2.applyColorMap(
            np.rint(np.clip(confidence, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO).reshape(-1, 3)
        inside = ((point_pixels[:, 0] >= 0) & (point_pixels[:, 0] < width)
                  & (point_pixels[:, 1] >= 0) & (point_pixels[:, 1] < height))
        for pixel, color in zip(point_pixels[inside], colors[inside]):
            cv2.circle(panel, tuple(pixel), 2,
                       tuple(int(value) for value in color), -1, cv2.LINE_AA)
    return _annotate(panel, label, f"{len(points)} filtered points")


def diagnostic_mosaic(images: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                      disparity: np.ndarray, confidence_image: np.ndarray,
                      points: np.ndarray, point_confidence: np.ndarray,
                      center: np.ndarray, rotation: np.ndarray,
                      half_extents: np.ndarray, disparity_range: tuple[int, int],
                      workspace_xy: tuple[np.ndarray, np.ndarray],
                      frame_k: int) -> np.ndarray:
    """Six synchronized diagnostic panels at the processed image size."""
    height, width = disparity.shape
    minimum, count = disparity_range
    disparity_panel = _colorize_scalar(
        disparity, minimum, minimum + count, disparity > 0.0)
    disparity_panel = _annotate(
        disparity_panel, "main disparity", f"frame {frame_k}")
    confidence_panel = _colorize_scalar(
        confidence_image, 0.0, 1.0, confidence_image > 0.0)
    confidence_panel = _annotate(
        confidence_panel, "left-right confidence",
        f"accepted {int(np.count_nonzero(confidence_image))} px")
    low, high = workspace_xy
    xy = _orthographic_panel(
        points, point_confidence, center, rotation, half_extents, (0, 1),
        ((float(low[0]), float(high[0])),
         (float(low[1]), float(high[1]))),
        (width, height), "cloud top view (XY)")
    xz = _orthographic_panel(
        points, point_confidence, center, rotation, half_extents, (0, 2),
        ((float(low[0]), float(high[0])), (-0.005, 0.20)),
        (width, height), "cloud side view (XZ)")
    return np.vstack((
        np.hstack((_masked_image(images["main"], masks["main"], "rectified main"),
                   _masked_image(images["aux"], masks["aux"], "rectified aux"))),
        np.hstack((disparity_panel, confidence_panel)),
        np.hstack((xy, xz)),
    ))


def draw_point_cloud(scene: mujoco.MjvScene, points: np.ndarray,
                     confidence: np.ndarray) -> None:
    """Append as many confidence-colored cloud points as the scene can hold."""
    available = scene.maxgeom - scene.ngeom
    if available <= 0 or points.shape[0] == 0:
        return
    if points.shape[0] > available:
        selected = np.linspace(0, points.shape[0] - 1, available).round().astype(int)
        points = points[selected]
        confidence = confidence[selected]
    radius = np.array([POINT_RADIUS_M, 0.0, 0.0])
    identity = np.eye(3).reshape(-1)
    for point, value in zip(points, confidence):
        rgba = np.array([1.0 - value, 0.35 + 0.55 * value, value, 0.9],
                        dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            radius, np.asarray(point, dtype=np.float64), identity, rgba)
        scene.ngeom += 1


def _viewer_indices(records, root, config, split):
    labels = relabel_static(records, config)
    windows = static_windows(labels, config.min_static_window_frames)
    workspace_xy = load_workspace_bounds()
    windows = in_workspace_windows(records, windows, *workspace_xy)
    development, held_out = split_windows(windows, config.held_out_fraction)
    selected = held_out if split == "held-out" else development
    return _flatten(visible_windows(
        root, records, selected, config.min_mask_area_px)), workspace_xy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--sam2-model", choices=("tiny", "base+"), default="tiny")
    parser.add_argument("--source", choices=("inpainted", "raw"),
                        default="inpainted")
    parser.add_argument("--split", choices=("held-out", "development"),
                        default="held-out")
    parser.add_argument("--frame-k", type=int,
                        help="start at this recorded frame number")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--display-scale", type=float, default=0.65)
    parser.add_argument("--no-mujoco", action="store_true")
    parser.add_argument("--save-frame", type=Path,
                        help="write one six-panel diagnostic and exit")
    args = parser.parse_args()
    if args.fps <= 0.0 or args.display_scale <= 0.0:
        parser.error("--fps and --display-scale must be positive")

    config = load_config()
    records, meta = load_dataset(args.dataset)
    preprocessor = StereoPreprocessor(args.calibration, config)
    root = _cache_root(args.dataset, args.calibration, config, args.sam2_model)
    report_path = root / "sgbm_report.yaml"
    if not report_path.exists():
        raise RuntimeError(
            f"missing {report_path}; run real.tracking.eval_dense_stereo first")
    with report_path.open() as stream:
        report = yaml.safe_load(stream)
    if not report["passes_all_gates"]:
        raise RuntimeError("the frozen StereoSGBM report did not pass its gates")
    candidate = SGBMCandidate(**report["frozen_candidate"])
    minimum = int(report["disparity"]["min"])
    count = int(report["disparity"]["num"])
    indices, workspace_xy = _viewer_indices(
        records, root, config, args.split)
    if not indices:
        raise RuntimeError(f"no visible {args.split} frames")
    position = 0
    if args.frame_k is not None:
        matches = [offset for offset, index in enumerate(indices)
                   if int(records[index]["k"]) == args.frame_k]
        if not matches:
            raise RuntimeError(
                f"frame k={args.frame_k} is not in the visible {args.split} split")
        position = matches[0]

    half_extents = np.asarray(meta["half_extents"], dtype=np.float64)
    model, data, gt_qposadr, gt_geom = make_view_model(half_extents)

    def render(index):
        record = records[index]
        images, masks = _load_prepared(root, record, args.source)
        disparity, reverse = sgbm_disparities(
            images["main"], images["aux"], candidate, minimum, count)
        confidence_image = confidence_map(
            disparity, reverse, masks["main"], masks["aux"],
            config.lr_max_error_px)
        result = run_frame(
            root, record, args.source, candidate, minimum, count,
            preprocessor, workspace_xy)
        center, rotation = gt_pose(record)
        mosaic = diagnostic_mosaic(
            images, masks, disparity, confidence_image, result["points"],
            result["confidence"], center, rotation, half_extents,
            (minimum, count), workspace_xy, int(record["k"]))
        return record, result, center, rotation, mosaic

    if args.save_frame is not None:
        _, _, _, _, mosaic = render(indices[position])
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save_frame), mosaic):
            raise RuntimeError(f"failed to write {args.save_frame}")
        print(f"saved {args.save_frame}")
        return

    camera_view = None
    if not args.no_mujoco:
        camera_view = mujoco.viewer.launch_passive(model, data)
        camera_view.cam.lookat[:] = [0.20, 0.0, 0.03]
        camera_view.cam.distance = 0.55
        camera_view.cam.azimuth = 135.0
        camera_view.cam.elevation = -28.0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    playing = False
    next_frame_at = time.monotonic()
    dirty = True
    try:
        while camera_view is None or camera_view.is_running():
            if dirty:
                record, result, center, rotation, mosaic = render(indices[position])
                T_base_body = np.eye(4)
                T_base_body[:3, :3] = rotation
                T_base_body[:3, 3] = center
                pos, quat = mat_to_pos_quat(T_base_body)
                data.qpos[gt_qposadr:gt_qposadr + 3] = pos
                data.qpos[gt_qposadr + 3:gt_qposadr + 7] = quat
                model.geom_rgba[gt_geom, 3] = 0.22
                mujoco.mj_forward(model, data)
                if camera_view is not None:
                    camera_view.user_scn.ngeom = 0
                    draw_point_cloud(camera_view.user_scn, result["points"],
                                     result["confidence"])
                    camera_view.sync()
                shown = cv2.resize(mosaic, None, fx=args.display_scale,
                                   fy=args.display_scale,
                                   interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW_NAME, shown)
                print(f"frame {record['k']} ({position + 1}/{len(indices)}), "
                      f"points={len(result['points'])}", flush=True)
                dirty = False
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                playing = not playing
                next_frame_at = time.monotonic()
            elif key in (ord("a"), 81):
                position = (position - 1) % len(indices)
                playing = False
                dirty = True
            elif key in (ord("d"), 83):
                position = (position + 1) % len(indices)
                playing = False
                dirty = True
            if playing and time.monotonic() >= next_frame_at:
                position = (position + 1) % len(indices)
                next_frame_at += 1.0 / args.fps
                dirty = True
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        if camera_view is not None:
            camera_view.close()


if __name__ == "__main__":
    main()
