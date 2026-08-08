"""Live tag-GT versus tag-free sponge-observation viewer.

Both rig cameras feed the same pipeline used by deployment: SAM3 seeds a
SAM2 tracker in each view, mask centroids produce the cyan live point, and a
two-view visual hull refreshes the orange center/√M observation only after the
sponge is stationary and sufficiently visible. The observation holds while
the sponge moves or is occluded, exactly as the policy sees it.

The sponge's calibrated AprilTags are evaluation-only ground truth. Every
accepted tag (<=60 degree incidence) from either camera votes for the body
pose. In the MuJoCo window:

- translucent green box: tag-derived physical sponge pose;
- green ellipsoid: exact √M implied by that tag pose and the sponge dimensions;
- orange ellipsoid: held visual-hull center/√M observation;
- cyan sphere: current live mask-centroid triangulation.

A perfect shape observation makes the green and orange ellipsoids coincide.
The synchronized camera window shows SAM masks, accepted tags, static/gate
state, numerical errors and observation ages. Press q/Esc there, or close the
MuJoCo window, to stop.

Run:
    conda run --no-capture-output -n mujoco_env \
      python -m real.tracking.view_object_estimate
"""
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from panel.sim_stream import draw_object_channels, draw_sqrtm_ellipsoid
from real.calib.extrinsics import (
    PoseEMA,
    average_transforms,
    base_cam_from_table,
    load_extrinsics,
    mat_to_pos_quat,
)
from real.marker_spec import TABLE_TAG_ID
from real.rollout.object_obs import (
    CAM_EMA_ALPHA,
    REPROMPT_AFTER_EMPTY,
    draw_object_overlay,
)
from real.tracking.eval_estimator import principal_axis_angle_deg
from real.tracking.hull_shape import hull_estimate
from real.tracking.sam_seg import (
    SAM2_MODELS,
    MaskTracker,
    find_text_mask,
    load_sam3,
    mask_centroid,
    text_to_mask,
)
from real.tracking.tag_body_calib import (
    MAX_INCIDENCE_DEG,
    body_pose_from_tag,
    incidence_angle_deg,
    load_sponge_tags,
)
from real.vision.detect import make_detector
from real.vision.overlay import (
    GREEN,
    RED,
    TABLE_BLUE,
    WHITE,
    YELLOW,
    OverlayLine,
    StereoViewer,
    TagStyle,
    annotate_tags,
)
from real.vision.pose import PoseEstimator
from real.vision.stereo import pixel_rays, triangulate_rays
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera
from src.shape_obs import (
    ObjectChannelDriver,
    box_sqrtm,
    sqrtm_from_cov,
    sqrtm_from_upper,
    sqrtm_upper,
)

GT_ELLIPSOID_RGBA = (0.1, 1.0, 0.1, 0.35)


def make_view_model(half_extents):
    """Minimal table scene whose only model body is the tag-derived sponge."""
    hx, hy, hz = np.asarray(half_extents, dtype=np.float64)
    xml = f"""
    <mujoco model="object_estimate_comparison">
      <visual>
        <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0 0 0"/>
        <global azimuth="150" elevation="-25"/>
      </visual>
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


def triangulated_live(masks, geometry):
    """Two-mask centroid triangulation in the base frame, or None."""
    rays = []
    for camera in CAMERA_NAMES:
        centroid = mask_centroid(masks[camera])
        if centroid is None or camera not in geometry:
            return None
        rays.append(pixel_rays(np.array([centroid]), *geometry[camera]))
    points, _ = triangulate_rays(*rays[0], *rays[1])
    return points[0]


def tag_body_gt(tag_poses, anchors, tag_transforms):
    """Fused T_base_body plus accepted (camera, tag) incidence angles."""
    body_views = []
    accepted = {}
    for camera in CAMERA_NAMES:
        if camera not in anchors:
            continue
        candidates = []
        for tag, (rvec, tvec) in tag_poses[camera].items():
            if tag not in tag_transforms:
                continue
            angle = incidence_angle_deg(rvec, tvec)
            if angle > MAX_INCIDENCE_DEG:
                continue
            candidates.append(body_pose_from_tag(
                rvec, tvec, tag_transforms[tag]))
            accepted[(camera, tag)] = angle
        if candidates:
            body_views.append(anchors[camera] @ average_transforms(candidates))
    if not body_views:
        return None, accepted
    return average_transforms(body_views), accepted


def comparison_metrics(T_base_body, center, sqrtm6, half_extents):
    """User-facing center/shape discrepancies, or None before a precise fix."""
    if T_base_body is None or not np.any(sqrtm6):
        return None
    gt_sqrtm = box_sqrtm(T_base_body[:3, :3], half_extents)
    est_sqrtm = sqrtm_from_upper(sqrtm6)
    delta = (center - T_base_body[:3, 3]) * 1000.0
    return {
        "delta_mm": delta,
        "center_mm": float(np.linalg.norm(delta)),
        "sqrtm_mm": float(np.linalg.norm(est_sqrtm - gt_sqrtm) * 1000.0),
        "axis_deg": principal_axis_angle_deg(est_sqrtm, gt_sqrtm),
        "half_mm": np.sort(np.linalg.eigvalsh(est_sqrtm)) * np.sqrt(3.0) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="sponge", help="SAM3 text prompt")
    parser.add_argument("--sam2-model", choices=sorted(SAM2_MODELS), default="tiny")
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frame pairs; 0 runs until the viewer closes")
    parser.add_argument("--no-camera-view", action="store_true",
                        help="show only MuJoCo; close its window to stop")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be >= 0")

    half_extents, tag_transforms = load_sponge_tags()
    T_base_table, _, _, _ = load_extrinsics()
    detector = make_detector(args.family)
    wanted = set(tag_transforms) | {TABLE_TAG_ID}
    caps, mats, dists, estimators, anchors, cam_ema = {}, {}, {}, {}, {}, {}
    trackers = {}
    try:
        for camera in CAMERA_NAMES:
            caps[camera], mats[camera], dists[camera] = open_rig_camera(camera)
            estimators[camera] = PoseEstimator(mats[camera], dists[camera])
            cam_ema[camera] = PoseEMA(CAM_EMA_ALPHA)

        print(f"prompting SAM3 with {args.prompt!r} on both views...", flush=True)
        sam3 = load_sam3()
        for camera in CAMERA_NAMES:
            ok, frame = caps[camera].read()
            if not ok:
                raise RuntimeError(f"camera read failed on '{camera}'")
            mask, score = text_to_mask(sam3, frame, args.prompt)
            print(f"  {camera}: score {score:.2f}, area {int(mask.sum())} px",
                  flush=True)
            trackers[camera] = MaskTracker(args.sam2_model)
            trackers[camera].prime(frame, mask)

        model, data, gt_qposadr, gt_geom = make_view_model(half_extents)
        camera_view = None if args.no_camera_view else StereoViewer(
            "tag GT vs SAM shape observation")
        driver = ObjectChannelDriver()
        empty_run = {camera: 0 for camera in CAMERA_NAMES}
        window_max_area = {camera: 0.0 for camera in CAMERA_NAMES}
        hull_centers, hull_Ms = [], []
        frame_index = 0

        print("MuJoCo: green=tag GT, orange=precise sqrtM, cyan=live centroid. "
              "Hold the sponge still for 0.5 s to refresh orange.", flush=True)
        with mujoco.viewer.launch_passive(model, data) as mj_viewer:
            mj_viewer.cam.lookat[:] = [0.20, 0.0, 0.03]
            mj_viewer.cam.distance = 0.65
            mj_viewer.cam.azimuth = 135.0
            mj_viewer.cam.elevation = -30.0
            while mj_viewer.is_running() and (args.frames == 0 or
                                               frame_index < args.frames):
                frame_index += 1
                frames, masks, centroids, detections, tag_poses = {}, {}, {}, {}, {}
                geometry = {}
                for camera in CAMERA_NAMES:
                    ok, frame = caps[camera].read()
                    if not ok:
                        raise RuntimeError(f"camera read failed on '{camera}'")
                    frames[camera] = frame
                    mask = trackers[camera].track(frame)
                    if mask.any():
                        empty_run[camera] = 0
                    else:
                        empty_run[camera] += 1
                        if empty_run[camera] >= REPROMPT_AFTER_EMPTY:
                            empty_run[camera] = 0
                            found = find_text_mask(sam3, frame, args.prompt)
                            if found is not None:
                                mask, score = found
                                trackers[camera].prime(frame, mask)
                                print(f"re-acquired '{camera}' score {score:.2f}",
                                      flush=True)
                    masks[camera] = mask
                    centroids[camera] = mask_centroid(mask)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detections[camera] = {
                        det.id: det for det in detector.detect(gray) if det.id in wanted
                    }
                    tag_poses[camera] = {
                        tag: estimators[camera].estimate(det)
                        for tag, det in detections[camera].items()
                    }
                    if TABLE_TAG_ID in tag_poses[camera]:
                        anchors[camera] = cam_ema[camera].update(base_cam_from_table(
                            T_base_table, *tag_poses[camera][TABLE_TAG_ID]))
                    if camera in anchors:
                        geometry[camera] = (mats[camera], dists[camera],
                                            anchors[camera])

                t = time.monotonic()
                live = triangulated_live(masks, geometry)
                driver.ingest_live(t, live)
                static = live is not None and driver.static_now()
                gate = False
                hull_ms = float("nan")
                if live is None:
                    window_max_area = {camera: 0.0 for camera in CAMERA_NAMES}
                    hull_centers, hull_Ms = [], []
                elif static:
                    areas = {camera: float(masks[camera].sum())
                             for camera in CAMERA_NAMES}
                    for camera in CAMERA_NAMES:
                        window_max_area[camera] = max(window_max_area[camera],
                                                     areas[camera])
                    visible = [areas[camera] / window_max_area[camera]
                               for camera in CAMERA_NAMES]
                    gate = driver.gate_open(visible)
                    if gate and len(geometry) == len(CAMERA_NAMES):
                        started = time.monotonic()
                        hull_center, hull_M = hull_estimate(
                            masks, geometry, CAMERA_NAMES)
                        hull_ms = (time.monotonic() - started) * 1000.0
                        if hull_center is not None:
                            hull_centers.append(hull_center)
                            hull_Ms.append(hull_M)
                            driver.ingest_precise(
                                t, np.mean(hull_centers, axis=0),
                                sqrtm_from_cov(np.mean(hull_Ms, axis=0)))
                else:
                    window_max_area = {camera: 0.0 for camera in CAMERA_NAMES}
                    hull_centers, hull_Ms = [], []

                served_live, live_age, center, sqrtm6, precise_age = driver.serve(t)
                T_base_body, accepted = tag_body_gt(
                    tag_poses, anchors, tag_transforms)
                metrics = comparison_metrics(
                    T_base_body, center, sqrtm6, half_extents)

                if T_base_body is None:
                    model.geom_rgba[gt_geom, 3] = 0.0
                else:
                    model.geom_rgba[gt_geom, 3] = 0.22
                    pos, quat = mat_to_pos_quat(T_base_body)
                    data.qpos[gt_qposadr:gt_qposadr + 3] = pos
                    data.qpos[gt_qposadr + 3:gt_qposadr + 7] = quat
                mujoco.mj_forward(model, data)
                mj_viewer.user_scn.ngeom = 0
                if T_base_body is not None:
                    draw_sqrtm_ellipsoid(
                        mj_viewer.user_scn, T_base_body[:3, 3],
                        sqrtm_upper(box_sqrtm(T_base_body[:3, :3], half_extents)),
                        GT_ELLIPSOID_RGBA)
                draw_object_channels(
                    mj_viewer.user_scn, served_live, center, sqrtm6)
                mj_viewer.sync()

                if static and gate:
                    state, state_color = "STATIC / PRECISE REFRESH", GREEN
                elif static:
                    state, state_color = "STATIC / OCCLUDED HOLD", YELLOW
                elif live is None:
                    state, state_color = "NO LIVE / HOLD", RED
                else:
                    state, state_color = "MOVING / HOLD", YELLOW
                if metrics is None:
                    error_line = OverlayLine("comparison unavailable", RED)
                    size_line = OverlayLine("estimated half-sizes unavailable", RED)
                else:
                    delta = np.array2string(metrics["delta_mm"], precision=1,
                                            suppress_small=True)
                    error_line = OverlayLine(
                        f"estimate-GT center {delta} mm | norm {metrics['center_mm']:.1f} mm "
                        f"| sqrtM {metrics['sqrtm_mm']:.1f} mm | axis "
                        f"{metrics['axis_deg']:.1f} deg", WHITE)
                    sizes = np.array2string(metrics["half_mm"], precision=1,
                                            suppress_small=True)
                    size_line = OverlayLine(
                        f"estimated equivalent half-sizes {sizes} mm | true "
                        f"{np.sort(half_extents * 1000.0)}", WHITE)

                if camera_view is not None:
                    views, camera_lines = {}, {}
                    for camera in CAMERA_NAMES:
                        if camera in geometry:
                            views[camera] = draw_object_overlay(
                                frames[camera].copy(), masks[camera],
                                centroids[camera], served_live, center, sqrtm6,
                                *geometry[camera])
                        else:
                            views[camera] = frames[camera].copy()
                        styles = {TABLE_TAG_ID: TagStyle("table", TABLE_BLUE, 3)}
                        for tag in tag_transforms:
                            angle = accepted.get((camera, tag))
                            if angle is None:
                                styles[tag] = TagStyle(f"tag {tag} rejected", RED, 3)
                            else:
                                styles[tag] = TagStyle(
                                    f"tag {tag} GT {angle:.0f}deg", GREEN, 3)
                        annotate_tags(views[camera], detections[camera].values(), styles)
                        camera_lines[camera] = [OverlayLine(
                            f"{camera} anchor={'OK' if camera in anchors else 'MISSING'} "
                            f"mask={int(masks[camera].sum())} px",
                            GREEN if camera in anchors and masks[camera].any() else RED)]
                    header = [
                        OverlayLine(
                            f"frame {frame_index} | {state} | ages live={live_age:.2f}s "
                            f"precise={precise_age:.2f}s | hull={hull_ms:.0f} ms",
                            state_color),
                        error_line,
                        size_line,
                        OverlayLine(
                            "MuJoCo: green tag GT | orange precise sqrtM | cyan live | q/ESC quits",
                            WHITE),
                    ]
                    if camera_view.show(views, camera_lines, header) in (27, ord("q")):
                        break

                if frame_index % 30 == 0 and metrics is not None:
                    print(f"frame {frame_index}: center {metrics['center_mm']:.1f} mm, "
                          f"sqrtM {metrics['sqrtm_mm']:.1f} mm, axis "
                          f"{metrics['axis_deg']:.1f} deg, {state}", flush=True)
    finally:
        for cap in caps.values():
            cap.release()
        detector.close()
        if 'camera_view' in locals() and camera_view is not None:
            camera_view.close()


if __name__ == "__main__":
    main()
