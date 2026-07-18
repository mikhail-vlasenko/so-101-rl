"""Two-camera triangulation validation against the sponge's AprilTag.

End-to-end accuracy check of the binocular geometry stack before any
segmentation model sits on top: both cameras are independently anchored to the
arm base frame per frame via the fixed table tag (`t_base_table` in
extrinsics.yaml is camera-independent), the cube tag's four corners are
matched across views by construction, and each corner is triangulated
ray-by-ray (`real.stereo`). The printed tag edge is known exactly
(marker_spec.TAG_SIZE_MM), so the recovered corner-to-corner distances measure
*metric* accuracy — intrinsics + anchoring + triangulation combined — while
the ray-pair gaps measure cross-view consistency, and the triangulated centre
is compared against each camera's solvePnP estimate of the same tag.

Unlike the rollout pipeline (real.marker_obs) there is deliberately no camera
EMA: per-frame raw anchoring exposes the true jitter this rig will feed the
triangulator. A static scene is assumed (sponge at rest), so the free-running
cameras' capture-time offset does not enter.

Run:
    conda run -n mujoco_env python -m real.stereo_check --frames 100
"""
import argparse
import os

import cv2
import numpy as np
import yaml

from real.camera import open_camera, device_index_for_serial, SERIALS
from real.detect import make_detector
from real.extrinsics import base_cam_from_table, load_extrinsics, rt_to_mat
from real.marker_spec import (
    CUBE_TAG_ID,
    MARKER_EXPOSURE,
    MARKER_GAIN,
    TABLE_TAG_ID,
    TAG_SIZE_MM,
)
from real.pose import PoseEstimator, intrinsics_path, load_intrinsics
from real.stereo import pixel_rays, triangulate_rays

CAMERA_NAMES = ("main", "aux")


def open_rig_camera(name):
    """Open one unit at its own calibrated focus with its own intrinsics."""
    path = intrinsics_path(name)
    with open(path) as f:
        focus = int(yaml.safe_load(f)["focus_absolute"])
    camera_matrix, dist_coeffs = load_intrinsics(path)
    cap = open_camera(device=device_index_for_serial(SERIALS[name]), focus=focus,
                      exposure=MARKER_EXPOSURE, gain=MARKER_GAIN)
    return cap, camera_matrix, dist_coeffs


def annotate(frame, dets):
    view = frame.copy()
    for d in dets.values():
        pts = d.corners.astype(np.int32)
        cv2.polylines(view, [pts], True, (0, 255, 0), 2)
        c = pts.mean(axis=0).astype(int)
        cv2.circle(view, tuple(c), 3, (0, 0, 255), -1)
        cv2.putText(view, str(d.id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    return view


def fmt_mm(v):
    return np.array2string(np.asarray(v) * 1000.0, precision=2, suppress_small=True)


def main():
    parser = argparse.ArgumentParser(description="Two-camera triangulation check")
    parser.add_argument("--frames", type=int, default=100,
                        help="frames to accumulate (both views must see table + cube tags)")
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--save-frames", default=None,
                        help="directory to write the last annotated frame pair into")
    args = parser.parse_args()

    T_base_table, _, _, _ = load_extrinsics()
    detector = make_detector(args.family)
    wanted = {TABLE_TAG_ID, CUBE_TAG_ID}
    edge_nominal_m = TAG_SIZE_MM[CUBE_TAG_ID] / 1000.0

    caps, mats, dists, estimators = {}, {}, {}, {}
    for name in CAMERA_NAMES:
        caps[name], mats[name], dists[name] = open_rig_camera(name)
        estimators[name] = PoseEstimator(mats[name], dists[name])

    seen = {name: {"table": 0, "cube": 0} for name in CAMERA_NAMES}
    tri_centers, gaps_center, gaps_corner = [], [], []
    edges, pnp_pos, cam_pos = [], {n: [] for n in CAMERA_NAMES}, {n: [] for n in CAMERA_NAMES}
    last = {}

    for _ in range(args.frames):
        frame_dets = {}
        for name in CAMERA_NAMES:
            ok, frame = caps[name].read()
            if not ok:
                raise RuntimeError(f"camera read failed on '{name}'")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = {d.id: d for d in detector.detect(gray) if d.id in wanted}
            frame_dets[name] = dets
            last[name] = (frame, dets)
            for role, tag in (("table", TABLE_TAG_ID), ("cube", CUBE_TAG_ID)):
                seen[name][role] += tag in dets
        if not all(wanted <= set(frame_dets[name]) for name in CAMERA_NAMES):
            continue

        rays, pnp = {}, {}
        for name in CAMERA_NAMES:
            dets = frame_dets[name]
            T_base_cam = base_cam_from_table(
                T_base_table, *estimators[name].estimate(dets[TABLE_TAG_ID]))
            cam_pos[name].append(T_base_cam[:3, 3])
            T_base_cube = T_base_cam @ rt_to_mat(*estimators[name].estimate(dets[CUBE_TAG_ID]))
            pnp[name] = T_base_cube[:3, 3]
            corners = dets[CUBE_TAG_ID].corners
            pixels = np.vstack([corners, corners.mean(axis=0, keepdims=True)])
            rays[name] = pixel_rays(pixels, mats[name], dists[name], T_base_cam)

        points, gaps = triangulate_rays(*rays["main"], *rays["aux"])
        corners3d, center3d = points[:4], points[4]
        tri_centers.append(center3d)
        gaps_center.append(gaps[4])
        gaps_corner.append(gaps[:4].mean())
        edges.append([np.linalg.norm(corners3d[i] - corners3d[(i + 1) % 4])
                      for i in range(4)])
        for name in CAMERA_NAMES:
            pnp_pos[name].append(pnp[name])

    for name in CAMERA_NAMES:
        caps[name].release()

    n = len(tri_centers)
    print(f"\nframes with full detections in both views: {n}/{args.frames}")
    for name in CAMERA_NAMES:
        print(f"  {name}: table tag in {seen[name]['table']}/{args.frames}, "
              f"cube tag in {seen[name]['cube']}/{args.frames}")
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for name, (frame, dets) in last.items():
            path = os.path.join(args.save_frames, f"stereo_check_{name}.jpg")
            cv2.imwrite(path, annotate(frame, dets))
            print(f"  wrote {path}")
    if n == 0:
        raise RuntimeError("no frame had table + cube tags in both views; "
                           "check framing (use --save-frames to inspect)")

    tri = np.array(tri_centers)
    edges = np.array(edges) * 1000.0
    edge_err = edges.mean() - edge_nominal_m * 1000.0
    print(f"\ntriangulated cube-tag centre (base frame, mm):")
    print(f"  mean {fmt_mm(tri.mean(axis=0))}   per-axis std {fmt_mm(tri.std(axis=0))}")
    print(f"ray-pair gap (cross-view consistency, mm):")
    print(f"  centre {np.mean(gaps_center) * 1000:.2f} mean / {np.max(gaps_center) * 1000:.2f} max"
          f"   corners {np.mean(gaps_corner) * 1000:.2f} mean")
    print(f"recovered tag edge (nominal {edge_nominal_m * 1000:.1f} mm):")
    print(f"  {edges.mean():.2f} ± {edges.std():.2f} mm  (bias {edge_err:+.2f} mm, "
          f"{100.0 * edge_err / (edge_nominal_m * 1000.0):+.1f}%)")
    print("per-camera solvePnP of the same tag vs triangulation (mm):")
    for name in CAMERA_NAMES:
        p = np.array(pnp_pos[name])
        d = np.linalg.norm(p - tri, axis=1)
        print(f"  {name}: pnp mean {fmt_mm(p.mean(axis=0))}   "
              f"|pnp - tri| {d.mean() * 1000:.2f} mean / {d.max() * 1000:.2f} max")
    dpnp = np.linalg.norm(np.array(pnp_pos["main"]) - np.array(pnp_pos["aux"]), axis=1)
    print(f"  |pnp_main - pnp_aux| {dpnp.mean() * 1000:.2f} mean / {dpnp.max() * 1000:.2f} max")
    print("per-frame camera anchor jitter (std of T_base_cam translation, mm):")
    for name in CAMERA_NAMES:
        print(f"  {name}: {fmt_mm(np.array(cam_pos[name]).std(axis=0))}")


if __name__ == "__main__":
    main()
