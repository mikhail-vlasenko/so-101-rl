"""Dual-camera SAM tracking → triangulated object position vs tag ground truth.

First step of the vision-estimator characterization (vision_multicam plan):
SAM 3 is prompted once per view with a text label (non-interactive), SAM 2
tracks the mask per view in streaming mode, each mask reduces to its pixel
centroid, and the centroid pair triangulates to a base-frame point
(real.vision.stereo) — the same geometry the tag pipeline just validated to sub-mm.

The sponge's AprilTag, detected in the same frames, provides ground truth: its
triangulated center moves rigidly with the sponge, so over a run

  - the MEAN of (sam - tag) is geometry, not error (mask centroid and tag
    center are different physical points on the object);
  - the SPREAD around that mean is the estimator's noise;
  - the SAM ray-pair gap is a real effect worth measuring, not a defect: each
    camera sees a different visible-surface centroid, so the two rays aim at
    genuinely different points. Its magnitude bounds the viewpoint-dependence
    error term the sim DR model must cover.

Per-stage timing (read / track / detect) feeds the delay_ms/frame_ms budget
for putting vision in the control loop. Captures are sequential and the scene
is assumed quasi-static; the free-running-camera sync offset is a separate,
unmeasured term (TODO.md).

Run:
    conda run -n mujoco_env python -m real.tracking.sam_track --frames 300
    python -m real.tracking.sam_track --gui          # live overlay window, q quits
"""
import argparse
import os
import time

import cv2
import numpy as np

from real.vision.detect import make_detector
from real.calib.table_anchor import TableAnchorTracker
from real.marker_spec import CUBE_TAG_ID, TABLE_TAG_IDS
from real.vision.overlay import YELLOW, StereoViewer, TagStyle, annotate_tags
from real.tracking.sam_seg import SAM2_MODELS, MaskTracker, load_sam3, mask_centroid, text_to_mask
from real.vision.stereo import pixel_rays, triangulate_rays
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera


def overlay(frame, mask, centroid, tag_det, label):
    view = frame.copy()
    m = mask.astype(np.uint8)
    view[m == 1] = (0.55 * np.array((0, 200, 0)) + 0.45 * view[m == 1]).astype(np.uint8)
    if centroid is not None:
        cv2.circle(view, (int(centroid[0]), int(centroid[1])), 6, (0, 0, 255), -1)
    if tag_det is not None:
        annotate_tags(view, [tag_det], {
            tag_det.id: TagStyle(f"GT tag {tag_det.id}", YELLOW)
        })
    cv2.putText(view, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return view


def summarize(name, arr_ms):
    a = np.array(arr_ms)
    return f"{name} {a.mean():6.1f} ms mean / {np.percentile(a, 95):6.1f} ms p95"


def main():
    parser = argparse.ArgumentParser(description="Dual-camera SAM tracking vs tag GT")
    parser.add_argument("--prompt", default="sponge", help="SAM3 text prompt")
    parser.add_argument("--model", choices=sorted(SAM2_MODELS), default="tiny",
                        help="SAM2 tracker size")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--gui", action="store_true",
                        help="live overlay window (native cv2), q quits early")
    parser.add_argument("--save-frames", default=None,
                        help="directory to write the last annotated frame pair into")
    args = parser.parse_args()

    detector = make_detector(args.family)
    wanted = set(TABLE_TAG_IDS) | {CUBE_TAG_ID}

    caps, mats, dists, anchors, trackers = {}, {}, {}, {}, {}
    for name in CAMERA_NAMES:
        caps[name], mats[name], dists[name] = open_rig_camera(name)
        anchors[name] = TableAnchorTracker(mats[name], dists[name])

    print(f"prompting SAM3 with {args.prompt!r} on both views...", flush=True)
    sam3 = load_sam3()
    first = {}
    for name in CAMERA_NAMES:
        ok, frame = caps[name].read()
        if not ok:
            raise RuntimeError(f"camera read failed on '{name}'")
        mask, score = text_to_mask(sam3, frame, args.prompt)
        print(f"  {name}: score {score:.2f}, area {int(mask.sum())} px", flush=True)
        first[name] = (frame, mask)
    del sam3

    for name in CAMERA_NAMES:
        trackers[name] = MaskTracker(args.model)
        trackers[name].prime(*first[name])
    print(f"tracking with SAM2 {args.model} for {args.frames} frames...", flush=True)

    t_read = {n: [] for n in CAMERA_NAMES}
    t_track = {n: [] for n in CAMERA_NAMES}
    t_detect = {n: [] for n in CAMERA_NAMES}
    centroids_px = {n: [] for n in CAMERA_NAMES}
    sam_pts, sam_gaps, tag_pts, tag_gaps, deltas = [], [], [], [], []
    lost = {n: 0 for n in CAMERA_NAMES}
    viewer = StereoViewer("sam_track") if args.gui else None
    loop_t0 = time.monotonic()
    n_loops = 0

    for _ in range(args.frames):
        rays_sam, rays_tag, views = {}, {}, {}
        for name in CAMERA_NAMES:
            t0 = time.monotonic()
            ok, frame = caps[name].read()
            if not ok:
                raise RuntimeError(f"camera read failed on '{name}'")
            t1 = time.monotonic()
            mask = trackers[name].track(frame)
            t2 = time.monotonic()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = {d.id: d for d in detector.detect(gray) if d.id in wanted}
            t3 = time.monotonic()
            t_read[name].append((t1 - t0) * 1e3)
            t_track[name].append((t2 - t1) * 1e3)
            t_detect[name].append((t3 - t2) * 1e3)

            centroid = mask_centroid(mask)
            if centroid is None:
                lost[name] += 1
            else:
                centroids_px[name].append(centroid)
            if args.gui or args.save_frames:
                views[name] = overlay(frame, mask, centroid,
                                      dets.get(CUBE_TAG_ID), f"{name} {args.prompt}")
            anchors[name].observe(dets)
            T_base_cam = anchors[name].value()
            if T_base_cam is None:
                continue
            if centroid is not None:
                _o, _d = pixel_rays(np.array([centroid]), mats[name], dists[name], T_base_cam)
                rays_sam[name] = (_o, _d)
            if CUBE_TAG_ID in dets:
                center_px = dets[CUBE_TAG_ID].corners.mean(axis=0, keepdims=True)
                rays_tag[name] = pixel_rays(center_px, mats[name], dists[name], T_base_cam)
        n_loops += 1

        sam_pt = None
        if len(rays_sam) == 2:
            pts, gaps = triangulate_rays(*rays_sam["main"], *rays_sam["aux"])
            sam_pt = pts[0]
            sam_pts.append(sam_pt)
            sam_gaps.append(gaps[0])
        if len(rays_tag) == 2:
            pts, gaps = triangulate_rays(*rays_tag["main"], *rays_tag["aux"])
            tag_pts.append(pts[0])
            tag_gaps.append(gaps[0])
            if sam_pt is not None:
                deltas.append(sam_pt - pts[0])

        if args.gui and views:
            if viewer.show(views) in (27, ord("q")):
                break

    wall = time.monotonic() - loop_t0
    for name in CAMERA_NAMES:
        caps[name].release()
    detector.close()
    if viewer is not None:
        viewer.close()
    if args.save_frames and views:
        os.makedirs(args.save_frames, exist_ok=True)
        for name, view in views.items():
            path = os.path.join(args.save_frames, f"sam_track_{name}.jpg")
            cv2.imwrite(path, view)
            print(f"wrote {path}")

    def mm(v):
        return np.array2string(np.asarray(v) * 1000.0, precision=2, suppress_small=True)

    print(f"\n{n_loops} loops in {wall:.1f} s -> {n_loops / wall:.1f} fps effective")
    print(f"triangulated: sam {len(sam_pts)}, tag {len(tag_pts)}, paired {len(deltas)}; "
          f"lost mask frames: " + ", ".join(f"{n}={lost[n]}" for n in CAMERA_NAMES))
    for name in CAMERA_NAMES:
        print(f"  {name}: {summarize('read', t_read[name])} | "
              f"{summarize('sam2', t_track[name])} | "
              f"{summarize('tags', t_detect[name])}")
    if not deltas:
        raise RuntimeError("no frame yielded both a SAM and a tag triangulation")

    sam_arr, tag_arr, delta = np.array(sam_pts), np.array(tag_pts), np.array(deltas)
    for name in CAMERA_NAMES:
        px = np.array(centroids_px[name])
        print(f"  {name} mask centroid jitter: std ({px[:, 0].std():.2f}, "
              f"{px[:, 1].std():.2f}) px over {len(px)} frames")
    print(f"\nSAM triangulated point (base frame, mm):")
    print(f"  mean {mm(sam_arr.mean(axis=0))}   per-axis std {mm(sam_arr.std(axis=0))}")
    print(f"  ray gap {np.mean(sam_gaps) * 1000:.2f} mm mean / "
          f"{np.percentile(sam_gaps, 95) * 1000:.2f} mm p95   "
          f"(viewpoint-dependent centroid disparity)")
    print(f"tag triangulated center (same frames, mm):")
    print(f"  mean {mm(tag_arr.mean(axis=0))}   per-axis std {mm(tag_arr.std(axis=0))}   "
          f"ray gap {np.mean(tag_gaps) * 1000:.2f} mm mean")
    print(f"sam - tag (mm):  mean {mm(delta.mean(axis=0))}  <- geometry offset, not error")
    print(f"  jitter around that mean: per-axis std {mm(delta.std(axis=0))}, "
          f"|delta - mean| {np.linalg.norm(delta - delta.mean(axis=0), axis=1).mean() * 1000:.2f} mm mean")


if __name__ == "__main__":
    main()
