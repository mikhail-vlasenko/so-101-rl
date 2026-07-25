"""Offline estimator evaluation against the recorded dataset's tag GT.

Every candidate shape estimator is judged the same way, fully offline, so
estimator work never needs the physical rig: masks are computed once per
camera with the SAM stack (real/tracking/sam_seg.py; cached under the dataset
directory, keeping the recording itself estimator-agnostic), the candidate
turns mask pairs into per-frame (centroid, M) measurements, static windows
average M in the linear domain before the square root, and everything is
scored against the tag-derived GT body pose recorded by
real/tracking/record_shapes.py.

Acceptance criterion (plan decision 4, fixed before any estimator existed):
on held-out static windows the per-component errors must sit within the
dr=full noise envelope — 2 sigma of the corresponding knobs, read from
conf/dr/full.yaml at eval time (no duplicated numbers):

- precise center per-component error   vs  2*sqrt(noise.precise_sigma^2 + bias.precise_sigma^2)
- hull depth spread (added along the    vs  2*bias.sqrtm_depth_sigma
  cameras' shared axis)
- sqrtM eigenvalue error               vs  cube_size_jitter/2 / sqrt(3)  (the size-DR envelope)
- live jitter (std in a static window) vs  2*noise.live_sigma, per occlusion slice
- live window-mean spread              vs  2*bias.live_sigma

Ship the estimator when green; escalate (plan decision 3) when not, choosing
from the occlusion/component slices below.

Run:
    conda run -n mujoco_env python -m real.tracking.eval_estimator datasets/sponge_<stamp>
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import pos_quat_to_mat
from real.vision.stereo import pixel_rays, triangulate_rays
from src.shape_obs import (
    box_sqrtm,
    camera_depth_axis,
    depth_spread_excess,
    sqrtm_from_cov,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DR_FULL_YAML = REPO_ROOT / "conf" / "dr" / "full.yaml"

# A static window must have at least this many usable frame pairs to score.
MIN_WINDOW_FRAMES = 5
# Occlusion slices (fraction of the window's peak mask area that is missing).
OCCLUSION_BINS = (0.0, 0.25, 0.5, 0.75, 1.01)


def load_dataset(dataset_dir: Path):
    """(records, meta) from a record_shapes.py dataset directory."""
    with open(dataset_dir / "meta.yaml") as f:
        meta = yaml.safe_load(f)
    records = []
    with open(dataset_dir / "index.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    assert records, f"empty dataset index in {dataset_dir}"
    return records, meta


def compute_masks(dataset_dir: Path, records, meta, prompt, sam2_model,
                  reprompt_after=30):
    """Run the SAM stack over every recorded frame per camera; cache each mask
    as a PNG under <dataset>/masks/<sam2_model>/. Mirrors the online pipeline:
    one SAM3 text prompt seeds a streaming SAM2 tracker per camera, and a run
    of `reprompt_after` empty masks triggers re-acquisition via SAM3.

    A recording can outlast the object (the sponge gets carried out of view at
    the end of a session), so a prompt that finds nothing yields an empty mask
    and a retry `reprompt_after` frames later rather than an error."""
    from real.tracking.sam_seg import MaskTracker, find_text_mask, load_sam3

    mask_dir = dataset_dir / "masks" / sam2_model
    todo = [(rec, name) for rec in records for name in meta["cameras"]
            if not (mask_dir / rec["frame"][name]).with_suffix(".png").exists()]
    if not todo:
        return mask_dir
    print(f"computing {len(todo)} masks into {mask_dir} ...", flush=True)
    sam3 = load_sam3()
    tracker = MaskTracker(sam2_model)
    for name in meta["cameras"]:
        (mask_dir / "frames").mkdir(parents=True, exist_ok=True)
        primed = False
        empty_run = 0
        retry_in = 0
        for rec in records:
            out_path = (mask_dir / rec["frame"][name]).with_suffix(".png")
            if out_path.exists():
                continue
            frame = cv2.imread(str(dataset_dir / rec["frame"][name]))
            assert frame is not None, rec["frame"][name]
            if primed and empty_run < reprompt_after:
                mask = tracker.track(frame)
            elif retry_in > 0:
                retry_in -= 1
                mask = np.zeros(frame.shape[:2], dtype=bool)
            else:
                found = find_text_mask(sam3, frame, prompt)
                if found is None:
                    primed = False
                    retry_in = reprompt_after
                    mask = np.zeros(frame.shape[:2], dtype=bool)
                else:
                    mask, score = found
                    tracker.prime(frame, mask)
                    primed = True
                    print(f"  {name}: (re)prompted at k={rec['k']}, "
                          f"score {score:.2f}", flush=True)
            empty_run = 0 if mask.any() else empty_run + 1
            cv2.imwrite(str(out_path), mask.astype(np.uint8) * 255)
    return mask_dir


def load_mask(mask_dir: Path, rec, name):
    m = cv2.imread(str((mask_dir / rec["frame"][name]).with_suffix(".png")),
                   cv2.IMREAD_GRAYSCALE)
    assert m is not None, rec["frame"][name]
    return m > 127


def gt_pose(rec):
    """GT (center (3,), R (3,3)) in the base frame, or None without fused GT."""
    fused = rec["body"]["fused"]
    if fused is None:
        return None
    T = pos_quat_to_mat(fused["pos"], fused["quat"])
    return T[:3, 3], T[:3, :3]


def frame_geometry(rec, meta):
    """Per-camera (camera_matrix, dist, T_base_cam) for one record, or None
    when any camera is un-anchored this frame."""
    out = {}
    for name in meta["cameras"]:
        anchor = rec["T_base_cam"][name]
        if anchor is None:
            return None
        out[name] = (np.array(meta["camera_matrix"][name]),
                     np.array(meta["dist_coeffs"][name]),
                     pos_quat_to_mat(anchor["pos"], anchor["quat"]))
    return out


def live_centroid(masks, geometry, cameras):
    """Triangulated live centroid of the two mask centroids, or None when a
    view is empty — the same reduction the online pipeline serves."""
    rays = []
    for name in cameras:
        ys, xs = np.nonzero(masks[name])
        if xs.size == 0:
            return None
        mat, dist, T_base_cam = geometry[name]
        rays.append(pixel_rays(np.array([[xs.mean(), ys.mean()]]), mat, dist,
                               T_base_cam))
    pts, _ = triangulate_rays(*rays[0], *rays[1])
    return pts[0]


def static_windows(records):
    """Contiguous runs of static-labeled records, each at least
    MIN_WINDOW_FRAMES long."""
    windows = []
    run = []
    for rec in records:
        if rec["static"]:
            run.append(rec)
        else:
            if len(run) >= MIN_WINDOW_FRAMES:
                windows.append(run)
            run = []
    if len(run) >= MIN_WINDOW_FRAMES:
        windows.append(run)
    return windows


def principal_axis_angle_deg(S_est, S_gt):
    """Angle between the two major axes (symmetry-aware: an axis and its
    negation are the same axis)."""
    _, V_est = np.linalg.eigh(S_est)
    _, V_gt = np.linalg.eigh(S_gt)
    return float(np.degrees(np.arccos(np.clip(abs(V_est[:, -1] @ V_gt[:, -1]),
                                              0.0, 1.0))))


def load_envelope():
    """The dr=full acceptance envelope, read from the yaml at eval time."""
    with open(DR_FULL_YAML) as f:
        dr = yaml.safe_load(f)
    noise, bias = dr["obs_noise"], dr["obs_bias"]
    return {
        "center_m": 2.0 * float(np.hypot(noise["precise_sigma"], bias["precise_sigma"])),
        "depth_spread_m": 2.0 * bias["sqrtm_depth_sigma"],
        "eig_m": dr["cube_size_jitter"] / 2.0 / np.sqrt(3.0),
        "live_jitter_m": 2.0 * noise["live_sigma"],
        "live_bias_m": 2.0 * bias["live_sigma"],
    }


def occlusion_fraction(areas):
    """Per-frame missing-area fraction relative to the window's peak mask area
    (the offline stand-in for the online area-vs-static-baseline gate)."""
    areas = np.asarray(areas, dtype=np.float64)
    return 1.0 - areas / areas.max()


def evaluate(dataset_dir: Path, estimator_name: str, prompt: str,
             sam2_model: str) -> bool:
    from real.tracking.hull_shape import hull_estimate
    estimators = {"hull": hull_estimate}
    estimate = estimators[estimator_name]

    records, meta = load_dataset(dataset_dir)
    cameras = meta["cameras"]
    half_extents = np.array(meta["half_extents"])
    mask_dir = compute_masks(dataset_dir, records, meta, prompt, sam2_model)
    envelope = load_envelope()

    windows = static_windows(records)
    print(f"{len(records)} frame pairs, {len(windows)} static windows "
          f"(>= {MIN_WINDOW_FRAMES} frames each)")
    assert windows, "no usable static windows — record more settle segments"

    center_err, axis_err, eig_err, frob_err = [], [], [], []
    depth_excess = []
    live_jitter_by_occ = {i: [] for i in range(len(OCCLUSION_BINS) - 1)}
    live_window_means = []
    for win in windows:
        centers, Ms, lives, occ = [], [], [], []
        gt_centers, gt_Rs, cam_positions = [], [], []
        for rec in win:
            geometry = frame_geometry(rec, meta)
            gt = gt_pose(rec)
            if geometry is None or gt is None:
                continue
            masks = {name: load_mask(mask_dir, rec, name) for name in cameras}
            if any(not masks[name].any() for name in cameras):
                continue
            live = live_centroid(masks, geometry, cameras)
            if live is None:
                continue
            c, M = estimate(masks, geometry, cameras)
            centers.append(c)
            Ms.append(M)
            lives.append(live)
            occ.append(min(masks[name].sum() for name in cameras))
            gt_centers.append(gt[0])
            gt_Rs.append(gt[1])
            cam_positions.append([geometry[name][2][:3, 3] for name in cameras])
        if len(centers) < MIN_WINDOW_FRAMES:
            continue
        # Window aggregation: centers averaged, M averaged in the linear
        # domain BEFORE the square root (plan stage 3).
        est_center = np.mean(centers, axis=0)
        est_sqrtm = sqrtm_from_cov(np.mean(Ms, axis=0))
        gt_center = np.mean(gt_centers, axis=0)
        gt_sqrtm = box_sqrtm(Rotation.from_matrix(np.stack(gt_Rs)).mean().as_matrix(),
                             half_extents)
        center_err.append(est_center - gt_center)
        axis_err.append(principal_axis_angle_deg(est_sqrtm, gt_sqrtm))
        # The DR model says the hull's shape error is spread added along the
        # axis the two views share (src/base_env._hull_sqrtm), so measure that
        # directly rather than an axis angle — the angle is a pose-dependent
        # consequence of this one number, not an independent error mode.
        depth_axis = camera_depth_axis(np.mean(cam_positions, axis=0), gt_center)
        depth_excess.append(depth_spread_excess(
            np.mean(Ms, axis=0), gt_sqrtm @ gt_sqrtm, depth_axis))
        eig_err.append(np.abs(np.sort(np.linalg.eigvalsh(est_sqrtm))
                              - np.sort(np.linalg.eigvalsh(gt_sqrtm))))
        frob_err.append(float(np.linalg.norm(est_sqrtm - gt_sqrtm)))
        # Live channel: jitter about the window mean, sliced by occlusion.
        lives = np.array(lives)
        window_mean = lives.mean(axis=0)
        live_window_means.append(window_mean - gt_center)
        occ_frac = occlusion_fraction(occ)
        for i, (lo, hi) in enumerate(zip(OCCLUSION_BINS[:-1], OCCLUSION_BINS[1:])):
            sel = (occ_frac >= lo) & (occ_frac < hi)
            if sel.sum() >= 2:
                live_jitter_by_occ[i].append(
                    np.linalg.norm(lives[sel] - lives[sel].mean(axis=0), axis=1))

    assert center_err, "no static window produced a full set of measurements"
    center_err = np.array(center_err)
    eig_err = np.array(eig_err)
    live_window_means = np.array(live_window_means)
    # The mean live offset is geometry (visible-surface point vs body center);
    # only its spread across windows is estimator error.
    live_bias_spread = np.linalg.norm(
        live_window_means - live_window_means.mean(axis=0), axis=1)

    checks = []

    def check(label, value, bound):
        checks.append((label, value, bound, value <= bound))

    check("precise center |err| p95 (m)",
          float(np.percentile(np.abs(center_err).max(axis=1), 95)),
          envelope["center_m"])
    check("hull depth spread p95 (m)",
          float(np.percentile(depth_excess, 95)), envelope["depth_spread_m"])
    check("sqrtM eigenvalue |err| p95 (m)",
          float(np.percentile(eig_err.max(axis=1), 95)), envelope["eig_m"])
    check("live window-mean spread p95 (m)",
          float(np.percentile(live_bias_spread, 95)), envelope["live_bias_m"])
    for i, (lo, hi) in enumerate(zip(OCCLUSION_BINS[:-1], OCCLUSION_BINS[1:])):
        samples = live_jitter_by_occ[i]
        if samples:
            check(f"live jitter p95, occ {lo:.0%}-{min(hi, 1.0):.0%} (m)",
                  float(np.percentile(np.concatenate(samples), 95)),
                  envelope["live_jitter_m"])

    # Reported, not gated: the principal-axis angle is a pose-dependent
    # consequence of the depth spread above (a box lying along the shared
    # camera axis inflates in length, one lying across it has a different axis
    # inflated), so it diagnoses rather than judges.
    print(f"\nestimator '{estimator_name}' on {len(center_err)} static windows "
          f"(sqrtM Frobenius err mean {np.mean(frob_err) * 1e3:.2f} mm, "
          f"principal-axis angle mean {np.mean(axis_err):.1f} deg):")
    ok = True
    for label, value, bound, passed in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label:42s} "
              f"{value:.4f}  <= {bound:.4f}")
    print("\nACCEPTED — ship it" if ok else
          "\nREJECTED — escalate per plan decision 3 using the slices above")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Score a shape estimator offline")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--estimator", default="hull", choices=["hull"])
    parser.add_argument("--prompt", default="sponge")
    parser.add_argument("--sam2-model", default="tiny")
    args = parser.parse_args()
    return 0 if evaluate(args.dataset, args.estimator, args.prompt,
                         args.sam2_model) else 1


if __name__ == "__main__":
    raise SystemExit(main())
